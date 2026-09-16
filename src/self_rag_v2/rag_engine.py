import os
from pathlib import Path
from typing import List, TypedDict, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, START, END

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

# -----------------------------
# 1. Models & Retriever Setup
# -----------------------------
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    google_api_key=GOOGLE_API_KEY,
)
llm = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    temperature=0,
    google_api_key=GOOGLE_API_KEY,
)

INDEX_DIR = PROJECT_ROOT / "faiss_index"

# Load documents (Ensure your PDFs are placed inside a ./documents directory)
pdf_paths = [
    PROJECT_ROOT / "documents" / "Company_Policies.pdf",
    PROJECT_ROOT / "documents" / "Company_Profile.pdf",
    PROJECT_ROOT / "documents" / "Product_and_Pricing.pdf",
]

if (INDEX_DIR / "index.faiss").exists():
    # Load the cached index instead of re-embedding
    vector_store = FAISS.load_local(
        str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
else:
    docs = []
    for path in pdf_paths:
        if os.path.exists(path):
            docs.extend(PyPDFLoader(path).load())

    if docs:
        chunks = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150).split_documents(docs)
        vector_store = FAISS.from_documents(chunks, embeddings)
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(str(INDEX_DIR))
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    else:
        retriever = None

# -----------------------------
# 2. Graph State & Schemas
# -----------------------------
class State(TypedDict):
    question: str
    retrieval_query: str
    rewrite_tries: int
    need_retrieval: bool
    docs: List[Document]
    relevant_docs: List[Document]
    context: str
    answer: str
    issup: Literal["fully_supported", "partially_supported", "no_support"]
    evidence: List[str]
    retries: int
    isuse: Literal["useful", "not_useful"]
    use_reason: str


class RetrieveDecision(BaseModel):
    should_retrieve: bool = Field(..., description="True if external documents are needed to answer reliably, else False.")

# class RelevanceDecision(BaseModel):
#     is_relevant: bool = Field(..., description="True ONLY if the document contains info that can directly answer the question.")

class RelevanceDecision(BaseModel):
    relevant_indices: List[int] = Field(
        default_factory=list,
        description="0-based indices of the documents (from the numbered list) that are relevant to the question. Empty list if none are relevant."
    )    

class IsSUPDecision(BaseModel):
    issup: Literal["fully_supported", "partially_supported", "no_support"]
    evidence: List[str] = Field(default_factory=list)

class IsUSEDecision(BaseModel):
    isuse: Literal["useful", "not_useful"]
    reason: str = Field(..., description="Short reason in 1 line.")

class RewriteDecision(BaseModel):
    retrieval_query: str = Field(..., description="Rewritten query optimized for vector retrieval against internal company PDFs.")


# -----------------------------
# 3. Node & Routing Definitions
# -----------------------------
should_retrieve_llm = llm.with_structured_output(RetrieveDecision)
relevance_llm = llm.with_structured_output(RelevanceDecision)
issup_llm = llm.with_structured_output(IsSUPDecision)
isuse_llm = llm.with_structured_output(IsUSEDecision)
rewrite_llm = llm.with_structured_output(RewriteDecision)

def decide_retrieval(state: State):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You decide whether retrieval is needed.\nReturn JSON with key: should_retrieve (boolean).\n\nGuidelines:\n- should_retrieve=True if answering requires specific facts from company documents.\n- should_retrieve=False for general explanations/definitions.\n- If unsure, choose True."),
        ("human", "Question: {question}")
    ])
    decision: RetrieveDecision = should_retrieve_llm.invoke(prompt.format_messages(question=state["question"]))
    return {"need_retrieval": decision.should_retrieve}

def route_after_decide(state: State) -> Literal["generate_direct", "retrieve"]:
    return "retrieve" if state["need_retrieval"] else "generate_direct"

def generate_direct(state: State):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer using only your general knowledge.\nIf it requires specific company info, say:\n'I don't know based on my general knowledge.'"),
        ("human", "{question}")
    ])
    out = llm.invoke(prompt.format_messages(question=state["question"]))
    return {"answer": out.content}

def retrieve(state: State):
    if not retriever:
        return {"docs": []}
    q = state.get("retrieval_query") or state["question"]
    return {"docs": retriever.invoke(q)}

# def is_relevant(state: State):
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", "You are judging document relevance at a TOPIC level.\nReturn JSON matching the schema.\n\nA document is relevant if it discusses the same entity or topic area as the question.\nIt does NOT need to contain the exact answer.\n\nWhen unsure, return is_relevant=true."),
#         ("human", "Question:\n{question}\n\nDocument:\n{document}")
#     ])
#     relevant_docs: List[Document] = []
#     for doc in state.get("docs", []):
#         decision: RelevanceDecision = relevance_llm.invoke(prompt.format_messages(question=state["question"], document=doc.page_content))
#         if decision.is_relevant:
#             relevant_docs.append(doc)
#     return {"relevant_docs": relevant_docs}

def is_relevant(state: State):
    docs = state.get("docs", [])
    if not docs:
        return {"relevant_docs": []}

    numbered_docs = "\n\n".join(
        f"[{i}]\n{doc.page_content}" for i, doc in enumerate(docs)
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are judging document relevance at a TOPIC level.\nReturn JSON matching the schema.\n\nA document is relevant if it discusses the same entity or topic area as the question.\nIt does NOT need to contain the exact answer.\n\nEach document below is numbered in [brackets]. Return the indices of ALL relevant documents.\nWhen unsure about a document, include it."),
        ("human", "Question:\n{question}\n\nDocuments:\n{documents}")
    ])
    decision: RelevanceDecision = relevance_llm.invoke(
        prompt.format_messages(question=state["question"], documents=numbered_docs)
    )
    relevant_docs = [docs[i] for i in decision.relevant_indices if 0 <= i < len(docs)]
    return {"relevant_docs": relevant_docs}

def route_after_relevance(state: State) -> Literal["generate_from_context", "no_answer_found"]:
    if state.get("relevant_docs") and len(state["relevant_docs"]) > 0:
        return "generate_from_context"
    return "no_answer_found"

def generate_from_context(state: State):
    context = "\n\n---\n\n".join([d.page_content for d in state.get("relevant_docs", [])]).strip()
    if not context:
        return {"answer": "No answer found.", "context": ""}
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a business RAG chatbot.\nTask: Answer the question based on the context. Don't mention that you are getting a context in your answer."),
        ("human", "Question:\n{question}\n\nContext:\n{context}")
    ])
    out = llm.invoke(prompt.format_messages(question=state["question"], context=context))
    return {"answer": out.content, "context": context}

def no_answer_found(state: State):
    return {"answer": "No answer found.", "context": ""}

def is_sup(state: State):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are verifying whether the ANSWER is supported by the CONTEXT.\nReturn JSON with keys: issup, evidence.\nissup must be one of: fully_supported, partially_supported, no_support."),
        ("human", "Question:\n{question}\n\nAnswer:\n{answer}\n\nContext:\n{context}\n")
    ])
    decision: IsSUPDecision = issup_llm.invoke(prompt.format_messages(question=state["question"], answer=state.get("answer", ""), context=state.get("context", "")))
    return {"issup": decision.issup, "evidence": decision.evidence}

MAX_RETRIES = 3
def route_after_issup(state: State) -> Literal["accept_answer", "revise_answer"]:
    if state.get("issup") == "fully_supported" or state.get("retries", 0) >= MAX_RETRIES:
        return "accept_answer"
    return "revise_answer"

def revise_answer(state: State):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a STRICT reviser.\nUse ONLY quotes from CONTEXT.\nFormat:\n- <direct quote from CONTEXT>\n- <direct quote from CONTEXT>"),
        ("human", "Question:\n{question}\n\nCurrent Answer:\n{answer}\n\nCONTEXT:\n{context}")
    ])
    out = llm.invoke(prompt.format_messages(question=state["question"], answer=state.get("answer", ""), context=state.get("context", "")))
    return {"answer": out.content, "retries": state.get("retries", 0) + 1}

def is_use(state: State):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are judging USEFULNESS of the ANSWER for the QUESTION.\nReturn JSON with keys: isuse, reason."),
        ("human", "Question:\n{question}\n\nAnswer:\n{answer}")
    ])
    decision: IsUSEDecision = isuse_llm.invoke(prompt.format_messages(question=state["question"], answer=state.get("answer", "")))
    return {"isuse": decision.isuse, "use_reason": decision.reason}

MAX_REWRITE_TRIES = 2
def route_after_isuse(state: State) -> Literal["END", "rewrite_question", "no_answer_found"]:
    if state.get("isuse") == "useful":
        return "END"
    if state.get("rewrite_tries", 0) >= MAX_REWRITE_TRIES:
        return "no_answer_found"
    return "rewrite_question"

def rewrite_question(state: State):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Rewrite the user's QUESTION into a query optimized for vector retrieval over INTERNAL company PDFs."),
        ("human", "QUESTION:\n{question}\n\nPrevious retrieval query:\n{retrieval_query}\n\nAnswer (if any):\n{answer}")
    ])
    decision: RewriteDecision = rewrite_llm.invoke(prompt.format_messages(question=state["question"], retrieval_query=state.get("retrieval_query", ""), answer=state.get("answer", "")))
    return {
        "retrieval_query": decision.retrieval_query,
        "rewrite_tries": state.get("rewrite_tries", 0) + 1,
        "docs": [],
        "relevant_docs": [],
        "context": "",
    }


# -----------------------------
# 4. Build and Compile Graph
# -----------------------------
g = StateGraph(State)

g.add_node("decide_retrieval", decide_retrieval)
g.add_node("generate_direct", generate_direct)
g.add_node("retrieve", retrieve)
g.add_node("is_relevant", is_relevant)
g.add_node("generate_from_context", generate_from_context)
g.add_node("no_answer_found", no_answer_found)
g.add_node("is_sup", is_sup)
g.add_node("revise_answer", revise_answer)
g.add_node("is_use", is_use)
g.add_node("rewrite_question", rewrite_question)

g.add_edge(START, "decide_retrieval")
g.add_conditional_edges("decide_retrieval", route_after_decide, {"generate_direct": "generate_direct", "retrieve": "retrieve"})
g.add_edge("generate_direct", END)
g.add_edge("retrieve", "is_relevant")
g.add_conditional_edges("is_relevant", route_after_relevance, {"generate_from_context": "generate_from_context", "no_answer_found": "no_answer_found"})
g.add_edge("no_answer_found", END)
g.add_edge("generate_from_context", "is_sup")
g.add_conditional_edges("is_sup", route_after_issup, {"accept_answer": "is_use", "revise_answer": "revise_answer"})
g.add_edge("revise_answer", "is_sup")
g.add_conditional_edges("is_use", route_after_isuse, {"END": END, "rewrite_question": "rewrite_question", "no_answer_found": "no_answer_found"})
g.add_edge("rewrite_question", "retrieve")

rag_app = g.compile()


# -----------------------------
# 5. Public Execution Wrapper
# -----------------------------
def run_self_rag_pipeline(question: str) -> dict:
    initial_state = {
        "question": question,
        "retrieval_query": "",
        "rewrite_tries": 0,
        "docs": [],
        "relevant_docs": [],
        "context": "",
        "answer": "",
        "issup": "no_support",
        "evidence": [],
        "retries": 0,
        "isuse": "not_useful",
        "use_reason": "",
    }
    result = rag_app.invoke(initial_state, config={"recursion_limit": 20})
    return {
        "answer": result.get("answer", ""),
        "evidence": result.get("evidence", []),
        "is_useful": result.get("isuse") == "useful",
        "rewrite_tries": result.get("rewrite_tries", 0)
    } 