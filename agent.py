"""
agent.py — Shared agent module for Agentic AI Course Assistant
This file contains the CapstoneState, all node functions,
and the graph assembly. Imported by capstone_streamlit.py.
"""

import os
from typing import TypedDict, List

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

load_dotenv()

# ─────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────
PDF_FOLDER = "pdfs"   # folder containing your 12 PDF files

DOMAIN_NAME = "Agentic AI Course Assistant"
DOMAIN_DESCRIPTION = (
    "An AI-powered assistant that helps B.Tech students understand "
    "Agentic AI concepts using a knowledge base, memory, and intelligent "
    "tools like retrieval, comparison, and study planning."
)

KB_TOPICS = [
    "LLM_API_Agents", "Tool_Calling", "Memory_systems", "Embeddings",
    "LangChain", "LangGraph", "MultiAgent", "Autonomous_Agents",
    "RAG", "RagMemory", "Evaluation", "Deployment",
]


# ─────────────────────────────────────────────────────────
# STATE DEFINITION
# ─────────────────────────────────────────────────────────
class CapstoneState(TypedDict):
    # Input
    question: str
    # Memory
    messages: List[dict]
    # Routing
    route: str          # "retrieve" | "memory_only" | "tool"
    intent: str         # "concept" | "code" | "search" | "compare" | "plan"
    # RAG
    retrieved: str
    sources: List[str]
    # Tool
    tool_name: str
    tool_input: str
    tool_result: str
    # Answer
    answer: str
    # Quality control
    faithfulness: float
    eval_retries: int
    # Domain-specific
    code_snippet: str
    search_results: str


# ─────────────────────────────────────────────────────────
# MODEL + KB LOADER  (called once by Streamlit's @st.cache_resource)
# ─────────────────────────────────────────────────────────
def load_llm_and_kb():
    """
    Returns (llm, embedder, chroma_collection).
    Loads PDFs from the `pdfs/` folder, splits them, embeds, and
    stores in a fresh in-memory ChromaDB collection.
    """
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    embedder = DefaultEmbeddingFunction()

    client = chromadb.Client()
    try:
        client.delete_collection("capstone_kb")
    except Exception:
        pass
    collection = client.create_collection(
        "capstone_kb",
        embedding_function=embedder,
    )

    all_docs = []
    if os.path.isdir(PDF_FOLDER):
        for fname in os.listdir(PDF_FOLDER):
            if fname.endswith(".pdf"):
                path = os.path.join(PDF_FOLDER, fname)
                loader = PyPDFLoader(path)
                docs = loader.load()
                for d in docs:
                    d.metadata["topic"] = fname.replace(".pdf", "")
                all_docs.extend(docs)

    if not all_docs:
        raise RuntimeError(
            f"No PDFs found in '{PDF_FOLDER}/' folder. "
            "Please create a 'pdfs/' directory next to this file "
            "and add your course PDF files there."
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)

    texts = [c.page_content for c in chunks]
    metas = [c.metadata for c in chunks]

    collection.add(
        documents=texts,
        ids=[f"id_{i}" for i in range(len(texts))],
        metadatas=metas,
    )

    return llm, embedder, collection


# ─────────────────────────────────────────────────────────
# NODE FUNCTIONS
# ─────────────────────────────────────────────────────────
def make_nodes(llm, embedder, collection):
    """
    Returns a dict of node functions bound to the given llm/embedder/collection.
    This factory pattern lets Streamlit cache the heavy objects once and
    pass them into the node closures.
    """

    # ── Node 1: Memory ────────────────────────────────────
    def memory_node(state: CapstoneState) -> dict:
        msgs = state.get("messages", [])
        msgs = msgs + [{"role": "user", "content": state["question"]}]
        if len(msgs) > 6:
            msgs = msgs[-6:]
        # Clear stale fields from previous run so they don't bleed into this turn
        return {
            "messages": msgs,
            "answer": "",
            "retrieved": "",
            "sources": [],
            "tool_name": "",
            "tool_result": "",
            "route": "",
            "intent": "",
            "faithfulness": 0.0,
            "eval_retries": 0,
        }

    # ── Node 2: Router ────────────────────────────────────
    def router_node(state: CapstoneState) -> dict:
        question = state["question"]
        messages = state.get("messages", [])
        recent = "; ".join(
            f"{m['role']}: {m['content'][:60]}"
            for m in messages[-3:-1]
        ) or "none"

        prompt = f"""
You are a routing assistant for a Course Assistant chatbot for Agentic AI (B.Tech 4th year).

Decide the best action for answering the user's question.

Available options:
- retrieve    → for concepts from syllabus (RAG, LangGraph, embeddings, memory, tools, etc.)
- memory_only → if question refers to previous conversation (e.g., "what did you just say?")
- tool        → if special capability is needed
- chat        → if message is conversational (greetings, thanks, "ok", "good", "nice", acknowledgements)

Use tool when:
- question asks for code explanation     → use code tool
- question asks for comparison           → use compare tool
- question asks for study plan / roadmap → use plan tool
- question asks for latest/current info  → use web search tool

Recent conversation: {recent}
Current question: {question}

Reply with ONLY ONE WORD: retrieve OR memory_only OR tool OR chat
"""
        response = llm.invoke(prompt)
        decision = response.content.strip().lower()

        if "memory" in decision:
            decision = "memory_only"
        elif "tool" in decision:
            decision = "tool"
        elif "chat" in decision:
            decision = "chat"
        else:
            decision = "retrieve"

        return {"route": decision}

    # ── Node 3a: Intent classifier (used before tool node) ──
    def intent_node(state: CapstoneState) -> dict:
        question = state["question"]
        prompt = f"""
Classify the intent of the following question into ONE word:
- code    → if the user wants code explained
- compare → if the user wants to compare two concepts
- plan    → if the user wants a study plan or roadmap
- search  → if the user wants latest / current information

Question: {question}

Reply with only one word: code OR compare OR plan OR search
"""
        response = llm.invoke(prompt)
        intent = response.content.strip().lower()
        if intent not in ("code", "compare", "plan", "search"):
            intent = "search"
        return {"intent": intent}

    # ── Node 3b: Retrieval ────────────────────────────────
    def retrieval_node(state: CapstoneState) -> dict:
        results = collection.query(query_texts=[state["question"]], n_results=3)
        chunks = results["documents"][0]
        topics = [m.get("topic", "unknown") for m in results["metadatas"][0]]
        context = "\n\n---\n\n".join(
            f"[{topics[i]}]\n{chunks[i]}" for i in range(len(chunks))
        )
        return {"retrieved": context, "sources": topics}

    def skip_retrieval_node(state: CapstoneState) -> dict:
        return {"retrieved": "", "sources": []}

    # ── Node 4: Tool ─────────────────────────────────────
    def tool_node(state: CapstoneState) -> dict:
        question = state["question"]
        intent = state.get("intent", "search")

        tool_name = ""
        tool_result = ""

        if intent == "search":
            tool_name = "web_search"
            try:
                from ddgs import DDGS
                with DDGS() as ddgs:
                    results = list(ddgs.text(question, max_results=3))
                tool_result = "\n".join(
                    f"{r['title']}: {r['body'][:200]}" for r in results
                )
            except Exception as e:
                tool_result = f"Web search error: {e}"

        elif intent == "code":
            tool_name = "code_explainer"
            tool_result = (
                f"Explain the following code step-by-step:\n\n{question}\n\n"
                "Focus on:\n- What the code does\n- Key logic\n- Important concepts"
            )

        elif intent == "plan":
            tool_name = "study_plan"
            tool_result = (
                f"Create a structured study plan for: {question}\n\n"
                "Include:\n- Key topics\n- Order of learning\n"
                "- Timeline (days/weeks)\n- Practical steps"
            )

        elif intent == "compare":
            tool_name = "compare_concepts"
            tool_result = (
                f"Compare the following concepts clearly:\n\n{question}\n\n"
                "Include:\n- Differences\n- Use cases\n- When to use each"
            )

        else:
            tool_name = "none"
            tool_result = "No tool used."

        return {"tool_name": tool_name, "tool_result": tool_result}

    # ── Node 5: Answer ────────────────────────────────────
    def answer_node(state: CapstoneState) -> dict:
        question = state["question"]
        retrieved = state.get("retrieved", "")
        tool_result = state.get("tool_result", "")
        messages = state.get("messages", [])
        eval_retries = state.get("eval_retries", 0)

        history_text = ""
        for m in messages[:-1]:
            role = "Student" if m["role"] == "user" else "Assistant"
            history_text += f"{role}: {m['content']}\n"

        context_block = ""
        if retrieved:
            context_block = f"\n\nKnowledge Base Context:\n{retrieved}"
        if tool_result and tool_result != "No tool used.":
            context_block += f"\n\nTool Result:\n{tool_result}"

        retry_note = ""
        if eval_retries > 0:
            retry_note = (
                "\n[NOTE: Previous answer had low faithfulness. "
                "Stay strictly grounded in the provided context.]"
            )

        route = state.get("route", "retrieve")
        chat_note = ""
        if route == "chat":
            chat_note = "\n- If the message is conversational (greeting, thanks, acknowledgement), reply briefly and naturally in 1 sentence. Do NOT repeat previous answers."

        system_prompt = f"""You are an expert Agentic AI teaching assistant for B.Tech 4th year students.

Your role:
- FIRST check if the question is about Agentic AI, LLMs, LangChain, LangGraph, RAG, embeddings, memory, tools, agents, or deployment
- If the question is completely unrelated (e.g. HTML, cooking, sports), say ONLY: "This is outside the scope of this course. I can only help with Agentic AI topics."
- Otherwise answer ONLY from the provided knowledge base context
- Be concise and structured — avoid unnecessarily long answers
- Use bullet points or numbered lists only when genuinely helpful
- For comparisons, keep to 3-5 key differences maximum{chat_note}{retry_note}
"""
        user_prompt = f"""Conversation so far:
{history_text}

Student question: {question}{context_block}

Provide a clear, educational answer:"""

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        return {"answer": response.content.strip()}

    # ── Node 6: Evaluator ─────────────────────────────────
    def eval_node(state: CapstoneState) -> dict:
        answer = state.get("answer", "")
        retrieved = state.get("retrieved", "")
        retries = state.get("eval_retries", 0)

        if not retrieved:
            return {"faithfulness": 1.0, "eval_retries": retries}

        prompt = f"""Rate the faithfulness of this answer to the context.
Reply with ONLY a decimal number between 0.0 and 1.0.

Context: {retrieved[:500]}
Answer: {answer[:300]}
Faithfulness score:"""

        try:
            score = float(llm.invoke(prompt).content.strip().split()[0])
            score = max(0.0, min(1.0, score))
        except Exception:
            score = 0.5

        return {"faithfulness": score, "eval_retries": retries + 1}

    # ── Node 7: Memory updater ────────────────────────────
    def update_memory_node(state: CapstoneState) -> dict:
        msgs = state.get("messages", [])
        answer = state.get("answer", "")
        msgs = msgs + [{"role": "assistant", "content": answer}]
        if len(msgs) > 8:
            msgs = msgs[-8:]
        return {"messages": msgs}

    return {
        "memory_node": memory_node,
        "router_node": router_node,
        "intent_classifier_node": intent_node,
        "retrieval_node": retrieval_node,
        "skip_retrieval_node": skip_retrieval_node,
        "tool_node": tool_node,
        "answer_node": answer_node,
        "eval_node": eval_node,
        "update_memory_node": update_memory_node,
    }


# ─────────────────────────────────────────────────────────
# GRAPH ASSEMBLY
# ─────────────────────────────────────────────────────────
def build_agent(llm, embedder, collection):
    """
    Builds and compiles the LangGraph StateGraph.
    Returns the compiled agent app.
    """
    nodes = make_nodes(llm, embedder, collection)
    memory = MemorySaver()
    builder = StateGraph(CapstoneState)

    # Add nodes
    builder.add_node("memory_node",            nodes["memory_node"])
    builder.add_node("router_node",            nodes["router_node"])
    builder.add_node("intent_classifier_node", nodes["intent_classifier_node"])
    builder.add_node("retrieval_node",         nodes["retrieval_node"])
    builder.add_node("skip_retrieval_node", nodes["skip_retrieval_node"])
    builder.add_node("tool_node",           nodes["tool_node"])
    builder.add_node("answer_node",         nodes["answer_node"])
    builder.add_node("eval_node",           nodes["eval_node"])
    builder.add_node("update_memory_node",  nodes["update_memory_node"])

    # Entry point
    builder.set_entry_point("memory_node")
    builder.add_edge("memory_node", "router_node")

    # Routing logic
    def route_decision(state: CapstoneState) -> str:
        r = state.get("route", "retrieve")
        if r == "tool":
            return "intent_classifier_node"
        elif r == "memory_only" or r == "chat":
            return "skip_retrieval_node"
        else:
            return "retrieval_node"

    builder.add_conditional_edges(
        "router_node",
        route_decision,
        {
            "intent_classifier_node": "intent_classifier_node",
            "retrieval_node":         "retrieval_node",
            "skip_retrieval_node":    "skip_retrieval_node",
        },
    )

    # After intent_classifier → tool → answer
    builder.add_edge("intent_classifier_node", "tool_node")
    builder.add_edge("tool_node",           "answer_node")

    # After retrieval → answer
    builder.add_edge("retrieval_node",      "answer_node")
    builder.add_edge("skip_retrieval_node", "answer_node")

    # Eval → retry or finish
    builder.add_edge("answer_node", "eval_node")

    def eval_decision(state: CapstoneState) -> str:
        faith = state.get("faithfulness", 1.0)
        retries = state.get("eval_retries", 0)
        if faith < 0.6 and retries < 2:
            return "retrieval_node"          # re-retrieve and re-answer
        return "update_memory_node"

    builder.add_conditional_edges(
        "eval_node",
        eval_decision,
        {
            "retrieval_node":     "retrieval_node",
            "update_memory_node": "update_memory_node",
        },
    )

    builder.add_edge("update_memory_node", END)

    return builder.compile(checkpointer=memory)


# ─────────────────────────────────────────────────────────
# CONVENIENCE: ask() for notebook / testing use
# ─────────────────────────────────────────────────────────
def ask(agent_app, question: str, thread_id: str = "default") -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    result = agent_app.invoke({"question": question}, config=config)
    return result