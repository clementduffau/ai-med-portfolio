from typing import Annotated, Sequence, TypedDict

from langchain_ollama import ChatOllama, OllamaEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.messages import (
    BaseMessage,
    ToolMessage,
    SystemMessage,
    HumanMessage,
)

from langchain_core.tools import tool

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END

# -----------------------
# CONFIG
# -----------------------
PDF_PATH = "Borzoi_genomics.pdf"              
PERSIST_DIRECTORY = "./chroma_db"          
COLLECTION_NAME = "Borzoi_genomics"

LLM_MODEL = "llama3.2"
EMBED_MODEL = "nomic-embed-text"         


# -----------------------
# LOAD + SPLIT
# -----------------------
pdf_loader = PyPDFLoader(PDF_PATH)
pages = pdf_loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
pages_split = text_splitter.split_documents(pages)


# -----------------------
# EMBEDDINGS + VECTORSTORE
# -----------------------
embeddings = OllamaEmbeddings(model=EMBED_MODEL)

vectorstore = Chroma.from_documents(
    documents=pages_split,
    embedding=embeddings,
    persist_directory=PERSIST_DIRECTORY if PERSIST_DIRECTORY else None,
    collection_name=COLLECTION_NAME,
)

# if PERSIST_DIRECTORY:
#     vectorstore.persist()

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},
)


# -----------------------
# TOOL
# -----------------------
@tool
def retrieve_tool(query: str) -> str:
    """Retrieve relevant passages from the article based on the user's query."""
    docs = retriever.invoke(query)
    if not docs:
        return "I found nothing relevant."

    results = []
    for i, doc in enumerate(docs):
        # include metadata when available (page number often exists)
        page = doc.metadata.get("page", None)
        page_info = f"(page {page})" if page is not None else ""
        results.append(f"Document {i+1} {page_info}:\n{doc.page_content}")
    return "\n\n".join(results)


tools = [retrieve_tool]
tools_dict = {t.name: t for t in tools}


# -----------------------
# LLM (tool-bound)
# -----------------------
llm = ChatOllama(model=LLM_MODEL, temperature=0).bind_tools(tools)


# -----------------------
# LANGGRAPH STATE
# -----------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def should_continue(state: AgentState) -> bool:
    last = state["messages"][-1]
    return hasattr(last, "tool_calls") and len(last.tool_calls) > 0


system_prompt = """
You are an intelligent AI assistant who answers questions about a health article.
Use the retrieve_tool to look up relevant parts of the article before answering.
When you answer, cite the specific excerpts you used (quote short snippets) and mention document numbers/pages if available.
"""


def call_llm(state: AgentState) -> AgentState:
    messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
    message = llm.invoke(messages)
    # IMPORTANT: must return as "messages": [message]
    return {"messages": [message]}


def take_action(state: AgentState) -> AgentState:
    tool_calls = state["messages"][-1].tool_calls
    results = []

    for t in tool_calls:
        name = t.get("name")
        args = t.get("args") or {}

        query = ""
        if isinstance(args, dict):
            query = args.get("query", "")
        elif isinstance(args, str):
            query = args

        if not isinstance(query, str) or not query.strip():
            query = next(
                (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
                ""
            )

        print(f"Calling Tool: {name} with query: {query!r}")

        if name not in tools_dict:
            result = "Incorrect tool name. Please retry with an available tool."
        else:
            result = tools_dict[name].invoke({"query": query})

        results.append(
            ToolMessage(
                tool_call_id=t["id"],
                name=name,
                content=str(result),
            )
        )

    return {"messages": results}

# -----------------------
# BUILD GRAPH
# -----------------------
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "retriever_agent", False: END},
)
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()


# -----------------------
# RUN LOOP
# -----------------------
def running_agent():
    print("\n=== RAG AGENT ===")
    while True:
        user_input = input("\nWhat is your question (type 'exit' to quit): ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        result = rag_agent.invoke({"messages": [HumanMessage(content=user_input)]})
        print("\n=== ANSWER ===")
        print(result["messages"][-1].content)


if __name__ == "__main__":
    running_agent()