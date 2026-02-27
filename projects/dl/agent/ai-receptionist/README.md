# AI Receptionist Agent (LangGraph)

A simple AI receptionist built with **LangGraph** and **LangChain** that simulates how an AI system can manage conversations for a medical practice.

The agent can:

- Book appointments
- Answer basic questions
- Take messages

This project demonstrates a **structured conversational AI system** with controlled orchestration instead of a simple chatbot.

---

# Architecture

The conversation is orchestrated using a **LangGraph state machine**:

```
User Input → Intent Detection → Field Extraction → Tool Call → Response
```

Main components:

- **LangGraph** → conversation orchestration
- **LLM** → intent detection and field extraction
- **Tools** → appointment booking, FAQ, messages
- **State** → stores conversation data

---

# Project Structure

```
src/

main.py      # Runs the conversation loop
graph.py     # LangGraph orchestration
state.py     # Conversation state definition
prompt.py    # LLM prompts
tools.py     # Business logic tools
```

---

# Example

```
User:
I want an appointment on 01/04

Assistant:
Your appointment has been scheduled.
```

---

# Installation

Install dependencies:

```
pip install langgraph langchain langchain-community ollama
```

Start Ollama:

```
ollama run llama3.2
```

Run the agent:

```
python src/main.py
```

---

# Purpose

This project demonstrates:

- LLM orchestration with LangGraph
- Tool-based AI agents
- Structured conversation state
- Multi-turn conversations

It is designed as a minimal **production-style AI agent prototype**.