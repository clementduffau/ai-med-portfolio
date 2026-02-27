# Medical RAG Agent (LangGraph + Ollama)

## Overview

This project implements a **Retrieval-Augmented Generation (RAG) medical assistant** that answers questions from a scientific article using a local language model and vector search.

The system retrieves relevant passages from a genomics research paper before generating answers, ensuring responses are **grounded in real scientific evidence** instead of relying only on the language model.

The knowledge base used in this project:

The system runs **entirely locally**, making it suitable for sensitive medical or research data.

---

## Architecture

Workflow:

-**User Question** → **LLM (Llama3.2)** → *retrieve_tool()** → **Chroma Vector** → **SearchRelevant** → **PassagesFinal** → **Answer with Citations**


The model retrieves information from the document before answering to reduce hallucinations and improve reliability.

---

## Technologies

- **LangGraph** – Agent workflow
- **LangChain** – Document processing
- **Ollama** – Local language model
- **ChromaDB** – Vector database

Models:


---

## Features

- Local medical AI assistant
- Retrieval-based answers
- Scientific document search
- Source citation
- Privacy-friendly (no API)
- Offline capable

---

## Medical Implications

This project demonstrates how AI systems can assist in **medical and genomics research** by improving access to scientific literature.

Potential applications include:

- Medical research assistants
- Clinical documentation search
- Genomics analysis support
- Evidence-based medical AI

Retrieval-based medical AI reduces the risk of hallucinated medical information and improves the reliability of AI-assisted decision support.

---

## Conclusion

This project shows how a **local RAG-based medical assistant** can provide reliable answers from scientific documents while maintaining data privacy.

Such systems can support researchers and healthcare professionals by providing fast access to evidence-based information and may contribute to safer and more transparent medical AI tools.