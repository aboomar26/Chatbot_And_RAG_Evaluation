# 🧪 LLM Evaluation Framework

A collection of automated evaluation pipelines for LLM-based systems using the **LLM-as-a-Judge** technique, tracked and visualized with **LangSmith**.

---

## 📦 What's in This Repo?

| Project | Description | Metrics |
|---|---|---|
| [`chatbot-evaluation/`](./chatbot-evaluation/) | Evaluates a standard chatbot (single-turn Q&A) | Correctness, Concision |
| [`rag-evaluation/`](./rag-evaluation/) | Evaluates a full RAG pipeline | Correctness, Answer Relevance, Groundedness, Retrieval Relevance |

---

## 🧠 Core Concept: LLM-as-a-Judge

Both projects share the same evaluation philosophy — instead of manually grading responses, a **judge LLM** automatically scores them.

```
Question + Reference Answer + Chatbot Response → Judge LLM → CORRECT / INCORRECT
```

| Approach | Scale | Cost | Consistency |
|---|---|---|---|
| Human review | Low | High | Variable |
| Rule-based checks | High | Low | High (but limited) |
| LLM-as-a-Judge | High | Medium | High |

---

## 🔀 Which Project Should I Use?

```
Is your system just answering questions directly from model knowledge?
    ↓ YES → chatbot-evaluation/
    
Does your system retrieve documents before generating an answer (RAG)?
    ↓ YES → rag-evaluation/
```

**Use `chatbot-evaluation/`** when you want to:
- Evaluate a simple Q&A bot
- Check factual correctness and response length
- Get started quickly with 2 metrics

**Use `rag-evaluation/`** when you want to:
- Evaluate a full RAG pipeline end-to-end
- Debug *where* failures happen (retriever vs. generator)
- Catch hallucinations with Groundedness metric
- Run 4 metrics for comprehensive coverage

---

## 🛠️ Shared Tech Stack

| Component | Tool |
|---|---|
| LLM (Chatbot + Judge) | `Qwen/Qwen2.5-7B-Instruct` |
| Model Loading | `transformers` (HuggingFace) |
| LLM Framework | `LangChain` + `langchain_huggingface` |
| Evaluation & Tracing | `LangSmith` |
| Runtime | Google Colab (GPU — T4 or better) |

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install --pre -U langchain langchain-openai langchain_community \
    langchain_core langchain_text_splitters unstructured \
    langchain_huggingface langchain_cohere
```

### 2. Set up LangSmith

```python
from google.colab import userdata
import os

os.environ['LANGCHAIN_TRACING'] = 'true'
os.environ['LANGCHAIN_API_KEY'] = userdata.get('LANGSMITH')
```

### 3. Open the notebook for your use case

- 📓 `chatbot-evaluation/Chatbot_Evaluation.ipynb`
- 📓 `rag-evaluation/RAG_Evaluation.ipynb`

---

## 📁 Project Structure

```
chatbot-and-rag-evaluation/
│
├── Chatbot_Evaluation/
│   ├── Chatbot_Evaluation.ipynb
│   └── README.md
│
├── RAG_Evaluation/
│   ├── RAG_Evaluation.ipynb
│   └── README.md
│
└── README.md
```


---

## 📈 Results & Tracking

All experiments are logged to **LangSmith**, giving you:
- Per-example scores for each metric
- Aggregate pass rates across your dataset
- Side-by-side experiment history to compare model versions

View results at: [smith.langchain.com](https://smith.langchain.com)