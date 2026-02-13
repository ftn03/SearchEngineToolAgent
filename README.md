# Search Engine Tool Agent

A conversational AI chatbot built with **Streamlit** and **LangChain** that can search the web, Wikipedia, and Arxiv to answer your questions.

## Features

- **Multi-tool agent** powered by LangChain's ReAct architecture
- **DuckDuckGo Search** for general web queries
- **Wikipedia** for encyclopedic knowledge
- **Arxiv** for scientific papers and research
- **Chat interface** with conversation history
- **Transparent reasoning** — expandable steps show the agent's thinking process and tool calls

## Tech Stack

- **Streamlit** — Web UI and chat interface
- **LangChain / LangGraph** — Agent orchestration and tool management
- **OpenAI GPT** — Language model (easily swappable with Groq/Ollama)
- **LangSmith** — Optional tracing and monitoring

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/ftn03/SearchEngineToolAgent.git
cd SearchEngineToolAgent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy the example file and fill in your API keys:

```bash
cp .env.example .env
```

Required keys:

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key (for GPT models) |
| `GROQ_API_KEY` | Groq API key (optional, for Llama/Qwen models) |
| `LANGCHAIN_API_KEY` | LangSmith API key (optional, for tracing) |
| `LANGCHAIN_PROJECT` | LangSmith project name (optional) |

### 4. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## How It Works

1. The user asks a question in the chat input
2. The LangChain agent decides which tool(s) to use
3. Each step is displayed with expandable details (reasoning, tool calls, tool results)
4. The final answer is displayed in the chat and saved to conversation history

## Switching LLM Providers

Edit `app.py` to swap the model. Examples:

```python
# OpenAI
llm = ChatOpenAI(model="gpt-5-nano-2025-08-07", streaming=True)

# Groq (Llama)
llm = ChatGroq(model="llama-3.3-70b-versatile", streaming=True)
```
