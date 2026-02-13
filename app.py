import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LangSmith tracking
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT', 'SearchEngineToolAgent')

import streamlit as st
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# Arxiv and Wikipedia tools
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv_tool = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

search_tool = DuckDuckGoSearchRun(name="ddgsSearch")

tools = [wiki_tool, arxiv_tool, search_tool]

# LLM
llm = ChatOpenAI(model="gpt-5-nano-2025-08-07", streaming=True)

system_prompt = """
You are a helpful assistant. You have access to tools to search the web, wikipedia and arxiv. Use them only when needed.
When facing an issue with one tool, try another tool. Your answer should be anchored to the tools.
Always respond in the same language as the user. Don't stop when facing an issue with one tool.
If none of the tools work, summarize the situation and say you can't find the answer.
"""
search_agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)

# Streamlit UI
st.title("🔍 Langchain - Chat with Search")
st.markdown("""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [https://blog.langchain.com/langchain-streamlit/]
""")

# Sidebar for settings
st.sidebar.title("Settings")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Hi, I am a chatbot 🤖 that can search the web 🌐. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

if prompt := st.chat_input(placeholder="Ask me anything"):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Convert session messages to proper format
        input_messages = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages]

        config = {"configurable": {"thread_id": "chat_session"}}

        final_response = ""

        with st.spinner("Thinking..."):
            step = 0
            for chunk in search_agent.stream(
                {"messages": input_messages},
                config=config,
                stream_mode="values"
            ):
                latest_message = chunk["messages"][-1]
                step += 1

                if isinstance(latest_message, AIMessage):
                    if latest_message.content and latest_message.tool_calls:
                        with st.expander(f"💭 Step {step} : Reasoning", expanded=False):
                            st.markdown(latest_message.content)
                    if latest_message.tool_calls:
                        for tool_call in latest_message.tool_calls:
                            st.info(f"🔧 Step {step} - Calling **{tool_call['name']}** with: `{tool_call['args']}`")
                    if latest_message.content and not latest_message.tool_calls:
                        final_response = latest_message.content

                elif isinstance(latest_message, ToolMessage):
                    with st.expander(f"📄 Step {step} : Result from **{latest_message.name}**", expanded=False):
                        st.markdown(latest_message.content[:100])

        if final_response:
            st.session_state.messages.append({"role": "assistant", "content": final_response})
            st.markdown(final_response)
        else:
            st.session_state.messages.append({"role": "assistant", "content": "I'm sorry, I couldn't find any information about that."})
            st.markdown("I'm sorry, I couldn't find any information about that.")
