import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Get Response from AI
def get_ai_response(query, chat_history):
    llm = ChatGroq(
        model="llama-3.1-8b-instant"
    )
    template = '''
        You are an assistant, You have the knowledge and experience of making AI chatbots for the clients for their websites.
        
        History: {history}
        Query: {query}
    '''
    prompt = PromptTemplate.from_template(
        template=template
    )
    chain = prompt | llm | StrOutputParser()
    return chain.stream({"history":chat_history, "query":query})

def main():

    # Streamlit Configuration
    st.set_page_config(
        page_title="Streaming Bot", 
        page_icon="🤖",
    )
    st.title("Streaming Chatbot")

    # Initializing Streamlit Session State
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Showing previous messages
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("human"):
                st.markdown(message.content)
        else:
            with st.chat_message("ai"):
                st.markdown(message.content)

    # Showing current messages
    user_query = st.chat_input("Your message")
    if user_query is not None and user_query != "":
        st.session_state.chat_history.append(HumanMessage(user_query))
        with st.chat_message("human"):
            st.markdown(user_query)
        with st.chat_message("ai"):
            ai_response = st.write_stream(get_ai_response(user_query, st.session_state.chat_history))
        st.session_state.chat_history.append(AIMessage(ai_response))

if __name__ == "__main__":
    main()