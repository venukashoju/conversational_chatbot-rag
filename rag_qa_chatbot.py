import os
from dotenv import load_dotenv
load_dotenv()
# os.environ['HF_TOKEN']=os.getenv('HF_TOKEN')
os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')

import streamlit as st
from langchain.chains import create_retrieval_chain,create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

hf_embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

st.title("Conversational RAG with PDF uploads and chat history")
st.write("Upload Pdf's and chat")

groq_api_key=st.text_input("Enter Your Groq API key",type='password')

if groq_api_key:
    llm=ChatGroq(model="openai/gpt-oss-20b",groq_api_key=groq_api_key)
    session_id=st.text_input("session_id",value="default_session")

    if 'store' not in st.session_state:
        st.session_state.store={}
    uploaded_files=st.file_uploader("Choose a pdf file",type='pdf',accept_multiple_files=True)
    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            temppdf=f"./temp.pdf"
            with open(temppdf,"wb") as f:
                f.write(uploaded_file.getvalue())
                file_name=uploaded_file.name
            
            loader = PyPDFLoader(temppdf)
            docs=loader.load()
            documents.extend(docs)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        splits=text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(splits,hf_embeddings)
        retriever= vectorstore.as_retriever()

        conextualize_q_system_prompt=(
            "Given a chat history and the latest user question"
            "which might references context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do not answer the question "
            "just reformulate it if needed and otherwise return it as is. "
        )
        contextualize_q_prompt =  ChatPromptTemplate.from_messages(
            [
                ("system",conextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ]
        )

        history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

        system_prompt=(
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer"
            "the question. If you don't know answer,say that you"
            "don't know. Use ten sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
            "\n\n"
        )
        
        qa_prompt= ChatPromptTemplate.from_messages(
            [
                ('system',system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ]
        )
        question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
        chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
        conversational_rag_chain=RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input=st.text_input("Your question")
        if user_input:
            session_history=get_session_history(session_id)
            response=conversational_rag_chain.invoke(
                {
                    "input":user_input},
                config={"configurable":{"session_id":session_id}
                },
            )
            st.write(st.session_state.store)
            st.success(f"Assistant: {response['answer']}")

            st.write("Chat History",session_history.messages)
