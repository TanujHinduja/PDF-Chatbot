## RAG Q&A Conversation With PDF- HANDWRITTEN also Including Chat History

import streamlit as st
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
import fitz  # PyMuPDF
import base64
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_core.documents import Document


if "HF_TOKEN" in st.secrets:
    os.environ['HF_TOKEN'] = st.secrets["HF_TOKEN"]
else:
    # Fallback only if secrets are missing (prevents the NoneType crash)
    os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN", "")

# Force the model to run on CPU to avoid the "meta tensor" error
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

def analyze_handwritten_pdf(uploaded_file, api_key):
    vision_llm = ChatGroq(
        groq_api_key=api_key, 
        model_name="meta-llama/llama-4-scout-17b-16e-instruct" # Recommended vision model (check your available models)
    )
    
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    results = []
    full_extracted_text = "" 

    status_container = st.empty()
    status_container.info(f"Analyzing {len(doc)} page(s)...")

    for page_num, page in enumerate(doc):
        # 1. Downscale the image to reduce dimensions (e.g., 150 DPI is usually enough for handwriting)
        # Standard PDF is 72 DPI. Matrix(2,2) = 144 DPI. If it's too big, use Matrix(1,1) or (0.5, 0.5)
        zoom = 1.0  # Adjust this: 0.5 reduces size by 4x, 1.0 is standard
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        # 2. Convert to JPEG instead of PNG and reduce quality
        # PNG is lossless and huge. JPEG with quality=70 is much smaller.
        img_data = pix.tobytes("jpeg", jpg_quality=70)
        
        base64_image = base64.b64encode(img_data).decode("utf-8")

        prompt = (
            "Transcribe ALL handwritten text on this page accurately. "
            "Then, grade the answers found. "
            "Return the output as plain text."
        )

        msg = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url", 
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}" # Note: Changed png to jpeg here too
                    }
                }
            ]
        )

        try:
            response = vision_llm.invoke([msg])
            page_content = response.content
            
            results.append(f"**Page {page_num + 1}:**\n{page_content}")
            full_extracted_text += f"\n\n--- Page {page_num + 1} Handwriting Analysis ---\n{page_content}"
            
        except Exception as e:
            results.append(f"Error on page {page_num + 1}: {str(e)}")
    
    status_container.empty()
    return results, full_extracted_text


# MAIN APP 
st.title("PDF Chatbot with Handwriting Analysis")
st.write("Upload Pdf's and chat with their content")


# API KEY LOADING
api_key = None

# 1. Try to get the key from Streamlit Secrets
if "GROQ_API_KEY" in st.secrets:
    api_key = st.secrets["GROQ_API_KEY"]
else:
    # 2. If not in secrets, ask the user
    api_key = st.text_input("Enter your Groq API key:", type="password")

# 3. CRITICAL FIX: Stop the app if the key is missing
if not api_key:
    st.info("⚠️ Please enter your Groq API key above to continue.")
    st.stop()  # stops the app here until a key is entered


# Initialize Main LLM
llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile")

# Session State Setup
if 'store' not in st.session_state:
    st.session_state.store = {}

session_id = st.text_input("Session ID", value="default_session")

# PDF Uploader
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

# 1. PROCESS PDFS (Only runs once per upload)
if uploaded_files:
    if "vectorstore" not in st.session_state:
        st.info("Processing PDFs into database... please wait.")
        documents = []
        
        # 1. Load the documents
        for uploaded_file in uploaded_files:
            temppdf = f"./temp.pdf"
            uploaded_file.seek(0)
            with open(temppdf, "wb") as file:   
                file.write(uploaded_file.getvalue()) 
            
            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)

        # 2. Check if text was actually found
        if len(documents) == 0 or not any(doc.page_content.strip() for doc in documents):
            st.warning("⚠️ No digital text found! This appears to be a handwritten or scanned PDF. You can use the 'Analyze Handwriting' button above to extract handwritten text.")
        else:
            # Only proceed to splitting and embedding if text exists
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
            splits = text_splitter.split_documents(documents)
            
            if len(splits) > 0:
                st.session_state.vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
                st.success("Documents Processed! You can now chat below.")
            else:
                st.warning("Text was found but it was too short to process.")
        

    # 2. HANDWRITING ANALYSIS BUTTON
    with st.expander("📝 Analyze Handwritten Content (Vision AI)"):
        if st.button("Run Handwriting Analysis"):
            for uploaded_file in uploaded_files:
                st.write(f"Analyzing: {uploaded_file.name}")
                uploaded_file.seek(0)
                
                # Get the report AND the raw text
                reports, raw_vision_text = analyze_handwritten_pdf(uploaded_file, api_key)
                
                # 1. Show the report to the user
                for report in reports:
                    st.markdown(report)
                    st.divider()
                
                # 2. THE BRIDGE: Inject this text into the Chat Database
                if raw_vision_text:
                    # Create a LangChain Document
                    vision_doc = Document(
                        page_content=raw_vision_text, 
                        metadata={"source": f"handwriting_{uploaded_file.name}"}
                    )
                    
                    # Add to VectorStore
                    if "vectorstore" not in st.session_state:
                        # Create new store if one doesn't exist (e.g. only scanned PDF uploaded)
                        st.session_state.vectorstore = Chroma.from_documents(
                            documents=[vision_doc], 
                            embedding=embeddings
                        )
                    else:
                        # Add to existing store
                        st.session_state.vectorstore.add_documents([vision_doc])
                        
                    st.success("✅ Handwriting analysis saved to memory! You can now ask questions about it below.")

    # 3. CHAT INTERFACE
    # Ensure we have a vectorstore before trying to chat
    if "vectorstore" in st.session_state:
        retriever = st.session_state.vectorstore.as_retriever()

        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session:str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("Your question:")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            )
            st.write("Assistant:", response['answer'])
            
            with st.expander("View Chat History"):
                st.write(session_history.messages)