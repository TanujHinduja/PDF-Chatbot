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
import io    
from PIL import Image 

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
        model_name="meta-llama/llama-4-scout-17b-16e-instruct"
    )
    
    # Read file stream
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    results = []
    full_extracted_text = "" 

    status_container = st.empty()
    status_container.info(f"Analyzing {len(doc)} page(s)...")

    for page_num, page in enumerate(doc):
        try:
            # --- STEP 1: RESIZE (Manage Dimensions) ---
            # Get default pixmap first to check size
            pix = page.get_pixmap()

            # If wider than 1024px, scale it down
            if pix.width > 1024:
                zoom = 1024 / pix.width
                matrix = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=matrix, alpha=False) # alpha=False forces RGB (no transparency)
            else:
                pix = page.get_pixmap(alpha=False) # alpha=False forces RGB

            # --- STEP 2: COMPRESS (Manage File Size) ---
            # Use PIL (Pillow) to handle the compression safely
            # 1. Create a PIL Image from the raw fitz data
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # 2. Save compressed JPEG to a memory buffer
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=70) # Standard PIL compression
            img_data = buffer.getvalue()
            
            # 3. Encode to Base64
            base64_image = base64.b64encode(img_data).decode("utf-8")

            # --- STEP 3: SEND TO AI ---
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
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            )

            response = vision_llm.invoke([msg])
            page_content = response.content
            
            results.append(f"**Page {page_num + 1}:**\n{page_content}")
            full_extracted_text += f"\n\n--- Page {page_num + 1} Handwriting Analysis ---\n{page_content}"

        except Exception as e:
            # Capture errors specifically per page so one bad page doesn't kill the whole app
            error_msg = f"Error on page {page_num + 1}: {str(e)}"
            print(error_msg) # Print to console for debugging
            results.append(error_msg)
    
    status_container.empty()
    return results, full_extracted_text

# --- MAIN APP ---
st.title("PDF Chatbot with Handwriting Analysis")
st.write("Upload Pdf's and chat with their content")


# --- API KEY LOADING ---
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
    st.stop()  # <--- This stops the app here until a key is entered


# Initialize Main LLM
llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile")

# Session State Setup
if 'store' not in st.session_state:
    st.session_state.store = {}

session_id = st.text_input("Session ID", value="default_session")

# PDF Uploader
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)


if uploaded_files:
    # Get the names of the currently uploaded files
    current_names = sorted([f.name for f in uploaded_files])
    
    # Check if these are different from what we processed last time
    if "processed_file_list" not in st.session_state or st.session_state.processed_file_list != current_names:
        st.info("🔄 New files detected! Clearing previous memory...")
        
        # 1. Delete the old VectorStore (The Brain)
        if "vectorstore" in st.session_state:
            del st.session_state.vectorstore
            
        # 2. Delete the old Chat History
        if "store" in st.session_state:
            st.session_state.store = {}
            
        # 3. Update the tracker to the new files
        st.session_state.processed_file_list = current_names


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
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(documents)
            
            if len(splits) > 0:
                st.session_state.vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
                st.success("Documents Processed! You can now chat below.")
            else:
                st.warning("Text was found but it was too short to process.")
        

    # 2. HANDWRITING ANALYSIS BUTTON
    with st.expander("📝 Analyze Handwritten Content"):
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

                    if 'text_splitter' not in locals():
                         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

                    vision_splits = text_splitter.split_documents([vision_doc])
                    
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
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 2})

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
