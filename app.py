import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from PyPDF2 import PdfReader

# تنظیمات صفحه
st.set_page_config(page_title="Chat with PDF (Ollama + FAISS)", layout="wide")
st.title("📄 Chat with your PDF (Offline + Local AI)")

# مرحله ۱: آپلود PDF
uploaded_file = st.file_uploader("Upload a PDF file:", type=["pdf"])

if uploaded_file:
    with st.spinner("Reading your PDF..."):
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""

    # تقسیم متن به تکه‌ها
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
    docs = [Document(page_content=chunk) for chunk in text_splitter.split_text(text)]

    # مرحله ۲: ساخت embedding
    with st.spinner("Creating embeddings with Ollama... (first time may take a minute)"):
        embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434")  # مطمئن شو Ollama همین پورت رو داره
                    
        vectorstore = FAISS.from_documents(docs, embeddings)
    
          # یا llama3 برای تست

    # مرحله ۳: ساخت مدل چت
    llm = ChatOllama(model="llama3")
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    # حافظه چت
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.subheader("💬 Ask questions about your PDF")
    user_input = st.text_input("Type your question:")

    if user_input:
        with st.spinner("Thinking..."):
            result = qa_chain({
                "question": user_input,
                "chat_history": st.session_state.chat_history
            })
            answer = result["answer"]
            st.session_state.chat_history.append((user_input, answer))

            st.markdown(f"**🧠 You:** {user_input}")
            st.markdown(f"**🤖 Bot:** {answer}")

    # نمایش تاریخچه
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### Chat History")
        for q, a in reversed(st.session_state.chat_history):
            st.markdown(f"**You:** {q}")
            st.markdown(f"**Bot:** {a}")
