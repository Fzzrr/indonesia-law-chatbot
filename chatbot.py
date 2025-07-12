import os
from pdfminer.high_level import extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate  # Perbaikan import

def load_pdf_and_split(pdf_path):
    full_text = extract_text(pdf_path)
    doc = Document(page_content=full_text)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
    docs = text_splitter.split_documents([doc])
    return docs

def build_vectorstore(docs):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

def build_prompt_template():
    template = """
### System:
Anda adalah AI asisten hukum yang hanya akan menjawab pertanyaan terkait Undang-Undang Dasar Republik Indonesia Tahun 1945 dan topik hukum lainnya dalam dokumen ini. 
Jika pertanyaan diluar topik hukum jawab dengan "Tidak ada informasi yang relevan dalam dokumen ini."

### Context:
{context}

### Question:
{question}

### Response:
    """
    return PromptTemplate(input_variables=["context", "question"], template=template) # perbaiki typo

def main():
    pdf_path = "Undang-Undang Dasar Republik Indonesia Tahun 1945.pdf"
    
    # Load & split dokumen
    docs = load_pdf_and_split(pdf_path)
    
    # Build vektor store
    vector_store = build_vectorstore(docs)
    
    # Load LLM Ollama
    llm = OllamaLLM(model="deepseek-r1")
    
    # Build prompt template
    prompt = build_prompt_template()
    
    # Buat chain RetrievalQA dengan prompt yang sudah benar
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(),
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt, "document_variable_name": "context"}
    )
    
    print("Input keys for qa_chain:", qa_chain.input_keys)  # Debug: harus berisi ["query"]
    
    while True:
        query = input("Masukkan pertanyaan tentang UUD 1945 (atau ketik 'exit' untuk keluar): ")
        if query.lower() == "exit":
            break
        
        # Panggil chain dengan key 'query' sesuai input_variables chain expectation
        result = qa_chain.invoke({"query": query})
        print("Jawaban:", result["result"])
        
        # Jika ingin menampilkan sumber dokumen:
        # for doc in result["source_documents"]:
        #     print(f"Sumber: {doc.metadata.get('source')}")
    
if __name__ == "__main__":
    main()