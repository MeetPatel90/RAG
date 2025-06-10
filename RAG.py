# Install: pip install streamlit sentence-transformers groq PyMuPDF faiss-cpu

import streamlit as st
import fitz
from sentence_transformers import SentenceTransformer
import groq
import numpy as np
import faiss
import pickle
import os

st.title("📚 Simple RAG with FAISS")
st.write("Upload PDF/Text → Ask Questions → Get Smart Answers!")

# API Key
api_key = st.text_input("🔑 Groq API Key:", type="password")
st.caption("Get free key: https://console.groq.com/")

# File Upload
uploaded_file = st.file_uploader(
    "📄 Upload Document",
    type=['pdf', 'txt']
)

if uploaded_file and api_key:


    with st.spinner("🔍 Reading document..."):
        if uploaded_file.name.endswith('.pdf'):

            pdf_bytes = uploaded_file.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")

            clean_text = ""
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                clean_text += page.get_text() + "\n\n"

            doc.close()

        else:

            clean_text = str(uploaded_file.read(), "utf-8")

    st.success(f"✅ Extracted {len(clean_text):,} characters of clean text")

    # Show first bit of text
    with st.expander("📋 Preview of extracted text:"):
        st.text_area("First 500 characters:", clean_text[:500], height=150)

    # STEP 2: Break into chunks
    chunk_size = 400
    chunks = []
    for i in range(0, len(clean_text), chunk_size):
        chunk = clean_text[i:i+chunk_size]
        if chunk.strip():  # Skip empty chunks
            chunks.append(chunk)

    st.write(f"📝 Created {len(chunks)} chunks")

    # STEP 3: Create embeddings
    @st.cache_resource
    def load_embedding_model():
        return SentenceTransformer('BAAI/bge-large-en-v1.5')

    model = load_embedding_model()

    with st.spinner("🧠 Converting to embeddings..."):
        embeddings = model.encode(chunks, show_progress_bar=True)

    st.write("✅ Created embeddings")

    # STEP 4: Store in FAISS (Python 3.8 compatible!)
    @st.cache_data
    def setup_faiss_database(_embeddings, _chunks):
        # Convert embeddings to numpy array
        embeddings_array = np.array(_embeddings).astype('float32')

        # Create FAISS index
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for similarity

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings_array)

        # Add embeddings to index
        index.add(embeddings_array)

        # Create metadata
        metadata = []
        for i, chunk in enumerate(_chunks):
            metadata.append({
                "chunk_id": i,
                "source": uploaded_file.name,
                "text": chunk
            })

        return index, metadata, embeddings_array

    index, metadata, embeddings_array = setup_faiss_database(embeddings, chunks)
    st.write("💾 Stored in FAISS index")

    # STEP 5: Ask Questions!
    st.header("❓ Ask Your Question")
    question = st.text_area("What do you want to know?", height=100)

    if question:
        with st.spinner("🔍 Searching for relevant information..."):
            # Convert question to embedding
            question_embedding = model.encode([question])
            question_embedding = np.array(question_embedding).astype('float32')

            # Normalize for cosine similarity
            faiss.normalize_L2(question_embedding)

            # Search index
            k = 3  # Number of results
            scores, indices = index.search(question_embedding, k)

            # Get relevant chunks and metadata
            relevant_chunks = []
            relevant_metadata = []
            for i, idx in enumerate(indices[0]):
                if idx < len(metadata):
                    relevant_chunks.append(metadata[idx]["text"])
                    relevant_metadata.append({
                        "chunk_id": metadata[idx]["chunk_id"],
                        "source": metadata[idx]["source"],
                        "score": scores[0][i]
                    })

            context = "\n\n".join(relevant_chunks)

        # Generate answer with Groq
        with st.spinner("🤖 Generating answer..."):
            client = groq.Groq(api_key=api_key)

            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. Answer questions based on the provided context. Be accurate and cite specific information when possible."
                    },
                    {
                        "role": "user",
                        "content": f"""Context from document:

{context}

Question: {question}

Please provide a comprehensive answer based on the context above."""
                    }
                ],
                temperature=0.1
            )

        # Display answer
        st.subheader("🎯 Answer:")
        st.write(response.choices[0].message.content)

        # Show sources
        with st.expander("📚 Sources Used"):
            for i, (chunk, meta) in enumerate(zip(relevant_chunks, relevant_metadata)):
                similarity = meta["score"] * 100
                st.write(f"**Source {i+1}** (Similarity: {similarity:.1f}%)")
                st.text_area(f"Content {i+1}", chunk, height=150, key=f"source_{i}")

else:
    st.info("👆 Upload a document and enter your API key to get started!")
