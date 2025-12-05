import faiss
import numpy as np

def build_vectorstore(chunks, embedder):
    vectors = embedder.encode(chunks)
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index, vectors

def retrieve(query, index, vectors, chunks, embedder, top_k=3):
    q = embedder.encode([query])
    D, I = index.search(q, top_k)
    return [chunks[i] for i in I[0]]
