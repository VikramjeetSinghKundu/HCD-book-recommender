# ---------- app.py ----------
import streamlit as st
import pandas as pd, numpy as np, faiss
from sentence_transformers import SentenceTransformer

# ---------- 1.  Caching loader ----------
@st.cache_resource(hash_funcs={"_faiss.Index": id})
def load_store():
    df = pd.read_parquet("books_processed.parquet")
    vecs = np.stack(df.embeddings_proc.values).astype("float32")
    faiss.normalize_L2(vecs)                             # cosine via inner-prod
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    embed = SentenceTransformer("all-MiniLM-L6-v2")      # ← same model you used
    return df, index, embed

df, index, embedder = load_store()

# ---------- 2.  Streamlit UI ----------
st.set_page_config(page_title="Book Recommender", page_icon="📚")
st.title("📚 Book Recommender")

mode = st.radio("Recommend by…", ["Book title", "Natural-language query"])

if mode == "Book title":
    pick = st.selectbox(
        "Choose a book (sampled list for speed)",
        df.book_title.sample(5000).sort_values().tolist())
    if st.button("Find similar"):
        idx = df[df.book_title == pick].index[0]
        D, I = index.search(np.expand_dims(index.reconstruct(idx),0), 6)
        recs = df.iloc[I[0][1:]][["book_title","category"]].copy()
        recs["similarity"] = D[0][1:]
        st.dataframe(recs.reset_index(drop=True))

else:                   # free-text search
    query = st.text_area("Describe what you want to read")
    k     = st.slider("Results", 3, 10, 5)
    if st.button("Recommend"):
        qvec = embedder.encode(query, normalize_embeddings=True)
        D, I = index.search(np.expand_dims(qvec,0), k)
        recs = df.iloc[I[0]][["book_title","category"]].copy()
        recs["similarity"] = D[0]
        st.dataframe(recs.reset_index(drop=True))
