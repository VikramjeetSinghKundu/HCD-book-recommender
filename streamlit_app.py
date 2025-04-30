"""
Streamlit app: Longformer-based Book Recommender + free HF-RAG chat
-------------------------------------------------------------------
Files expected in same folder:
  • books_processed.parquet   (parent_asin, book_title, category, embeddings_proc)
  • requirements.txt          (see README)
"""

import os, time, faiss, torch, numpy as np, pandas as pd, streamlit as st
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import InferenceClient   # free text-generation API

MODEL_ID = "allenai/longformer-base-4096"     # same model used offline
GEN_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"   # HF hosted, free

# ──────────────────────────────────────────────────────────────────────────
# 1. CACHED RESOURCES
# ──────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_longformer():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModel.from_pretrained(MODEL_ID).to("cpu").eval()
    return tok, mdl

@st.cache_resource(hash_funcs={"_faiss.Index": id})
def load_index():
    df   = pd.read_parquet("books_processed.parquet")
    vecs = np.stack(df.embeddings_proc.values).astype("float32")
    faiss.normalize_L2(vecs)                       # cosine via inner-product
    index = faiss.IndexFlatIP(vecs.shape[1]); index.add(vecs)
    return df, index, vecs

tokenizer, model = load_longformer()
df, index, all_vecs = load_index()

# optional chat client (works even without token but with strict rate limit)
hf_token = st.secrets.get("HF_API_TOKEN") or os.getenv("HF_API_TOKEN", "")
chat_client = InferenceClient(model=GEN_MODEL, token=hf_token, timeout=30)

# ──────────────────────────────────────────────────────────────────────────
# 2.  EMBEDDING HELPER  (Longformer mean-pool)
# ──────────────────────────────────────────────────────────────────────────
def embed_longformer(texts: list[str]) -> np.ndarray:
    with torch.no_grad():
        enc = tokenizer(
            texts, padding="longest", truncation=True,
            max_length=4096, return_tensors="pt"
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}
        out = model(**enc).last_hidden_state               # (B, L, 768)
        mask = enc["attention_mask"].unsqueeze(-1).expand(out.size()).float()
        vec  = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        vec  = torch.nn.functional.normalize(vec, p=2, dim=1)
    return vec.cpu().numpy()

# ──────────────────────────────────────────────────────────────────────────
# 3.  STREAMLIT UI
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="📚 Longformer Book Recommender", page_icon="📚")
st.title("📚 Book Recommender (Longformer + RAG Chat)")

mode = st.radio("Recommend by …", ["Book title", "Text query"])

if mode == "Book title":
    pick = st.selectbox(
        "Pick a book (sampled list for speed)",
        df.book_title.sample(5000).sort_values().tolist()
    )
    if st.button("Find similar"):
        row_idx = df[df.book_title == pick].index[0]
        sims, ids = index.search(all_vecs[row_idx:row_idx+1], 6)
        recs = df.iloc[ids[0][1:]][["book_title", "category"]].copy()
        recs["similarity"] = sims[0][1:]
        st.dataframe(recs.reset_index(drop=True))

        # Chatty explanation
        titles_block = "\n".join("• "+t for t in recs.book_title.tolist())
        with st.spinner("🗣️ Generating friendly explanation…"):
            reply = chat_client.text_generation(
                prompt=(
                  "You are a helpful bookseller.\n\n"
                  f"User chose: {pick}\n"
                  f"Here are similar books:\n{titles_block}\n\n"
                  "Describe why these suggestions fit.\n\nAssistant:"
                ),
                max_new_tokens=180, temperature=0.7, top_p=0.9
            )
        st.markdown("### Chatty answer")
        st.write(reply.strip())

else:   # Text query mode
    query = st.text_area("Describe what you’d like to read")
    k     = st.slider("Results", 3, 10, 5)
    if st.button("Recommend"):
        qvec = embed_longformer([query])
        sims, ids = index.search(qvec, k)
        recs = df.iloc[ids[0]][["book_title", "category"]].copy()
        recs["similarity"] = sims[0]
        st.dataframe(recs.reset_index(drop=True))

        titles_block = "\n".join("• "+t for t in recs.book_title.tolist())
        with st.spinner("🗣️ Generating friendly explanation…"):
            reply = chat_client.text_generation(
                prompt=(
                  "You are a helpful bookseller.\n\n"
                  f"User request: {query}\n"
                  f"Here are candidate books:\n{titles_block}\n\n"
                  "Explain why they match.\n\nAssistant:"
                ),
                max_new_tokens=180, temperature=0.7, top_p=0.9
            )
        st.markdown("### Chatty answer")
        st.write(reply.strip())

