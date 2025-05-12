# app.py  ──────────────────────────────────────────────────────────────
"""
streamlit_app.py
📚 Conversational Book Recommender (full 768‑d embeddings)

Required files in HF dataset:
  • books_processed.parquet        (964 545 × 768 vectors)
  • books_metadata_small.parquet   (author, category, review_snippet…)
"""

# ────────────────────────── imports & config ──────────────────────────
import os, re, streamlit as st, torch, faiss
import pandas as pd, numpy as np
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download, InferenceClient

st.set_page_config(page_title="📚 Book Recommender", page_icon="📚")

HF_REPO   = "VikramjeetSingh/books-recs"
BIG_FILE  = "books_processed.parquet"          # 3 GB, 768‑d vectors
META_FILE = "books_metadata_small.parquet"     # 160 MB
LLM_MODEL = "HuggingFaceH4/zephyr-7b-beta"     # free endpoint (<10 GB)


# ────────────────────────── loaders (cached) ──────────────────────────
@st.cache_resource(hash_funcs={"_faiss.Index": id})
def load_data():
    vec_path  = hf_hub_download(HF_REPO, BIG_FILE,  repo_type="dataset")
    meta_path = hf_hub_download(HF_REPO, META_FILE, repo_type="dataset")

    # 1️⃣ read full vectors into memory  (~3 GB)
    big_df = pd.read_parquet(vec_path,
                             columns=["book_title","category","vec"])
    vecs = np.stack(big_df.vec.values).astype("float32")  # (N,768)
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(vecs.shape[1]); index.add(vecs)

    # 2️⃣ metadata frame
    meta_df = pd.read_parquet(meta_path).set_index(big_df.index)

    return big_df[["book_title","category"]], meta_df, vecs, index

@st.cache_resource
def load_longformer():
    tok = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
    mdl = AutoModel.from_pretrained(
              "allenai/longformer-base-4096",
              torch_dtype=torch.float16,        # half‑precision to save RAM
              low_cpu_mem_usage=True
          ).to("cpu").eval()
    return tok, mdl

titles_df, meta_df, vecs, index = load_data()
tokenizer, model                = load_longformer()

chat_client = InferenceClient(
    model   = LLM_MODEL,
    token   = st.secrets.get("HF_API_TOKEN", os.getenv("HF_API_TOKEN","")),
    timeout = 30
)

# ────────────────────────── helper functions ──────────────────────────
def embed_query(text: str) -> np.ndarray:
    with torch.no_grad():
        enc = tokenizer(text,
                        padding="longest",
                        truncation=True,
                        max_length=4096,
                        return_tensors="pt").to(model.device)
        hid  = model(**enc).last_hidden_state
        m    = enc.attention_mask.unsqueeze(-1).expand(hid.size()).float()
        vec  = (hid * m).sum(1) / m.sum(1).clamp(min=1e-9)
        vec  = torch.nn.functional.normalize(vec, p=2, dim=1)
    return vec.cpu().numpy()      # (1, 768)

_q_author  = re.compile(r"by ([\w\s'.-]+)", re.I)
_q_cat     = re.compile(r"in ([\w &]+)", re.I)

def parse_query(q: str):
    info = {"author": None, "category": None, "clean": q}
    m = _q_author.search(q)
    if m:
        info["author"] = m.group(1).strip()
        info["clean"]  = info["clean"].replace(m.group(0), "")
    m = _q_cat.search(info["clean"])
    if m:
        info["category"] = m.group(1).strip()
        info["clean"]    = info["clean"].replace(m.group(0), "")
    return info

def recommend(user_q: str, k: int = 5):
    info = parse_query(user_q)

    mask = np.ones(len(titles_df), dtype=bool)
    if info["author"]:
        mask &= meta_df.author.str.contains(info["author"], case=False, na=False)
    if info["category"]:
        mask &= meta_df.category.str.contains(info["category"], case=False, na=False)
    cand = np.where(mask)[0]
    if cand.size == 0: cand = np.arange(len(titles_df))

    q_vec = embed_query(info["clean"])
    D, I  = index.search(q_vec, len(cand))
    ids, sims = I[0], D[0]

    picked = [i for i in ids if i in cand][:k]
    sims   = sims[[np.where(ids==i)[0][0] for i in picked]]

    recs = meta_df.iloc[picked][["book_title","author","category","review_snippet"]].copy()
    recs["similarity"] = sims
    return recs.reset_index(drop=True)

def chatty(user_q: str, recs: pd.DataFrame) -> str:
    bullets = "\n".join(f"• {r.book_title} — {r.author or 'Unknown'}"
                        for r in recs.itertuples())
    prompt  = (
        "You are a friendly bookseller.\n\n"
        f"User request: {user_q}\n\n"
        f"Candidate books:\n{bullets}\n\n"
        "Explain in 3‑4 sentences why these match, "
        "and suggest one extra title.\n\nAssistant:"
    )
    return chat_client.text_generation(prompt,
                                       max_new_tokens=200,
                                       temperature=0.7,
                                       top_p=0.9).strip()

# ────────────────────────── UI ──────────────────────────
st.title("📚 Book Recommender")

query = st.text_input("Tell me what you’d like to read … "
                      "(e.g. 'Space Opera by Frank Herbert')")
top_k = st.slider("How many suggestions do you want?", 3, 10, 5)

if st.button("Recommend"):
    with st.spinner("Searching books…"):
        recs = recommend(query, top_k)
    st.dataframe(recs[["book_title","author","category","similarity"]])

    with st.spinner("Chatting with bookseller AI…"):
        st.write(chatty(query, recs))
