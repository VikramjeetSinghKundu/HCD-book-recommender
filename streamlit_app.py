# app.py  ──────────────────────────────────────────────────────────────
import os, re, faiss, torch, streamlit as st
import pandas as pd, numpy as np
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import InferenceClient

# ─────────────── 1.  Cached loaders ───────────────
@st.cache_resource(hash_funcs={"_faiss.Index": id})
def load_data():
    # vectors
    vec_df = pd.read_parquet("books_processed.parquet")
    vecs   = np.stack(vec_df.vec.values).astype("float32")
    faiss.normalize_L2(vecs)
    index  = faiss.IndexFlatIP(vecs.shape[1]); index.add(vecs)
    # metadata
    meta_df = pd.read_parquet("books_metadata.parquet").set_index(vec_df.index)
    return vec_df, meta_df, vecs, index

@st.cache_resource
def load_longformer():
    tok  = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
    mdl  = AutoModel.from_pretrained("allenai/longformer-base-4096").to("cpu").eval()
    return tok, mdl

vec_df, meta_df, vecs, index = load_data()
tokenizer, model = load_longformer()

# free HF inference client (Mistral‑7B)
chat = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    token=os.getenv("HF_API_TOKEN", st.secrets.get("HF_API_TOKEN", "")),
    timeout=30
)

# ─────────────── 2.  Helpers ───────────────
def embed_longformer(texts:list[str]) -> np.ndarray:
    with torch.no_grad():
        enc = tokenizer(texts, padding="longest",
                        truncation=True, max_length=4096,
                        return_tensors="pt").to(model.device)
        hid  = model(**enc).last_hidden_state
        m    = enc.attention_mask.unsqueeze(-1).expand(hid.size()).float()
        vec  = (hid * m).sum(1) / m.sum(1).clamp(min=1e-9)
        vec  = torch.nn.functional.normalize(vec, p=2, dim=1)
    return vec.cpu().numpy()

def parse_query(q:str):
    out = {"author":None, "category":None, "clean":q}
    m = re.search(r"by ([\w\s'.-]+)", q, re.I)
    if m:
        out["author"] = m.group(1).strip()
        out["clean"]  = out["clean"].replace(m.group(0), "")
    m = re.search(r"in ([\w &]+)", out["clean"], re.I)
    if m:
        out["category"] = m.group(1).strip()
        out["clean"]    = out["clean"].replace(m.group(0), "")
    return out

def smart_recommend(user_q:str, k:int=5):
    info = parse_query(user_q)
    mask = np.ones(len(vec_df), dtype=bool)
    if info["author"]:
        mask &= meta_df.author.str.contains(info["author"], case=False, na=False)
    if info["category"]:
        mask &= meta_df.category.str.contains(info["category"], case=False, na=False)
    cand = np.where(mask)[0]
    if cand.size == 0: cand = np.arange(len(vec_df))

    qvec = embed_longformer([info["clean"]])
    D,I  = index.search(qvec, len(cand))
    ids, sims = I[0], D[0]

    # keep order & filter
    picked = [i for i in ids if i in cand][:k]
    sims   = sims[[np.where(ids==i)[0][0] for i in picked]]

    recs = meta_df.iloc[picked][["book_title","author","category"]].copy()
    recs["similarity"] = sims
    return recs.reset_index(drop=True), info

def explain(user_q, recs):
    titles = "\n".join(f"• {r.book_title} — {r.author or 'Unknown'}"
                       for r in recs.itertuples())
    prompt = (
      "You are a friendly bookseller.\n\n"
      f"User request: {user_q}\n"
      f"Candidate books:\n{titles}\n\n"
      "Explain briefly why these match, then suggest one other title.\n\nAssistant:"
    )
    ans = chat.text_generation(prompt, max_new_tokens=180,
                               temperature=0.7, top_p=0.9).strip()
    return ans

# ─────────────── 3.  Streamlit UI ───────────────
st.set_page_config(page_title="📚 Book Chat", page_icon="📚")
st.title("📚 Conversational Book Recommender")

q = st.text_input("Tell me what you’d like to read")
k = st.slider("How many suggestions?", 3, 10, 5)

if st.button("Recommend"):
    with st.spinner("Searching…"):
        recs, info = smart_recommend(q, k)
    st.dataframe(recs)

    with st.spinner("Generating explanation…"):
        chat_reply = explain(q, recs)
    st.markdown("### Bookseller says")
    st.write(chat_reply)
