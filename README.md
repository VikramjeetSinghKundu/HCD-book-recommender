# 📚 HCD Book Recommender

Longformer‑based dense retrieval + hybrid metadata scoring, wrapped in an
interactive Streamlit app.  
Enter any free‑text request—

*“space‑opera by Frank Herbert”*,
*“picture books about confidence building”*,
*“latest CPA exam prep”*—

and the system returns semantically similar Amazon books together with a
short, LLM‑generated “bookseller” explanation.

<p align="center">
  <img src="https://raw.githubusercontent.com/VikramjeetSinghKundu/HCD-book-recommender/main/docs/demo.gif"
       alt="demo animation" width="640"/>
</p>

---

## ✨ Key features
| Component | Details |
|-----------|---------|
| **Dense vectors** | 964 k Amazon books – 768‑d Longformer embeddings (review text) |
| **Hybrid boost** | Optional 768‑d metadata vector (title + author + category) |
| **Index** | FAISS `IndexFlatIP`, cosine search in \<100 ms |
| **Conversational layer** | RAG explanation from *HuggingFaceH4/zephyr‑7b‑beta* |
| **UI** | One‑file Streamlit app with caching, slider for *k* |

---

## 🗂 Repository structure

.
├── streamlit_app.py <- main app
├── requirements.txt <- Python dependencies
├── README.md <- you are here
└── docs/ <- screenshots / demo gif (optional)


Large data files are **not** stored in the repo; they are pulled on‑demand
from our Hugging Face dataset:

* [`books_processed.parquet`](https://huggingface.co/datasets/VikramjeetSingh/books-recs)  – 3.0 GB, 768‑d review vectors  
* `books_metadata_small.parquet` – 160 MB, author + category + snippet  

The app fetches them automatically the first time it runs and caches them under
`~/.cache/huggingface/`.

---

## ⚡ Quick start (local)

> Tested on Python 3.10, macOS 13 and Codespaces.

# 1. clone
git clone https://github.com/VikramjeetSinghKundu/HCD-book-recommender.git
cd HCD-book-recommender

# 2. create environment (conda or venv)
python -m venv .venv
source .venv/bin/activate

# 3. install deps
pip install -r requirements.txt
# ➜ ~1 GB download inc. Longformer / Zephyr weights

# 4. set HF token for Zephyr endpoint
export HF_API_TOKEN= <Token already included in the repository>

# 5. run
streamlit run streamlit_app.py
