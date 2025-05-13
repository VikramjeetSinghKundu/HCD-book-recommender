# 📚 HCD   Book   Recommender

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

HCD-book-recommender/\
├── .devcontainer/\
├── .github/\
├── .gitignore/\
├── .streamlit/\
&nbsp;&nbsp;&nbsp;&nbsp;   ├── secrets.toml\
├── streamlit_app.py\
├── requirements.txt\
├── README.md               
             

Large data files are **not** stored in the repo; they are pulled on‑demand
from our Hugging Face dataset:

* [`books_processed.parquet`](https://huggingface.co/datasets/VikramjeetSingh/books-recs)  – 3.0 GB, 768‑d review vectors  
* `books_metadata_small.parquet` – 160 MB, author + category + snippet  

The app fetches them automatically the first time it runs and caches them.

---

## ⚡ Quick start (local)

> Tested on Python 3.10, macOS 13 and Codespaces.

# 1. Clone
git clone https://github.com/VikramjeetSinghKundu/HCD-book-recommender.git \
cd HCD-book-recommender

# 2. One-Liner bootstrap which exectues a script where we create a venv, install dependencies and handles common pitfalls and issues
./devsetup.sh           # ← one‑liner bootstrap

# 3. Activate the virtual environment
source .venv/bin/activate

# 5. Run the app
python -m streamlit run streamlit_app.py
