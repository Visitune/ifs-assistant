import json
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path

DATA_DIR = Path("ifs_assistant/data")
CORPUS_PATH = DATA_DIR / "corpus.json"
CHROMA_PATH = DATA_DIR / "chroma_db"
EMBED_MODEL = "intfloat/multilingual-e5-large"

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    corpus = json.load(f)

ifs_docs = [d for d in corpus if d.get("source") == "ifs"]
print(f"Indexing {len(ifs_docs)} IFS requirements...")

model = SentenceTransformer(EMBED_MODEL)
client = chromadb.PersistentClient(path=str(CHROMA_PATH))
ifs_collection = client.get_or_create_collection(name="ifs_requirements")

ids = []
embeddings = []
metadatas = []
documents = []

for i, doc in enumerate(ifs_docs):
    text_to_embed = f"passage: {doc.get('embed_text', '')}"
    embedding = model.encode(text_to_embed, show_progress_bar=False)
    ids.append(doc.get("id", f"ifs_{i}"))
    embeddings.append(embedding.tolist())
    metadatas.append({
        "req_number": doc.get("req_number", ""),
        "severity": doc.get("severity", ""),
        "ko_flag": doc.get("ko_flag", False),
        "source": "ifs"
    })
    documents.append(doc.get("embed_text", ""))

if ids:
    ifs_collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
    print("Done!")
