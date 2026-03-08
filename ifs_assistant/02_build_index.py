"""
02_build_index.py
==================
Construit l'index vectoriel avec ChromaDB.
- Charge le corpus.json
- Encode les textes avec multilingual-e5-large (préfixe "passage: ")
- Sauvegarde dans ChromaDB
"""

import json
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Paths
DATA_DIR = Path("ifs_assistant/data")
CORPUS_PATH = DATA_DIR / "corpus.json"
CHROMA_PATH = DATA_DIR / "chroma_db"

# Modèle d'embedding (multilingual-e5-large selon la spec)
EMBED_MODEL = "intfloat/multilingual-e5-large"

print("=== IFS Food v8 - Construction de l'index vectoriel ===\n")

# ============================================================
# 1. Charger le corpus
# ============================================================
print("1. Chargement du corpus...")
with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    corpus = json.load(f)

print(f"   -> {len(corpus)} documents charges")

# Separer les types de documents
csv_docs = [d for d in corpus if d.get("source") == "csv"]
ifs_docs = [d for d in corpus if d.get("source") == "ifs"]

print(f"   -> {len(csv_docs)} documents CSV (cas de suspensions)")
print(f"   -> {len(ifs_docs)} documents IFS (referentiel pur)")

# ============================================================
# 2. Charger le modèle d'embedding
# ============================================================
print(f"\n2. Chargement du modele d'embedding: {EMBED_MODEL}")
print("   (premier lancement - telechargement si necessaire)")

model = SentenceTransformer(EMBED_MODEL)
print("   -> Modele charge")

# ============================================================
# 3. Encoder et indexer les documents CSV
# ============================================================
print("\n3. Indexation des cas de suspensions (CSV)...")

# Initialiser ChromaDB
client = chromadb.PersistentClient(path=str(CHROMA_PATH))

# Collection pour les cas CSV
csv_collection = client.get_or_create_collection(
    name="ifs_suspensions",
    metadata={"description": "Cas de suspensions IFS Food"}
)

# Preparer les donnees pour l'indexation
csv_ids = []
csv_embeddings = []
csv_metadatas = []
csv_documents = []

for i, doc in enumerate(csv_docs):
    # IMPORTANT: Prefixe "passage: " pour E5 (spec ligne 531)
    text_to_embed = f"passage: {doc.get('embed_text', '')}"
    
    embedding = model.encode(text_to_embed, show_progress_bar=False)
    
    csv_ids.append(doc.get("id", f"csv_{i}"))
    csv_embeddings.append(embedding.tolist())
    csv_metadatas.append({
        "req_number": doc.get("req_number", ""),
        "severity": doc.get("severity", ""),
        "ko_flag": doc.get("ko_flag", False),
        "source": "csv",
        "supplier": doc.get("supplier", ""),
        "country": doc.get("country", ""),
        "lock_date": doc.get("lock_date", "N/A"),
        "product_scope": doc.get("product_scope", "")
    })
    csv_documents.append(doc.get("embed_text", ""))

# Ajouter au vecteur store
if csv_ids:
    csv_collection.upsert(
        ids=csv_ids,
        embeddings=csv_embeddings,
        metadatas=csv_metadatas,
        documents=csv_documents
    )

print(f"   -> {len(csv_ids)} cas de suspensions indexes")

# ============================================================
# 4. Encoder et indexer les exigences IFS pures
# ============================================================
print("\n4. Indexation du referentiel IFS (exigences pures)...")

# Collection pour les exigences IFS
ifs_collection = client.get_or_create_collection(
    name="ifs_requirements",
    metadata={"description": "Exigences pures du referentiel IFS Food"}
)

ifs_ids = []
ifs_embeddings = []
ifs_metadatas = []
ifs_documents = []

for i, doc in enumerate(ifs_docs):
    # IMPORTANT: Prefixe "passage: " pour E5
    text_to_embed = f"passage: {doc.get('embed_text', '')}"
    
    embedding = model.encode(text_to_embed, show_progress_bar=False)
    
    ifs_ids.append(doc.get("id", f"ifs_{i}"))
    ifs_embeddings.append(embedding.tolist())
    ifs_metadatas.append({
        "req_number": doc.get("req_number", ""),
        "severity": doc.get("severity", ""),
        "ko_flag": doc.get("ko_flag", False),
        "source": "ifs"
    })
    ifs_documents.append(doc.get("embed_text", ""))

# Ajouter au vecteur store
if ifs_ids:
    ifs_collection.upsert(
        ids=ifs_ids,
        embeddings=ifs_embeddings,
        metadatas=ifs_metadatas,
        documents=ifs_documents
    )

print(f"   -> {len(ifs_ids)} exigences IFS indexees")

# ============================================================
# 5. Statistiques finales
# ============================================================
print("\n=== Index construit avec succes ===")
print(f"   Corpus: {CORPUS_PATH}")
print(f"   ChromaDB: {CHROMA_PATH}")
print(f"   Modele: {EMBED_MODEL}")
print(f"\nCollections:")
print(f"   - ifs_suspensions: {csv_collection.count()} documents")
print(f"   - ifs_requirements: {ifs_collection.count()} documents")

print("\n=== Pret pour RAG ===")
print("Pour interroger l'index, utiliser le prefi xe 'query: ' avec E5")
print("Exemple: model.encode(f'query: {question}')")
