# IFS Food v8 — Assistant Auditeur IA 🏭

Application intelligente d'assistance aux auditeurs **IFS Food v8**, utilisant une architecture RAG (Retrieval-Augmented Generation) pour qualifier les non-conformités.

## 🌟 Objectif
Permettre à un auditeur de décrire une situation observée et d'obtenir un verdict structuré (KO / Majeur / NC) basé sur :
1. **Le référentiel officiel IFS Food v8** (240 exigences).
2. **Le guide de bonnes pratiques IFS**.
3. **Un historique réel de plus de 700 suspensions** de certificats (provenant de la base IFS).

## 🛠️ Architecture Technique
- **Frontend** : Streamlit
- **Moteur RAG** : ChromaDB (Base vectorielle locale)
- **Embeddings** : `intfloat/multilingual-e5-large` (Modèle SOTA pour le multilingue FR/EN)
- **LLM** : Multi-providers via API (Groq, Gemini, OpenRouter)

## 📁 Structure du Projet
- `ifs_food_v8_fr.json` : Base de données du référentiel.
- `ifs_assistant/`
    - `app.py` : Point d'entrée de l'application.
    - `rag_engine.py` : Moteur de recherche et de construction de contexte.
    - `llm_providers.py` : Abstraction pour les appels aux modèles (Llama 3, Gemini, etc.).
    - `01_prepare_data.py` : Pipeline de parsing du CSV et enrichissement JSON.
    - `02_build_index.py` : script de construction de l'index vectoriel.

## 🚀 Installation & Lancement

### Local
1. Cloner le repo.
2. Installer les dépendances :
   ```bash
   pip install -r ifs_assistant/requirements.txt
   ```
3. Lancer l'application :
   ```bash
   python -m streamlit run ifs_assistant/app.py
   ```

### Streamlit Cloud
1. Connectez votre dépôt GitHub à Streamlit Cloud.
2. configurez les **Secrets** si vous souhaitez pré-remplir les clés API (ex: `GROQ_API_KEY`, `GEMINI_API_KEY`).
3. Spécifiez le chemin du fichier principal : `ifs_assistant/app.py`.

## 🧠 Fonctionnement de l'IA
L'assistant ne se contente pas de répondre par "intuition". Pour chaque question :
1. Il identifie l'exigence IFS concernée.
2. Il récupère le texte exact de la norme et les exemples du guide.
3. Il extrait 5 cas historiques réels similaires.
4. Il présente des statistiques sur le taux de KO/Majeur pour cette exigence.
5. Il synthétise le tout via le LLM pour fournir une justification béton.

---
*Développé pour l'excellence opérationnelle en audit de sécurité alimentaire.*
