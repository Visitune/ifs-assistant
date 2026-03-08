# IFS Food v8 — Assistant Auditeur IA
## Spécification complète du projet

> **Objectif** : Application Streamlit permettant à un auditeur IFS de décrire une situation observée et d'obtenir un avis structuré (KO / Majeur / NC classique) avec justifications issues du référentiel IFS Food v8 et de cas réels de suspensions de certificats.

---

## Table des matières

1. [Contexte et sources de données](#1-contexte-et-sources-de-données)
2. [Architecture générale](#2-architecture-générale)
3. [Structure des fichiers du projet](#3-structure-des-fichiers-du-projet)
4. [Étape 1 — Préparation des données](#4-étape-1--préparation-des-données)
5. [Étape 2 — Construction de l'index vectoriel](#5-étape-2--construction-de-lindex-vectoriel)
6. [Étape 3 — Moteur RAG](#6-étape-3--moteur-rag)
7. [Étape 4 — Abstraction LLM multi-provider](#7-étape-4--abstraction-llm-multi-provider)
8. [Étape 5 — Application Streamlit](#8-étape-5--application-streamlit)
9. [Prompt système de l'agent](#9-prompt-système-de-lagent)
10. [Installation et lancement](#10-installation-et-lancement)
11. [Requirements](#11-requirements)

---

## 1. Contexte et sources de données

### 1.1 Le référentiel IFS Food v8

- **Fichier source** : `ifs_food_v8_fr.json`
- **Contenu** : 5 chapitres, **240 exigences** dont **10 KO**
- **Structure d'une exigence** :
  ```json
  {
    "numero": "2.3.9",
    "estKO": true,
    "numeroKO": "KO N°3",
    "texte": "Texte officiel de l'exigence...",
    "onglets": {
      "bonnesPratiques": "...",
      "questionsExemple": "...",
      "elementsAVerifier": "...",
      "exemplesNonConformiteMajeure": "...",
      "exemplesKO": "..."
    }
  }
  ```

### 1.2 La base des suspensions IFS

- **Fichier source** : `LOCKEDIFS_-_version_OR__4_.csv`
- **Contenu** : **710 lignes** de suspensions de certificats réelles
- **Colonnes clés** :
  - `Lock reason` : contient le(s) numéro(s) d'exigence + type (KO/Major) + description de la NC en anglais
  - `Supplier`, `Country/Region`, `Product scopes`, `Certificate/Assessment lock date`
- **Statistiques** :
  - ~210 suspensions mentionnant KO
  - ~281 suspensions mentionnant Major
  - Top exigences : `2.3.9` (75x), `4.18.1` (66x), `5.11.3` (66x), `4.3.2` (55x), `4.10.2` (40x)

### 1.3 Langues

- JSON IFS : **français**
- CSV Lock reason : **anglais** majoritairement
- Embeddings : modèle multilingue requis

---

## 2. Architecture générale

```
┌─────────────────────────────────────────────────────────┐
│                    PHASE DE BUILD                        │
│  (lancée une seule fois)                                 │
│                                                          │
│  ifs_food_v8_fr.json ──┐                                │
│                         ├──► 01_prepare_data.py          │
│  LOCKEDIFS.csv ─────────┘         │                     │
│                              corpus.json                 │
│                                   │                      │
│                              02_build_index.py           │
│                                   │                      │
│                            chroma_db/ (index)            │
└───────────────────────────────────┼─────────────────────┘
                                    │
┌───────────────────────────────────▼─────────────────────┐
│                  PHASE RUNTIME (Streamlit)                │
│                                                          │
│  Question auditeur                                       │
│       │                                                  │
│       ▼                                                  │
│  rag_engine.py ──► ChromaDB (top-5 cas similaires)      │
│       │         ──► JSON IFS (exigence + guide)          │
│       │                                                  │
│       ▼                                                  │
│  llm_providers.py ──► Groq / Gemini / OpenAI-compat     │
│       │                                                  │
│       ▼                                                  │
│  Réponse structurée : KO / Majeur / NC + justification  │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Structure des fichiers du projet

```
ifs_assistant/
│
├── data/
│   ├── ifs_food_v8_fr.json          # Source : référentiel IFS (fourni)
│   ├── LOCKEDIFS.csv                # Source : suspensions réelles (fourni)
│   ├── corpus.json                  # Généré par 01_prepare_data.py
│   └── chroma_db/                   # Généré par 02_build_index.py
│       └── ...
│
├── 01_prepare_data.py               # Pipeline de préparation des données
├── 02_build_index.py                # Construction de l'index vectoriel
├── rag_engine.py                    # Moteur RAG (retrieval + construction prompt)
├── llm_providers.py                 # Abstraction multi-provider LLM
├── app.py                           # Application Streamlit
└── requirements.txt
```

---

## 4. Étape 1 — Préparation des données

**Fichier** : `01_prepare_data.py`

### Objectif

Transformer les deux sources brutes en un corpus JSON unifié, enrichi, prêt pour l'embedding.

### Logique de parsing du CSV

Le champ `Lock reason` peut contenir plusieurs exigences dans une même cellule. Exemple :
```
"2.3.9 KO N°3:
Texte de la NC 1...

4.3.2 Major:
Texte de la NC 2..."
```

**Algorithme de parsing** :
1. Découper sur les patterns `\d+\.\d+\.\d+ (KO|Major|NC):?`
2. Pour chaque segment extrait :
   - Extraire le numéro d'exigence (`req_number`)
   - Extraire le type (`KO`, `Major`, `NC`)
   - Nettoyer le texte de la NC (supprimer espaces, retours chariot inutiles)
3. Cross-référencer avec le JSON IFS pour injecter :
   - Texte officiel de l'exigence
   - Bonnes pratiques du guide
   - Exemples de NC majeures/KO du guide
4. Ignorer les segments sans correspondance dans le JSON

### Structure du corpus.json généré

```json
[
  {
    "id": "csv_001_2.3.9",
    "source": "csv",
    "req_number": "2.3.9",
    "severity": "KO",
    "ko_flag": true,
    "nc_text": "Texte de la NC réelle en anglais...",
    "req_text": "Texte officiel de l'exigence IFS en français...",
    "guide_bonnes_pratiques": "...",
    "guide_exemples_ko": "...",
    "guide_exemples_majeur": "...",
    "supplier": "Nom entreprise",
    "country": "Allemagne",
    "lock_date": "2025-04-28",
    "product_scope": "...",
    "embed_text": "[TEXTE COMPLET POUR EMBEDDING - voir ci-dessous]"
  }
]
```

### Construction du `embed_text`

Le champ `embed_text` est le texte qui sera passé au modèle d'embedding. Il doit être **dense en information métier** :

```
Exigence IFS {req_number} — {severity}

TEXTE OFFICIEL IFS :
{req_text}

BONNES PRATIQUES :
{guide_bonnes_pratiques}

NON-CONFORMITÉ OBSERVÉE :
{nc_text}
```

### Output attendu

- Nombre d'entrées : ~900-1000 (certaines lignes CSV génèrent plusieurs entrées)
- Fichier : `data/corpus.json`

---

## 5. Étape 2 — Construction de l'index vectoriel

**Fichier** : `02_build_index.py`

### Choix technologiques

| Composant | Choix | Raison |
|---|---|---|
| Modèle d'embedding | `intfloat/multilingual-e5-large` | Bilingue FR/EN, excellent sur textes techniques |
| Base vectorielle | **ChromaDB** | Local, sans serveur, simple, efficace pour <10k docs |
| Stockage | `data/chroma_db/` | Persistant, rechargé à chaque démarrage Streamlit |

### Logique

1. Charger `data/corpus.json`
2. Pour chaque entrée, encoder le champ `embed_text` via le modèle sentence-transformers
3. Stocker dans ChromaDB avec les métadonnées : `req_number`, `severity`, `ko_flag`, `source`
4. Créer également une **collection séparée** pour les exigences IFS pures (depuis JSON), permettant une recherche sur le référentiel uniquement

### Temps de build estimé

- ~5-10 minutes sur CPU standard pour ~1000 documents
- **À lancer une seule fois**, sauf si les sources sont mises à jour

---

## 6. Étape 3 — Moteur RAG

**Fichier** : `rag_engine.py`

### Classe `RAGEngine`

```python
class RAGEngine:
    def __init__(self, chroma_path, json_path, embed_model):
        # Charge ChromaDB + JSON IFS en mémoire
        pass

    def retrieve(self, query: str, req_number: str = None, top_k: int = 5) -> dict:
        """
        Retourne le contexte enrichi pour la question posée.
        
        Si req_number est fourni : filtre sur cette exigence en priorité
        Sinon : recherche sémantique pure dans tout le corpus
        
        Retourne :
        - matched_req : dict de l'exigence IFS (depuis JSON)
        - similar_cases : liste des top_k cas similaires du CSV
        - stats : { total_cases: int, ko_rate: float, major_rate: float }
        """
        pass

    def build_prompt(self, query: str, context: dict) -> str:
        """
        Construit le prompt final à envoyer au LLM.
        Voir section 9 pour le template complet.
        """
        pass
```

### Logique de retrieval

```
1. Si req_number détecté dans la question (regex \d+\.\d+\.\d+) :
   → Récupérer l'exigence exacte depuis le JSON
   → Filtrer ChromaDB sur req_number + recherche sémantique sur nc_text
   
2. Sinon (texte libre pur) :
   → Embedding de la question
   → Recherche sémantique dans tout ChromaDB
   → Identifier l'exigence la plus probable (celle des top résultats)
   → Récupérer le contexte IFS de cette exigence

3. Dans tous les cas :
   → Calculer les stats sur cette exigence dans le corpus
     (ex: "Cette exigence a généré 75 suspensions, dont 80% KO")
```

---

## 7. Étape 4 — Abstraction LLM multi-provider

**Fichier** : `llm_providers.py`

### Providers supportés

```python
PROVIDERS = {
    "Groq": {
        "models": [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768"
        ],
        "base_url": "https://api.groq.com/openai/v1",
        "sdk": "openai-compatible"
    },
    "Gemini": {
        "models": [
            "gemini-2.5-pro-preview-05-06",
            "gemini-2.0-flash-lite"
        ],
        "sdk": "google-generativeai"
    },
    "OpenRouter": {
        "models": [
            "openai/gpt-oss-120b",
            "meta-llama/llama-3.3-70b-instruct"
        ],
        "base_url": "https://openrouter.ai/api/v1",
        "sdk": "openai-compatible"
    }
}
```

### Interface unifiée

```python
class LLMProvider:
    def __init__(self, provider: str, model: str, api_key: str):
        pass

    def complete(self, system_prompt: str, user_message: str) -> str:
        """
        Appel unifié indépendant du provider.
        Retourne le texte de la réponse.
        """
        pass
```

**Note** : Groq et OpenRouter utilisent tous deux l'API compatible OpenAI (`openai` Python SDK avec `base_url` personnalisé). Gemini utilise le SDK `google-generativeai`. La classe `LLMProvider` abstrait cette différence.

---

## 8. Étape 5 — Application Streamlit

**Fichier** : `app.py`

### Layout

```
┌────────────────────────────────────────────────────────┐
│  🏭 IFS Food v8 — Assistant Auditeur IA                │
├──────────────┬─────────────────────────────────────────┤
│  SIDEBAR     │  ZONE PRINCIPALE                        │
│              │                                         │
│  ⚙️ Config   │  📋 Numéro d'exigence (optionnel)       │
│  Provider ▼  │  [  ex: 2.3.9  ]                        │
│  Modèle ▼    │                                         │
│  Clé API ●●● │  🔍 Décrivez la situation observée      │
│              │  ┌─────────────────────────────────┐   │
│  ℹ️ À propos  │  │ Zone de texte libre             │   │
│              │  │ (min. 50 caractères suggérés)   │   │
│              │  └─────────────────────────────────┘   │
│              │                                         │
│              │  [ 🔍 Analyser la situation ]           │
│              │                                         │
│              │  ─────────────────────────────────     │
│              │                                         │
│              │  RÉSULTAT :                             │
│              │  🔴 KO probable — Exigence 2.3.9        │
│              │                                         │
│              │  📖 Référentiel IFS                     │
│              │  [texte exigence + guide]               │
│              │                                         │
│              │  📊 Statistiques (75 cas similaires,    │
│              │     80% classés KO dans la base)        │
│              │                                         │
│              │  📂 Cas similaires (3 exemples)         │
│              │  [expandable cards]                     │
│              │                                         │
│              │  ⚠️ Disclaimer                          │
└──────────────┴─────────────────────────────────────────┘
```

### Comportements clés

- La **clé API est saisie dans la sidebar** et stockée uniquement en `st.session_state` (jamais écrite sur disque)
- Le **ChromaDB est chargé en cache** (`@st.cache_resource`) au premier lancement pour éviter le rechargement à chaque interaction
- Les **résultats sont affichés de façon progressive** (streaming si le provider le supporte)
- Un **badge coloré** indique le verdict : 🔴 KO / 🟠 Majeur / 🟡 NC classique / ⚪ Insuffisant pour conclure
- Les **cas similaires** sont affichés dans des `st.expander` avec pays, date, et texte NC

---

## 9. Prompt système de l'agent

```
Tu es un expert auditeur IFS Food v8 avec 15 ans d'expérience dans l'industrie 
alimentaire européenne. Tu assistes des auditeurs dans la qualification des 
non-conformités selon le référentiel IFS Food v8.

RÉFÉRENTIEL DISPONIBLE :
Exigence {req_number} : {req_text}

GUIDE IFS POUR CETTE EXIGENCE :
Bonnes pratiques : {guide_bonnes_pratiques}
Exemples de non-conformités majeures : {guide_exemples_majeur}
Exemples de KO : {guide_exemples_ko}

HISTORIQUE RÉEL DE SUSPENSIONS (base IFS) :
{similar_cases_formatted}

STATISTIQUES SUR CETTE EXIGENCE :
- {total_cases} suspensions réelles référencées
- {ko_rate}% classées KO / {major_rate}% classées Majeur

---
RÈGLES DE RÉPONSE :
Tu dois obligatoirement structurer ta réponse ainsi :

**VERDICT** : [KO / MAJEUR / NC CLASSIQUE / INSUFFISANT POUR CONCLURE]

**JUSTIFICATION IFS** : (référence exacte au texte de l'exigence et/ou du guide)

**ÉLÉMENTS DÉTERMINANTS** : (ce qui dans la situation décrite oriente vers ce verdict)

**POINTS À VÉRIFIER** : (questions complémentaires que l'auditeur devrait explorer)

**CAS SIMILAIRES RÉFÉRENCES** : (1-3 exemples issus de la base, avec date et pays)

---
RÈGLES MÉTIER :
- Un KO entraîne la suspension immédiate du certificat
- Un Majeur = plus de 20% de points déduits sur le chapitre concerné
- En cas de doute entre Majeur et KO, tu dois l'indiquer explicitement
- Tu ne te substitues PAS au jugement de l'auditeur certifié
- Si la description est insuffisante, demande des précisions plutôt que de conclure

DISCLAIMER (toujours inclure en fin de réponse) :
⚠️ Cet avis est fourni à titre d'assistance uniquement. La décision finale 
appartient à l'auditeur certifié IFS, conformément au référentiel IFS Food v8 
en vigueur et aux règles de son organisme de certification.
```

---

## 10. Installation et lancement

### Prérequis

- Python 3.10+
- ~2 Go d'espace disque (modèle d'embedding)
- Connexion internet pour le premier téléchargement du modèle

### Installation

```bash
# Cloner / créer le dossier projet
mkdir ifs_assistant && cd ifs_assistant

# Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements.txt

# Placer les fichiers sources dans data/
# data/ifs_food_v8_fr.json
# data/LOCKEDIFS.csv
```

### Build (à faire une seule fois)

```bash
# Étape 1 : préparer les données
python 01_prepare_data.py
# → génère data/corpus.json (~900-1000 entrées)

# Étape 2 : construire l'index vectoriel
python 02_build_index.py
# → génère data/chroma_db/
# → durée estimée : 5-15 min selon CPU
```

### Lancement de l'app

```bash
streamlit run app.py
```

Puis ouvrir `http://localhost:8501` dans le navigateur.

---

## 11. Requirements

```txt
# requirements.txt

# Interface
streamlit>=1.35.0

# Embeddings et RAG
sentence-transformers>=3.0.0
chromadb>=0.5.0

# LLM providers
openai>=1.30.0          # Pour Groq + OpenRouter (API-compatible)
google-generativeai>=0.8.0  # Pour Gemini

# Data
pandas>=2.0.0
numpy>=1.26.0

# Utilitaires
python-dotenv>=1.0.0
tqdm>=4.66.0
```

---

## Notes importantes pour Kilo Code

### Ordre d'implémentation recommandé

1. **`01_prepare_data.py`** → valider le parsing CSV et l'enrichissement JSON en premier
2. **`02_build_index.py`** → builder l'index, vérifier que les recherches retournent des résultats pertinents
3. **`rag_engine.py`** → tester le retrieval en script standalone avant d'intégrer dans Streamlit
4. **`llm_providers.py`** → tester chaque provider avec un appel simple
5. **`app.py`** → intégration finale

### Points d'attention critiques

- Le parsing regex du `Lock reason` est **la partie la plus fragile** : certaines lignes ont des formats atypiques, prévoir un fallback pour les lignes non parsées
- Le modèle d'embedding `multilingual-e5-large` nécessite de **préfixer les textes** avec `"passage: "` pour les documents indexés et `"query: "` pour les requêtes (spécification du modèle E5)
- ChromaDB doit être initialisé avec `PersistentClient` pour survivre aux redémarrages Streamlit
- Ne jamais logger ni écrire la clé API fournie par l'utilisateur

### Variables d'environnement optionnelles

```env
# .env (optionnel, pour développement local)
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=...
OPENROUTER_API_KEY=sk-or-...
```

En production Streamlit, l'utilisateur saisit sa clé directement dans l'interface.

---

*Document généré pour le projet IFS Food v8 Assistant Auditeur IA*
*Sources : ifs_food_v8_fr.json (240 exigences) + LOCKEDIFS.csv (710 suspensions)*