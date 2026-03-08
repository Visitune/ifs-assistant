# -*- coding: utf-8 -*-
"""
app.py
======
Interface Streamlit pour l'Assistant Auditeur IFS Food v8.
"""

import streamlit as st
from pathlib import Path
from rag_engine import RAGEngine
from llm_providers import LLMProvider, get_available_models
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration de la page
st.set_page_config(
    page_title="IFS Food v8 - Assistant Auditeur IA",
    page_icon="🏭",
    layout="wide"
)

# --- UTILS & PATHS ---
PROJECT_ROOT = Path(__file__).parent.parent
CHROMA_PATH = PROJECT_ROOT / "ifs_assistant" / "data" / "chroma_db"
JSON_PATH = PROJECT_ROOT / "ifs_food_v8_fr.json"

# --- CACHE DES RESSOURCES ---
@st.cache_resource
def load_rag_engine():
    return RAGEngine(str(CHROMA_PATH), str(JSON_PATH))

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Configuration")
    
    models_dict = get_available_models()
    provider = st.selectbox("LLM Provider", list(models_dict.keys()))
    model = st.selectbox("Modèle", models_dict[provider])
    
    # Récupérer la clé API par défaut de l'env ou demander à l'utilisateur
    env_key = os.getenv(f"{provider.upper()}_API_KEY")
    api_key = st.text_input(f"Clé API {provider}", value=env_key if env_key else "", type="password")
    
    st.divider()
    st.info("""
    **À propos**
    Cet assistant utilise le RAG (Retrieval Augmented Generation) basé sur :
    - Le référentiel IFS Food v8 (240 exigences)
    - La base des suspensions IFS (>700 cas réels)
    """)

st.title("🏭 IFS Food v8 — Assistant Auditeur IA")

tabs = st.tabs(["🔍 Analyse d'audit", "📖 Guide détaillé", "📊 Statistiques globales"])

with tabs[0]:
    st.markdown("Décrivez une situation rencontrée en audit pour obtenir un avis technique structuré.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        req_number = st.text_input("📋 Numéro d'exigence (Optionnel)", placeholder="ex: 2.3.9")
        situation = st.text_area(
            "🔍 Décrivez la situation observée", 
            height=250,
            placeholder="Décrivez précisément les faits, les preuves collectées et le contexte..."
        )
        
        analyze_btn = st.button("🔍 Analyser la situation", type="primary", use_container_width=True)

with tabs[1]:
    st.header("📖 Guide d'utilisation & Méthodologie")
    st.markdown("""
    ### 🚀 Comment utiliser l'assistant ?
    1. **Entrez votre clé API** dans la barre latérale (Sidebar).
    2. **Saisissez le numéro d'exigence** concerné (optionnel mais fortement recommandé pour plus de précision).
    3. **Décrivez la situation** de manière détaillée (ex: "Lors de la visite de l'atelier de conditionnement, j'ai observé que les nettoyeurs haute pression étaient utilisés à proximité de produits non protégés...").
    4. Cliquez sur **Analyser**.
    
    ### 🧠 Logique de construction (RAG)
    L'application repose sur une architecture **RAG (Retrieval-Augmented Generation)** :
    - **Indexation** : Le référentiel IFS Food v8 et l'historique des suspensions (>700 cas réels) ont été transformés en vecteurs (embeddings) via le modèle `multilingual-e5-large`.
    - **Recherche (Retrieval)** : Votre saisie est comparée à cette base de données vectorielle pour extraire les cas les plus similaires et le texte officiel correspondant.
    - **Analyse (Augmentation)** : Un LLM (Large Language Model) reçoit votre situation enrichie du contexte réglementaire et historique pour formuler une analyse experte.
    
    ### ⚠️ Précautions d'usage
    L'assistant est un outil d'aide à la décision. Il ne remplace pas le jugement professionnel de l'auditeur et doit être utilisé en complément d'une vérification directe sur le terrain.
    """)

with tabs[2]:
    st.header("📊 Statistiques du référentiel")
    st.write("Aperçu de la base de données de suspensions utilisée pour le RAG.")
    # On pourrait ajouter ici des graphiques simples basés sur corpus.json si nécessaire
    st.info("Cette section sera enrichie de graphiques interactifs dans une future mise à jour.")

# --- LOGIQUE D'ANALYSE ---
if analyze_btn:
    if not api_key:
        st.error("Veuillez saisir une clé API dans la barre latérale.")
    elif len(situation) < 20:
        st.warning("La description de la situation est trop courte pour une analyse pertinente.")
    else:
        with st.spinner("Analyse en cours (RAG + LLM)..."):
            try:
                # 1. Charger le moteur RAG
                rag = load_rag_engine()
                
                # 2. Retrieval
                context = rag.retrieve(situation, req_number if req_number else None)
                
                # 3. Préparer le prompt
                system_prompt = rag.build_prompt(situation, context)
                
                # 4. Appel LLM
                llm = LLMProvider(provider, model, api_key)
                response = llm.complete(system_prompt, situation)
                
                # --- AFFICHAGE DES RÉSULTATS ---
                with col2:
                    st.subheader("Résultat de l'Analyse")
                    
                    # Décoration selon le verdict probable
                    if "**VERDICT** : KO" in response:
                        st.error("🔴 KO probable")
                    elif "**VERDICT** : MAJEUR" in response:
                        st.warning("🟠 Majeur probable")
                    elif "**VERDICT** : NC" in response:
                        st.info("🟡 NC classique probable")
                    else:
                        st.write("⚪ Analyse terminée")
                        
                    st.markdown(response)
                    
                    st.divider()
                    
                    # Section Référentiel
                    with st.expander("📖 Référentiel IFS & Guide"):
                        req = context['matched_req']
                        if req:
                            st.write(f"**Exigence {req['numero']}** : {req['texte']}")
                            st.write("---")
                            st.write("**Bonnes Pratiques** :")
                            st.write(req['onglets'].get('bonnesPratiques'))
                        else:
                            st.write("Aucune correspondance exacte trouvée dans le référentiel.")
                            
                    # Section Statistiques
                    stats = context['stats']
                    st.write(f"📊 **Statistiques base IFS** ({stats['total_cases']} cas similaires trouvés)")
                    prog_ko = stats['ko_rate'] / 100
                    prog_maj = stats['major_rate'] / 100
                    st.write(f"Taux de KO historique : {stats['ko_rate']}%")
                    st.progress(prog_ko)
                    st.write(f"Taux de Majeur historique : {stats['major_rate']}%")
                    st.progress(prog_maj)
                    
                    # Section Cas Similaires
                    st.write("📂 **Exemples de cas réels**")
                    for case in context['similar_cases']:
                        meta = case['metadata']
                        with st.expander(f"Cas {meta['severity']} - {meta['country']} ({meta['lock_date']})"):
                            st.write(case['document'])
                            
            except Exception as e:
                st.error(f"Une erreur est survenue lors de l'analyse : {str(e)}")
                st.exception(e)

# --- FOOTER / DISCLAIMER ---
st.divider()
st.caption("Assistant expérimental. La décision finale appartient toujours à l'auditeur certifié.")
