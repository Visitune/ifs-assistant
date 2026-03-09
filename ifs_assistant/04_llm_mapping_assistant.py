import pandas as pd
import os
from pathlib import Path
from llm_providers import LLMProvider
import time

# Configuration
INPUT_PATH = Path("mapping_review.csv")
OUTPUT_PATH = Path("mapping_corrected_llm.csv")
PROVIDER = "Gemini" # Or Groq
MODEL = "gemini-1.5-flash"

SYSTEM_PROMPT = """Tu es un expert en audit IFS Food v8. 
Ta tâche est d'analyser une justification de suspension ('Lock reason') et d'extraire le NUMÉRO de l'exigence IFS concernée (format X.Y.Z ou X.Y.Z.W).
Si tu trouves plusieurs exigences, cite la plus importante ou liste les séparées par une virgule.
Si aucune exigence n'est clairement indentifiable, réponds 'AUCUNE'.
Réponds UNIQUEMENT le numéro de l'exigence."""

def assist_mapping():
    if not os.path.exists(INPUT_PATH):
        print(f"Erreur : {INPUT_PATH} introuvable.")
        return

    df = pd.read_csv(INPUT_PATH)
    # On ne traite que les lignes problématiques
    to_process = df[df["Status"].isin(["MISSING", "POTENTIAL_MISS", "MULTIPLE"])]
    
    # Récupérer la clé API
    api_key = os.getenv(f"{PROVIDER.upper()}_API_KEY")
    if not api_key:
        print(f"Clé API pour {PROVIDER} manquante.")
        return

    llm = LLMProvider(PROVIDER, MODEL, api_key)
    
    print(f"Traitement de {len(to_process)} lignes avec {MODEL}...")
    
    for idx, row in to_process.iterrows():
        try:
            prompt = f"Lock reason: {row['Lock_Reason']}"
            suggestion = llm.complete(SYSTEM_PROMPT, prompt).strip()
            df.at[idx, "Corrected_Requirement"] = suggestion
            print(f"Index {row['Original_Index']} -> Suggestion LLM: {suggestion}")
            time.sleep(1) # Rate limiting safe
        except Exception as e:
            print(f"Erreur index {idx}: {e}")

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Terminé. Suggestions sauvegardées dans {OUTPUT_PATH}")

if __name__ == "__main__":
    assist_mapping()
