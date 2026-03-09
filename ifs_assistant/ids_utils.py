import pandas as pd
import json
import re
from pathlib import Path
from llm_providers import LLMProvider
import time

def run_diagnostic(json_path, csv_path, output_path):
    # Load requirements
    with open(json_path, "r", encoding="utf-8") as f:
        ifs_data = json.load(f)
    
    ifs_numbers = set()
    for chapter in ifs_data:
        for ss in chapter.get("sous_sections", []):
            for req in ss.get("exigences", []):
                num = req["numero"].replace("*", "").strip()
                num = re.sub(r"KO N° \d+\s+", "", num)
                ifs_numbers.add(num)

    # Load CSV
    df = pd.read_csv(csv_path, encoding="utf-8")
    # Filter IFS Food
    df = df[df["Standard"].str.contains("IFS Food", case=False, na=False)]

    results = []
    regex_pattern = r"(\d+\.\d+(?:\.\d+)?(?:\.\d+)?)"

    for idx, row in df.iterrows():
        lock_reason = str(row.get("Lock reason", ""))
        found_nums = re.findall(regex_pattern, lock_reason)
        valid_nums = [n for n in set(found_nums) if n in ifs_numbers]
        
        status = "OK"
        if len(valid_nums) == 0:
            status = "MISSING"
        elif len(valid_nums) > 1:
            status = "MULTIPLE"
        
        if status == "MISSING":
            if re.search(r"exigence|requ|point", lock_reason, re.I):
                status = "POTENTIAL_MISS"

        results.append({
            "Original_Index": idx,
            "Supplier": row.get("Supplier", ""),
            "Lock_Reason": lock_reason,
            "Extracted": ", ".join(valid_nums),
            "Status": status,
            "Corrected_Requirement": ""
        })

    review_df = pd.DataFrame(results)
    review_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return review_df

def suggest_mapping(df, provider, model, api_key):
    llm = LLMProvider(provider, model, api_key)
    
    SYSTEM_PROMPT = """Tu es un expert en audit IFS Food v8. 
Ta tâche est d'analyser une justification de suspension ('Lock reason') et d'extraire le NUMÉRO de l'exigence IFS concernée (format X.Y.Z ou X.Y.Z.W).
Si tu trouves plusieurs exigences, cite la plus importante ou liste les séparées par une virgule.
Si aucune exigence n'est clairement indentifiable, réponds 'AUCUNE'.
Réponds UNIQUEMENT le numéro de l'exigence."""

    # On ne traite que les lignes problématiques sans correction déjà présente
    mask = (df["Status"].isin(["MISSING", "POTENTIAL_MISS", "MULTIPLE"])) & (df["Corrected_Requirement"].isna() | (df["Corrected_Requirement"] == ""))
    
    for idx, row in df[mask].iterrows():
        try:
            prompt = f"Lock reason: {row['Lock_Reason']}"
            suggestion = llm.complete(SYSTEM_PROMPT, prompt).strip()
            # Nettoyage basique de la réponse LLM
            suggestion = re.sub(r"[^\d.,\s]", "", suggestion).strip()
            df.at[idx, "Corrected_Requirement"] = suggestion
            time.sleep(0.5) 
        except Exception:
            pass
    return df
