import pandas as pd
import json
import re
from pathlib import Path

# Paths
DATA_DIR = Path(".")
JSON_PATH = DATA_DIR / "ifs_food_v8_fr.json"
CSV_PATH = DATA_DIR / "LOCKEDIFS - version OR (4).csv"
OUTPUT_PATH = DATA_DIR / "mapping_review.csv"

def run_diagnostic():
    # Load requirements
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        ifs_data = json.load(f)
    
    ifs_numbers = set()
    for chapter in ifs_data:
        for ss in chapter.get("sous_sections", []):
            for req in ss.get("exigences", []):
                num = req["numero"].replace("*", "").strip()
                num = re.sub(r"KO N° \d+\s+", "", num)
                ifs_numbers.add(num)

    # Load CSV
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
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
        
        # Check for keywords that might indicate a requirement but missed by regex
        # ex: "3,2,1" or "exigence 4.2.1"
        if status == "MISSING":
            if re.search(r"exigence|requ|point", lock_reason, re.I):
                status = "POTENTIAL_MISS"

        results.append({
            "Original_Index": idx,
            "Supplier": row.get("Supplier", ""),
            "Lock_Reason": lock_reason[:500],
            "Extracted": ", ".join(valid_nums),
            "Status": status,
            "Corrected_Requirement": "" # For user or LLM to fill
        })

    review_df = pd.DataFrame(results)
    review_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    
    print(f"Diagnostic terminé. Fichier créé : {OUTPUT_PATH}")
    print(f"Stats :")
    print(review_df["Status"].value_counts())

if __name__ == "__main__":
    run_diagnostic()
