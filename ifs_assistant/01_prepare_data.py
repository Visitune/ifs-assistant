"""
01_prepare_data.py
==================
Prépare les données IFS Food v8 pour le RAG.
- Parse le CSV des suspensions (Lock reason)
- Enrichit avec les données JSON du référentiel IFS
- Génère corpus.json
"""

# -*- coding: utf-8 -*-
import json
import re
import pandas as pd
from pathlib import Path
from collections import Counter
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Paths
DATA_DIR = Path(".")
OUTPUT_DIR = Path("ifs_assistant/data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. Charger les données sources
# ============================================================
print("=== Étape 1: Chargement des données sources ===")

# Charger le JSON IFS
print("Chargement ifs_food_v8_fr.json...")
with open(DATA_DIR / "ifs_food_v8_fr.json", "r", encoding="utf-8") as f:
    ifs_data = json.load(f)

# Extraire toutes les exigences dans un dict indexé par numero
ifs_requirements = {}
ko_numbers = set()

for chapter in ifs_data:
    for sous_section in chapter.get("sous_sections", []):
        for req in sous_section.get("exigences", []):
            numero = req["numero"].replace("*", "").strip()
            # Supprimer les préfixes comme "KO N° 9 " ou "KO N° 5 "
            numero = re.sub(r"KO N° \d+\s+", "", numero)
            
            if numero in ifs_requirements:
                # Collision detected: append text and other relevant fields
                existing = ifs_requirements[numero]
                existing["texte"] += "\n" + req.get("texte", "")
                # Merge onglets if necessary
                if "onglets" in req:
                    for k, v in req["onglets"].items():
                        if v:
                            existing["onglets"][k] = existing["onglets"].get(k, "") + "\n" + v
            else:
                ifs_requirements[numero] = req
                if req.get("estKO", False):
                    ko_numbers.add(numero)

print(f"  -> {len(ifs_requirements)} exigences IFS chargees")
print(f"  -> {len(ko_numbers)} KO identifies")

# Charger le CSV
print("Chargement LOCKEDIFS...")
df = pd.read_csv(DATA_DIR / "LOCKEDIFS - version OR (4).csv", encoding="utf-8")
print(f"  -> {len(df)} lignes total")

# Filtrer IFS Food uniquement
df = df[df["Standard"].str.contains("IFS Food", case=False, na=False)]
print(f"  -> {len(df)} lignes IFS Food")

# ============================================================
# 2. Parser le Lock reason avec regex amélioré et Surcharge (Override)
# ============================================================
print("\n=== Étape 2: Parsing du Lock reason ===")

# Charger les corrections manuelles si elles existent
OVERRIDE_PATH = Path("mapping_corrected.csv")
overrides = {}
if OVERRIDE_PATH.exists():
    print(f"  -> Chargement des surcharges depuis {OVERRIDE_PATH}...")
    df_over = pd.read_csv(OVERRIDE_PATH)
    # On crée un mapping index -> exigence(s)
    for _, r in df_over.iterrows():
        if pd.notna(r["Corrected_Requirement"]) and str(r["Corrected_Requirement"]).strip() != "" and str(r["Corrected_Requirement"]).upper() != "AUCUNE":
            overrides[int(r["Original_Index"])] = str(r["Corrected_Requirement"])
    print(f"     ({len(overrides)} surcharges actives)")

def parse_lock_reason(lock_reason, ifs_numbers, row_index=None):
    """
    Parse le champ Lock reason. 
    Priorise le dictionnaire de surcharge si row_index est fourni.
    """
    if row_index is not None and row_index in overrides:
        # Utiliser la surcharge
        req_numbers = [n.strip() for n in overrides[row_index].split(",")]
        # On valide quand même que les numéros existent dans le référentiel
        valid_nums = [n for n in req_numbers if n in ifs_numbers]
        if valid_nums:
            results = []
            for n in valid_nums:
                results.append({"req_number": n, "severity": "Major", "nc_text": str(lock_reason)})
            return results

    if pd.isna(lock_reason):
        return []
    
    text = str(lock_reason)
    text_lower = text.lower()
    results = []
    
    # 1. Déterminer la sévérité globale (fallback)
    global_severity = "Major"
    if any(k in text_lower for k in [" ko ", " ko.", " ko,", "(ko)", " d ", " d.", " d,", " d)", " d "]):
        global_severity = "KO"
    
    # 2. Chercher TOUS les numéros d'exigence (X.Y.Z.W, X.Y.Z ou X.Y)
    found_nums = re.findall(r"(\d+\.\d+(?:\.\d+)?(?:\.\d+)?)", text)
    valid_nums = [n for n in set(found_nums) if n in ifs_numbers]
    
    if not valid_nums:
        return []

    # 3. Pour chaque numéro valide, essayer de trouver un bloc spécifique ou utiliser le texte global
    for req_num in valid_nums:
        # Chercher si ce numéro est suivi d'une sévérité spécifique à proximité (20 chars)
        severity = global_severity
        pattern_near = rf"{re.escape(req_num)}.*?(\bKO\b|\bMajor\b|\bMayor\b|\bNC\b|\bD\b)"
        match_near = re.search(pattern_near, text, re.IGNORECASE | re.DOTALL)
        
        if match_near:
            sev_raw = match_near.group(1).upper()
            if sev_raw in ["KO", "D"]: severity = "KO"
            else: severity = "Major"

        # Tenter d'isoler le texte de la NC
        nc_text = text.strip()
        
        results.append({
            "req_number": req_num,
            "severity": severity,
            "nc_text": nc_text[:2000]
        })
    
    return results

# Parser chaque ligne
parsed_entries = []
stats = {"total_rows": 0, "rows_with_parsed": 0, "total_reqs": 0}
# Compteur global pour les IDs uniques
global_counter = 0

for idx, row in df.iterrows():
    stats["total_rows"] += 1
    
    lock_reason = row.get("Lock reason", "")
    parsed = parse_lock_reason(lock_reason, ifs_requirements, row_index=idx)
    
    if parsed:
        stats["rows_with_parsed"] += 1
    
    # Compteur pour gerer les doublons d'ID
    req_counter = {}
    
    for p in parsed:
        stats["total_reqs"] += 1
        
        # Créer l'entrée
        entry = {
            "id": f"csv_{idx:04d}_{p['req_number']}",
            "source": "csv",
            "req_number": p["req_number"],
            "severity": p["severity"],
            "ko_flag": p["severity"] == "KO",
            "nc_text": p["nc_text"],
            "supplier": row.get("Supplier", ""),
            "country": row.get("Country/Region", ""),
            "lock_date": row.get("Certificate/Assessment lock date", ""),
            "product_scope": row.get("Product scopes", "")
        }
        
        # Enrichir avec les données IFS
        if p["req_number"] in ifs_requirements:
            req_if = ifs_requirements[p["req_number"]]
            entry["req_text"] = req_if.get("texte", "")
            onglets = req_if.get("onglets", {})
            entry["guide_bonnes_pratiques"] = onglets.get("bonnesPratiques", "")
            entry["guide_exemples_ko"] = onglets.get("exemplesKO", "")
            entry["guide_exemples_majeur"] = onglets.get("exemplesNonConformites", "")
        
        # Construire embed_text
        entry["embed_text"] = f"Exigence IFS {p['req_number']} — {p['severity']}\n\n"
        if entry.get("req_text"):
            entry["embed_text"] += f"TEXTE OFFICIEL IFS :\n{entry['req_text']}\n\n"
        if entry.get("guide_bonnes_pratiques"):
            entry["embed_text"] += f"BONNES PRATIQUES :\n{entry['guide_bonnes_pratiques']}\n\n"
        if entry.get("guide_exemples_ko"):
            entry["embed_text"] += f"EXEMPLES DE KO :\n{entry['guide_exemples_ko']}\n\n"
        if entry.get("guide_exemples_majeur"):
            entry["embed_text"] += f"EXEMPLES DE NON-CONFORMITÉ MAJEURE :\n{entry['guide_exemples_majeur']}\n\n"
        if entry.get("nc_text"):
            entry["embed_text"] += f"NON-CONFORMITÉ OBSERVÉE :\n{entry['nc_text']}"
        
        parsed_entries.append(entry)

print(f"  -> Lignes avec exigences parsees: {stats['rows_with_parsed']}/{stats['total_rows']}")
print(f"  -> Total entrees generatees: {stats['total_reqs']}")

# Compter les sévérités
severity_counts = Counter([e["severity"] for e in parsed_entries])
print(f"  → Répartition sévérités: {dict(severity_counts)}")

# ============================================================
# 3. Ajouter les exigences IFS pures (sans NC)
# ============================================================
print("\n=== Étape 3: Ajout des exigences IFS pures ===")

# Nombre d'entrées CSV pour éviter les doublons d'ID
csv_count = len(parsed_entries)

for i, (numero, req) in enumerate(ifs_requirements.items()):
    entry = {
        "id": f"ifs_{i:04d}_{numero}",
        "source": "ifs",
        "req_number": numero,
        "severity": "KO" if numero in ko_numbers else "Requirement",
        "ko_flag": numero in ko_numbers,
        "nc_text": "",
        "req_text": req.get("texte", ""),
        "guide_bonnes_pratiques": req.get("onglets", {}).get("bonnesPratiques", ""),
        "guide_exemples_ko": req.get("onglets", {}).get("exemplesKO", ""),
        "guide_exemples_majeur": req.get("onglets", {}).get("exemplesNonConformites", ""),
    }
    
    # Construire embed_text
    severity_str = "KO" if numero in ko_numbers else "Exigence"
    entry["embed_text"] = f"Exigence IFS {numero} — {severity_str}\n\n"
    if entry.get("req_text"):
        entry["embed_text"] += f"TEXTE OFFICIEL IFS :\n{entry['req_text']}\n\n"
    if entry.get("guide_bonnes_pratiques"):
        entry["embed_text"] += f"BONNES PRATIQUES :\n{entry['guide_bonnes_pratiques']}\n\n"
    if entry.get("guide_exemples_ko"):
        entry["embed_text"] += f"EXEMPLES DE KO :\n{entry['guide_exemples_ko']}\n\n"
    if entry.get("guide_exemples_majeur"):
        entry["embed_text"] += f"EXEMPLES DE NON-CONFORMITÉ MAJEURE :\n{entry['guide_exemples_majeur']}"
    
    parsed_entries.append(entry)

print(f"  -> {len(ifs_requirements)} exigences IFS ajoutees")

# ============================================================
# 4. Normaliser les severites et sauvegarder
# ============================================================
print("\n=== Étape 4: Normalisation ===")

# Normaliser toutes les severites
for entry in parsed_entries:
    if entry["severity"] in ["Ko", "KO"]:
        entry["severity"] = "KO"
        entry["ko_flag"] = True
    elif entry["severity"] in ["Major", "MAJOR", "major"]:
        entry["severity"] = "Major"
        entry["ko_flag"] = False

print("  -> Severites normalisees")

print("\n=== Étape 5: Sauvegarde ===")

output_path = OUTPUT_DIR / "corpus.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(parsed_entries, f, ensure_ascii=False, indent=2)

print(f"  -> Corpus sauvegarde: {output_path}")
print(f"  -> Total entrees: {len(parsed_entries)}")

# Stats finales
csv_entries = len([e for e in parsed_entries if e["source"] == "csv"])
ifs_entries = len([e for e in parsed_entries if e["source"] == "ifs"])
print(f"    - Depuis CSV: {csv_entries}")
print(f"    - Depuis IFS: {ifs_entries}")

print("\n=== Terminé ===")
