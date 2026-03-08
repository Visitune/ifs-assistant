# Script pour analyser les formats de Lock reason - version amelioree
import pandas as pd
import re

df = pd.read_csv('LOCKEDIFS - version OR (4).csv', encoding='utf-8')

# Filtrer IFS Food
ifs_df = df[df['Standard'].str.contains('IFS Food', case=False, na=False)]

# Patterns plus complets
patterns = [
    # Pattern 1: X.Y.Z KO/Major/etc (cas ideal)
    r'(\d+\.\d+(?:\.\d+)?)\s*[-–—]?\s*(KO|Major|Mayor|NC|D)\s*(?:N[°o]?\s*\d+)?[:\s-]',
    # Pattern 2: X.Y.Z (MAJOR) entre parentheses
    r'(\d+\.\d+(?:\.\d+)?)\s*\((KO|Major|Mayor|NC|D)\)',
    # Pattern 3: X.Y.Z seul (sans severite visible)
    r'(\d+\.\d+(?:\.\d+)?)(?:\s*[-–—]|\s+)([A-Z][a-z]+)',
]

results = []

for idx, row in ifs_df.iterrows():
    reason = str(row['Lock reason']) if pd.notna(row['Lock reason']) else ''
    
    for pattern in patterns:
        matches = re.findall(pattern, reason, re.IGNORECASE)
        for m in matches:
            req_num = m[0].strip()
            severity = m[1].strip() if len(m) > 1 else 'Unknown'
            
            # Normaliser
            severity = severity.capitalize()
            if severity.lower() == 'mayor':
                severity = 'Major'  # Typo espagnol
            if severity.lower() == 'd':
                severity = 'KO'  # D = KO
            
            # Verifier que c'est un numero d'exigence IFS valide (X.Y ou X.Y.Z)
            if re.match(r'^\d+\.\d+(\.\d+)?$', req_num):
                results.append({
                    'row': idx,
                    'req': req_num,
                    'severity': severity,
                    'full_reason': reason[:300]
                })

# Dedupliquer par row+req
seen = set()
unique_results = []
for r in results:
    key = (r['row'], r['req'])
    if key not in seen:
        seen.add(key)
        unique_results.append(r)

print(f"Total matches trouves (after dedup): {len(unique_results)}")

# Compter par severity
from collections import Counter
severities = Counter([r['severity'] for r in unique_results])
print("\nPar severite:")
for k, v in sorted(severities.items()):
    print(f"  {k}: {v}")

# Exemples de chaque type
print("\n=== Exemples Major ===")
major_examples = [r for r in unique_results if r['severity'] == 'Major'][:3]
for r in major_examples:
    print(f"  {r['req']}: {r['full_reason'][:150]}")

print("\n=== Exemples KO ===")
ko_examples = [r for r in unique_results if r['severity'] == 'KO'][:3]
for r in ko_examples:
    print(f"  {r['req']}: {r['full_reason'][:150]}")

print("\n=== Exemples Unknown ===")
unknown_examples = [r for r in unique_results if r['severity'] == 'Unknown'][:5]
for r in unknown_examples:
    print(f"  {r['req']}: {r['full_reason'][:150]}")

# Compter les exigences uniques
unique_reqs = set([r['req'] for r in unique_results])
print(f"\nExigences uniques: {len(unique_reqs)}")
