# -*- coding: utf-8 -*-
"""
rag_engine.py
=============
Moteur RAG (Retrieval-Augmented Generation) pour l'Assistant IFS.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from sentence_transformers import SentenceTransformer

class RAGEngine:
    def __init__(self, chroma_path: str, json_path: str, model_name: str = "intfloat/multilingual-e5-large"):
        # Initialiser ChromaDB Client Persistant
        self.client = chromadb.PersistentClient(path=chroma_path)
        
        # Utiliser get_or_create_collection pour plus de résilience
        self.csv_collection = self.client.get_or_create_collection("ifs_suspensions")
        self.ifs_collection = self.client.get_or_create_collection("ifs_requirements")
        
        # Charger le JSON IFS original pour les données complètes
        with open(json_path, "r", encoding="utf-8") as f:
            ifs_data = json.load(f)
            
        # Indexer les exigences par numéro pour un accès rapide
        self.ifs_requirements = {}
        for chapter in ifs_data:
            for sous_section in chapter.get("sous_sections", []):
                for req in sous_section.get("exigences", []):
                    numero = req["numero"].replace("*", "")
                    self.ifs_requirements[numero] = req
        
        # Charger le modèle d'embedding
        self.model = SentenceTransformer(model_name)

    def retrieve(self, query: str, req_number: Optional[str] = None, top_k: int = 5) -> Dict[str, Any]:
        """
        Recherche sémantique et contextuelle.
        """
        # Préparer la requête pour E5 (prefix 'query: ')
        query_to_embed = f"query: {query}"
        query_vector = self.model.encode(query_to_embed).tolist()
        
        # 1. Identifier l'exigence
        matched_req = None
        
        # Regex pour détecter un numéro d'exigence dans le texte (ex: 2.3.9)
        regex_req = r"(\d+\.\d+(?:\.\d+)?)"
        found_nums = re.findall(regex_req, query)
        
        if req_number:
            matched_req = self.ifs_requirements.get(req_number)
        elif found_nums:
            matched_req = self.ifs_requirements.get(found_nums[0])
            req_number = found_nums[0]
            
        # Si toujours pas d'exigence, chercher dans la collection ifs_requirements
        if not matched_req:
            try:
                res_ifs = self.ifs_collection.query(
                    query_embeddings=[query_vector],
                    n_results=1
                )
                if res_ifs['metadatas'] and res_ifs['metadatas'][0]:
                    req_num = res_ifs['metadatas'][0][0]['req_number']
                    matched_req = self.ifs_requirements.get(req_num)
                    req_number = req_num
            except Exception as e:
                print(f"Erreur lors de la requête ifs_collection: {e}")
                # Fallback: On peut essayer de chercher par mot clé dans les titres si besoin
                pass

        # 2. Chercher des cas similaires (CSV)
        where_filter = {}
        if req_number:
            where_filter = {"req_number": req_number}
            
        similar_cases = []
        try:
            res_csv = self.csv_collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=where_filter if where_filter else None
            )
            
            if res_csv['metadatas']:
                for i in range(len(res_csv['metadatas'][0])):
                    similar_cases.append({
                        "metadata": res_csv['metadatas'][0][i],
                        "document": res_csv['documents'][0][i]
                    })
        except Exception as e:
            print(f"Erreur lors de la requête csv_collection: {e}")

        # 3. Calculer les statistiques (sur la base de tous les cas du même numéro)
        total_cases = 0
        ko_count = 0
        major_count = 0
        
        try:
            res_meta = self.csv_collection.get(
                where={"req_number": req_number} if req_number else None,
                include=["metadatas"]
            )
            all_cases_meta = res_meta['metadatas']
            
            total_cases = len(all_cases_meta)
            ko_count = sum(1 for m in all_cases_meta if m.get('ko_flag', False) or m.get('severity') == 'KO')
            major_count = sum(1 for m in all_cases_meta if m.get('severity') == 'Major')
        except Exception as e:
            print(f"Erreur lors de la récupération des stats: {e}")
        
        ko_rate = round((ko_count / total_cases * 100), 1) if total_cases > 0 else 0
        major_rate = round((major_count / total_cases * 100), 1) if total_cases > 0 else 0
        
        return {
            "req_number": req_number,
            "matched_req": matched_req,
            "similar_cases": similar_cases,
            "stats": {
                "total_cases": total_cases,
                "ko_rate": ko_rate,
                "major_rate": major_rate
            }
        }

    def build_prompt(self, query: str, context: Dict[str, Any]) -> str:
        """
        Construit le prompt final selon le template spécifié.
        """
        req_num = context.get("req_number", "Inconnu")
        req_data = context.get("matched_req", {})
        req_text = req_data.get("texte", "Non disponible")
        
        onglets = req_data.get("onglets", {})
        guide_bp = onglets.get("bonnesPratiques", "Non disponible")
        guide_ko = onglets.get("exemplesKO", "Non disponible")
        guide_major = onglets.get("exemplesNonConformites", "Non disponible")
        
        stats = context.get("stats", {})
        
        # Formater les cas similaires
        similar_formatted = ""
        for i, case in enumerate(context.get("similar_cases", [])):
            meta = case["metadata"]
            similar_formatted += f"Cas {i+1} ({meta.get('severity', 'NC')}) - {meta.get('country', 'N/A')} - {meta.get('lock_date', 'N/A')}:\n"
            similar_formatted += f"{case['document']}\n\n"
            
        if not similar_formatted:
            similar_formatted = "Aucun cas historique précis trouvé dans la base pour ce numéro."

        prompt_template = f"""Tu es un expert auditeur IFS Food v8 avec 15 ans d'expérience dans l'industrie alimentaire européenne. Tu assistes des auditeurs dans la qualification des non-conformités selon le référentiel IFS Food v8.

RÉFÉRENTIEL DISPONIBLE :
Exigence {req_num} : {req_text}

GUIDE IFS POUR CETTE EXIGENCE :
Bonnes pratiques : {guide_bp}
Exemples de non-conformités majeures : {guide_major}
Exemples de KO : {guide_ko}

HISTORIQUE RÉEL DE SUSPENSIONS (base IFS) :
{similar_formatted}

STATISTIQUES SUR CETTE EXIGENCE :
- {stats.get('total_cases', 0)} suspensions réelles référencées
- {stats.get('ko_rate', 0)}% classées KO / {stats.get('major_rate', 0)}% classées Majeur

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
⚠️ Cet avis est fourni à titre d'assistance uniquement. La décision finale appartient à l'auditeur certifié IFS, conformément au référentiel IFS Food v8 en vigueur et aux règles de son organisme de certification.
"""
        return prompt_template
