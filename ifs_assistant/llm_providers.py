# -*- coding: utf-8 -*-
"""
llm_providers.py
================
Abstraction multi-provider pour Groq, Gemini et OpenRouter.
"""

import os
from typing import Optional, List, Dict, Any
import google.generativeai as genai
from openai import OpenAI

class LLMProvider:
    def __init__(self, provider: str, model: str, api_key: str):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        
        if provider == "Groq":
            self.client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=api_key
            )
        elif provider == "OpenRouter":
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key
            )
        elif provider == "Gemini":
            genai.configure(api_key=api_key)
            self.gen_model = genai.GenerativeModel(model)
        else:
            raise ValueError(f"Provider inconnu: {provider}")

    def complete(self, system_prompt: str, user_message: str) -> str:
        """
        Appel unifié indépendant du provider.
        """
        if self.provider in ["Groq", "OpenRouter"]:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.1
            )
            return response.choices[0].message.content
        
        elif self.provider == "Gemini":
            # Gemini gère le système via un paramètre ou en début de prompt
            # On utilise ici l'instruction système
            response = self.gen_model.generate_content(
                f"{system_prompt}\n\nUSER MESSAGE:\n{user_message}",
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1
                )
            )
            return response.text
            
        return "Erreur: Provider non supporté"

def get_available_models() -> Dict[str, List[str]]:
    """Retourne la liste des modèles par provider selon la spec."""
    return {
        "Groq": [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768"
        ],
        "Gemini": [
            "gemini-2.0-pro-exp-02-05",
            "gemini-2.0-flash-lite",
            "gemini-1.5-pro",
            "gemini-1.5-flash"
        ],
        "OpenRouter": [
            "meta-llama/llama-3.3-70b-instruct",
            "openai/gpt-4o-mini",
            "anthropic/claude-3.5-sonnet"
        ]
    }
