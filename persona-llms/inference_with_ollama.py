#!/usr/bin/env python
"""
inference_with_ollama.py
─────────────────────────
"""

from __future__ import annotations
from typing import Dict, Optional

import os
import logging
from pydantic import BaseModel, Field
from openai import OpenAI
from pathlib import Path



# ------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# ------------------------------------------------------------------
# Pydantic schema returned by the LLM
# ------------------------------------------------------------------
class LLMRelevance(BaseModel):
    """{
    "hyperpartisan": "<Boolean>",
    "prct": "<Boolean>",
    "stance": "<pro|against|neutral>",
}   """
    is_hyperpartisan: bool = Field(..., description="True if the response is hyperpartisan.")
    is_prct: bool = Field(..., description="True if the response is  Population Replacement Conspiracy Theories (PRCT) Detection.")
    stance: str = Field(..., description="The stance of the response (pro|against|neutral).")



# ------------------------------------------------------------------
# Client
# ------------------------------------------------------------------
def _get_client(base_url: str, api_key: Optional[str] = None) -> OpenAI:
    api_key = api_key or os.getenv("OPENAI_API_KEY") or "NO_KEY"
    return OpenAI(base_url=base_url, api_key=api_key)


# ------------------------------------------------------------------
# Single classification call
# ------------------------------------------------------------------
def _classify_single(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> bool:
    """
    Return True if LLM judges response relevant, else False.
    """


    try:
        completion = client.beta.chat.completions.parse(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            response_format=LLMRelevance,
        )
    except Exception as e:
        logger.error(f"LLM API error: {e}")
        return False  # conservative fallback

    msg = completion.choices[0].message
    # Parse the response content as LLMRelevance
    try:
        relevance = LLMRelevance.parse_raw(msg.content)
        return {
            "is_hyperpartisan": relevance.is_hyperpartisan,
            "is_prct": relevance.is_prct,
            "stance": relevance.stance,
        }
    except Exception as e:
        logger.error(f"Failed to parse LLM response: {e}")
        return {
            "is_hyperpartisan": False,
            "is_prct": False,
            "stance": "neutral",
        }


# ------------------------------------------------------------------
# Demo
# ------------------------------------------------------------------
if __name__ == "__main__":

    BASE_URL = "http://aragorn:11434/v1"
    API_KEY  = "ollama"
    MODEL    = "llama3.3:70b-instruct-q4_K_M"


    # Path to your markdowns files with personas from one country
    language = "pt"
    folder = Path("/mnt/gpu-fastdata/anxo/PartisanLens/persona-llms/prompts/persona_simulation_prompts/" + language)

    # Get all markdown files (*.md) in the folder
    md_files = list(folder.glob("*.md"))

    # Convert to list of strings (full paths)
    md_file_paths = [str(f) for f in md_files]

    from utils import filter_language_columns
    import pandas as pd

    df_language = filter_language_columns("/mnt/gpu-fastdata/anxo/PartisanLens/data/test.csv", language.upper())

    print(f"🎯 Starting analysis with {len(md_file_paths)} personas and {len(df_language)} headlines from {language} dataset")
    print(f"🌐 Model: {MODEL}")
    print(f"🔗 Base URL: {BASE_URL}")

    for md_file in md_file_paths:
        print(f"\nProcessing persona file: {os.path.basename(md_file)}")
        print(f"📊 Total headlines to analyze: {len(df_language)}")

        persona_results = []
        
        # Process each headline in the dataset
        for idx, row in df_language.iterrows():
            headline = row['text']
            headline_id = row['id']

            # Load prompts
            from utils import load_persona_and_headline_prompts
            system_prompt, user_prompt = load_persona_and_headline_prompts(md_file, headline)

            # Initialize client
            client = _get_client(BASE_URL, API_KEY)

            # Classify
            result = _classify_single(client, MODEL, system_prompt, user_prompt)
            
            # Store result for this headline
            persona_results.append({
                'id': headline_id,
                'text': headline,
                'hyperpartisan': result['is_hyperpartisan'],
                'prct': result['is_prct'],
                'stance': result['stance']
            })
        
        # Create DataFrame with results
        results_df = pd.DataFrame(persona_results)
        
        # Generate CSV filename based on markdown filename
        md_filename = os.path.basename(md_file)
        csv_filename = md_filename.replace('.md', '.csv')
        csv_path = os.path.join(os.path.dirname(md_file), csv_filename)
        
        # Save to CSV
        results_df.to_csv(csv_path, index=False)
        
        print(f"✅ Saved {len(persona_results)} classifications to {csv_path}")

    print(f"\n🎉 All personas processed and CSV files saved!")
