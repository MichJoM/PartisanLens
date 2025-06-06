from typing import Union

import pandas as pd


PROMPT = """
You are a strict JSON generator. Analyze the following news headline and output a JSON object with this exact format:
{{
    "chain_of_thought": "<reasoning>",
    "hyperpartisan": "<Boolean>",
    "prct": "<Boolean>",
    "stance": "<pro|against|neutral>",
}}
Hyperpartisan: Does it show strong ideological bias or use emotionally charged language? (True/False) 
PRCT: Does it contain Population Replacement Conspiracy Theory content claiming a deliberate plan to replace native populations? (True/False) 
Stance: What is its stance toward immigration policies? (pro/against/neutral).

DO NOT include any commentary or explanation. Only return valid JSON.

Headline: {}
"""

LABELS_PROMPT = """
You are a strict JSON generator. Analyze the following news headline and output a JSON object with this exact format:
{{
    "hyperpartisan": "<Boolean>",
    "prct": "<Boolean>",
    "stance": "<pro|against|neutral>",
}}
Hyperpartisan: Does it show strong ideological bias or use emotionally charged language? (True/False) 
PRCT: Does it contain Population Replacement Conspiracy Theory content claiming a deliberate plan to replace native populations? (True/False) 
Stance: What is its stance toward immigration policies? (pro/against/neutral).

DO NOT include any commentary or explanation. Only return valid JSON.

Headline: {}
"""


def build_few_shots(few_shots: pd.DataFrame) -> list[Union[dict[str, str], dict[str, str], dict[str, str]]]:
    few_shot_prompt = []

    for index, row in few_shots.iterrows():
        hyperpartisan = row['hyperpartisan']
        prct = row['prct']
        stance = row['stance']

        instr = PROMPT.format(row["text"])

        few_shot_prompt.append({"from": "human", "value": f"{instr}"})

        answer = f"""{{ 
                "hyperpartisan": "{hyperpartisan}",
                "prct": "{prct}", 
                "stance": "{stance}"
        }}"""

        few_shot_prompt.append({"from": "gpt", "value": f"{answer}"})

    return few_shot_prompt
