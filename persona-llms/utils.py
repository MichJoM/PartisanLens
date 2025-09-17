from typing import Union
import pandas as pd
import pandas as pd


def load_persona_and_headline_prompts (persona_file_path: str, headline: str) -> str:
    """
    Load persona prompt from a file and combine it with a headline prompt.

    Args:
        persona_file_path (str): Path to the persona prompt file.
        headline (str): The headline to be analyzed.

    Returns:
        str: Combined prompt with persona and headline.
    """

    with open(persona_file_path, "r", encoding="utf-8") as f:
        persona_content = f.read()


    system_prompt_ending = """When performing your classification, output a JSON object with this format:
    ```json
    {
        "hyperpartisan": "<Boolean>",
        "prct": "<Boolean>",
        "stance": "<pro|against|neutral>",
    } ```
    DO NOT include any commentary or explanation. Only return valid JSON."""


    system_prompt = persona_content + "\n" + system_prompt_ending


    general_guidelines_prompt = "/mnt/gpu-fastdata/anxo/PartisanLens/persona-llms/prompts/general_annotation_guidelines_prompt.md"

    with open(general_guidelines_prompt, "r", encoding="utf-8") as f:
        general_guidelines_content = f.read()   

    # Replace the placeholder with the actual headline
    prompt = persona_content + general_guidelines_content.replace("{headlines}", headline)

    return system_prompt, prompt


def filter_language_columns(csv_path: str, language: str) -> pd.DataFrame:
    """
    Reads a CSV and filters rows for a specific language.
    
    Args:
        csv_path (str): Path to the CSV file.
        language (str): Language to filter (e.g. 'ITA', 'PT', 'SPA').
    
    Returns:
        pd.DataFrame: A DataFrame with all columns but only rows where language column matches the specified language.
    """
    # Read CSV
    df = pd.read_csv(csv_path)

    # Filter rows where language column matches the specified language
    filtered_df = df[df['language'] == language]

    return filtered_df


