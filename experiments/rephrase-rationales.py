import argparse
import json
import os
import re

import json_repair
import pandas as pd
from tqdm import tqdm
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

save_batch = 50
batch_size = 12


model_mapping = {
    "llama-70": {
        "name": "unsloth/Llama-3.3-70B-Instruct",
        "chat_template": "llama-3",
        "pattern": r"<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>"
    }
}


instruction = ("""
You are a strict JSON generator.
Rephrase and enrich the following explanation in English, using natural language and output a JSON object with this exact format:
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

Headline: '{}'. This is the templated explanation '{}', this is the hyperpartisan label '{}', this is the PRCT label '{}', and this is the stance '{}'.
Generate a step-by-step explanation that supports the given labels."""
)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run LLM inference on a dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset (CSV/TSV)")
    parser.add_argument("--output", type=str, default="rephrased-rationales.tsv", help="Output file path")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token for model access")
    return parser.parse_args()


def clean_generated_json(text):
    text = text.replace('<|finetune_right_pad_id|>', '')
    text = text.replace('<|start_header_id|>assistant<|end_header_id|>\n', '')
    text = text.replace('<|start_header_id|> assistant<|end_header_id|>\n', '')
    text = text.replace('<|start_header_id|>assistant<|end_header_id|>.', '')
    text = text.replace('<|start_header_id|>assistant<|end_header_id|>:', '')
    text = text.replace('assistant<|end_header_id|>\n', '')
    text = text.replace('assistant<|end_header_id|>:', '')
    text = text.replace('<|python_tag|>', '')

    return text


def main():
    print('Starting...', flush=True)

    args = parse_arguments()
    max_seq_length = 4096
    max_tokens = 4096

    df = pd.read_csv(f"{args.dataset}", sep=',')

    ids = df['id'].to_list()
    hyperpartisan = df['hyperpartisan_gold_label'].map({1: "True", 0: "False"}).to_list()
    prct = df['prct_gold_label'].map({1: "True", 0: "False"}).to_list()
    stance = df['stance_gold_label'].to_list()
    texts = df['text'].to_list()
    rationales = df['templated_rationales'].to_list()

    model_details = model_mapping['llama-70']
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_details["name"],
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
        token=args.hf_token
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template=model_details["chat_template"],
        mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
    )

    FastLanguageModel.for_inference(model)

    messages = []
    metadata = []
    for idx, h, p, s, t, r in zip(ids, hyperpartisan, prct, stance, texts, rationales):
        messages.append([{"from": "human", "value": instruction.format(t, r, h, p, s)}])
        metadata.append({"id": idx, "hyperpartisan": h, "prct": p, "stance": s, "text": t, "templated_rationales": r})

    print('Messages to inference size:', len(messages), flush=True)

    output_file = f"{args.output}"
    file_exists = os.path.isfile(output_file)

    for i in tqdm(range(0, len(messages), batch_size)):
        batch_messages = messages[i:i + batch_size]
        batch_metadata = metadata[i:i + batch_size]

        prompts = [tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        ) for message in batch_messages]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to("cuda")

        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_tokens,
            use_cache=True,
            temperature=None,  # Always pick the most likely token
            top_p=None,  # Disable nucleus sampling
            top_k=5
        )
        results = tokenizer.batch_decode(outputs)

        all_results = []
        for result, meta in zip(results, batch_metadata):
            pattern = model_details["pattern"]
            matches = re.findall(pattern, result, re.DOTALL)

            if not matches:
                fallback_pattern = r"<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eom_id\|>"
                fallback_match = re.findall(fallback_pattern, result, re.DOTALL)
                if fallback_match:
                    output = clean_generated_json(fallback_match[0])
                else:
                    print(f'INVALID FORMAT {result}', flush=True)
                    output = clean_generated_json(result)
            else:
                raw_response = matches[-1]
                raw_response = clean_generated_json(raw_response)
                decoded_object = json_repair.repair_json(raw_response)
                if type(decoded_object) is list:
                    output = json.dumps(decoded_object[-1])
                else:
                    output = decoded_object
            meta["rationales"] = output
            all_results.append(meta)

        batch_df = pd.DataFrame(all_results)
        batch_df.to_csv(
            output_file,
            sep="\t",
            index=False,
            mode="a" if file_exists else "w",  # Append mode if file exists
            header=not file_exists  # Only write header if file is new
        )
        print(f"Saved {i + len(batch_messages)}!", flush=True)
        file_exists = True


if __name__ == "__main__":
    main()
