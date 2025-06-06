import argparse
import os
import re

import pandas as pd
from tqdm import tqdm
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

from utils import build_few_shots, PROMPT, LABELS_PROMPT

batch_size = 24


model_mapping = {
    "llama3.1-8b": {
        "name": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "chat_template": "llama-3",
        "pattern": r"<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>"
    },
    "llama3.3-70": {
        "name": "unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
        "chat_template": "llama-3",
        "pattern": r"<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>",
        "instruction_part": "<|start_header_id|>user<|end_header_id|>\n\n",
        "response_part": "<|start_header_id|>assistant<|end_header_id|>\n\n"
    },
    "nemo": {
        "name": "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
        "chat_template": "mistral",
        "pattern": r"\[\/INST\](.*?)<\/s>"
    },
}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run LLM inference on a dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset (CSV/TSV)")
    parser.add_argument("--model", type=str, required=True, help="Model to use", choices=model_mapping.keys())
    parser.add_argument("--output", type=str, default="rephrased-rationales.csv", help="Output file path")
    parser.add_argument("--mode", type=str, default="rationales", choices=["labels", "rationales"], help="Mode of operation: 'rationales' or 'labels'")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token for model access")
    return parser.parse_args()


def main():
    print('Starting...', flush=True)

    args = parse_arguments()
    max_seq_length = 4096

    df = pd.read_csv(f'{args.dataset}')

    sentences = df['text'].to_list()

    messages = []
    few_shot_examples = pd.read_csv(f"few_shot_examples.csv")
    few_shot = build_few_shots(few_shot_examples)

    for sentence in sentences:
        if args.mode == "rationales":
            messages.append(few_shot + [{"from": "human", "value": PROMPT.format(sentence)}])
        elif args.mode == "labels":
            messages.append(few_shot + [{"from": "human", "value": LABELS_PROMPT.format(sentence)}])
        else:
            raise ValueError("Invalid mode. Choose either 'rationales' or 'labels'.")


    model_details = model_mapping[args.model]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_details['name'],
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

    print('Messages to inference size:', len(messages), flush=True)
    generated = []
    for message in tqdm(messages):
        inputs = tokenizer.apply_chat_template(
            message,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")

        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=256,
            use_cache=True,
            temperature=None, # Always pick the most likely token
            top_p=None, # Disable nucleus sampling
            top_k=5
        )
        result = tokenizer.batch_decode(outputs)
        pattern = model_details["pattern"]
        matches = re.findall(pattern, result[0], re.DOTALL)
        if len(matches) == 0:
            generated.append(result[0])
            continue
        output = matches[-1].replace('\r', '').replace('\n', '')
        generated.append(output)

    predictions_file = f"{args.output}"
    if os.path.exists(predictions_file):
        df_existing = pd.read_csv(predictions_file)
        df_existing['prediction'] = generated
        df_existing.to_csv(predictions_file, index=False)
        print(f"Updated existing file: {predictions_file}", flush=True)
    else:
        df = df[['id', 'text', 'hyperpartisan_gold_label', 'prct_gold_label', 'stance_gold_label']]
        df['prediction'] = generated
        df.to_csv(predictions_file, index=False)
        print(f"Created new predictions file: {predictions_file}", flush=True)


if __name__ == "__main__":
    main()
