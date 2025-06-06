import argparse
import json
import os

import pandas as pd
import wandb
from datasets import Dataset
from transformers import (
    TrainingArguments,
)
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template


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
    parser = argparse.ArgumentParser(description="Run LLM fine-tuning on a dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset (CSV/TSV)")
    parser.add_argument("--model", type=str, required=True, help="Model to use", choices=model_mapping.keys())
    parser.add_argument("--new_model_name", type=str, default="new_model.tsv", help="Name of the new fine-tuned model.")
    parser.add_argument("--mode", type=str, default="rationales", choices=["labels", "rationales"], help="Mode of operation: 'rationales' or 'labels'")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token for model access")
    return parser.parse_args()


def row_to_list(row):
    gpt_json = {
        "chain_of_thought": row["rationales"],
        "hyperpartisan": row["hyperpartisan"],
        "prct": row["prct"],
        "stance": row["stance"],
    }

    return [
        {"from": "human", "value": row["text"]},
        {"from": "gpt", "value": json.dumps(gpt_json)}
    ]


def to_bool(value):
    return bool(int(value))


def main():
    print('Starting...', flush=True)

    args = parse_arguments()

    df = pd.read_csv(f'{args.dataset}')

    df["hyperpartisan"] = df["hyperpartisan_gold_label"].apply(to_bool)
    df["prct"] = df["prct_gold_label"].apply(to_bool)
    df["stance"] = df["stance_gold_label"]

    model_details = model_mapping[args.model]

    max_seq_length = 4096
    seed = 47
    base_model_id = model_details["name"]
    new_model = args.new_model_name

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_id,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
        token=args.hf_token
    )

    model = FastLanguageModel.get_peft_model(
        model=model,
        r=8,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        lora_alpha=16,
        lora_dropout=0,
        bias='none',
        use_gradient_checkpointing='unsloth',
        random_state=seed,
        max_seq_length=max_seq_length,
        use_rslora=False,
    )


    df["conversations"] = df.apply(row_to_list, axis=1)
    df["conversations"] = df["conversations"].apply(json.dumps)
    df_conversations = df[["conversations"]]
    dataset = Dataset.from_pandas(df_conversations)

    tokenizer = get_chat_template(
        tokenizer,
        chat_template=model_details["chat_template"],
        mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
    )


    def formatting_prompts_func(examples):
        conversations = [json.loads(c) for c in examples["conversations"]]
        texts = [
            tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
            for c in conversations
        ]
        return {"text": texts}


    dataset = dataset.map(formatting_prompts_func, batched=True)

    wandb.login(key='')
    wandb_project = (new_model)

    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project

    training_arguments = TrainingArguments(
        output_dir=".outputs",
        num_train_epochs=2,
        per_device_train_batch_size=12,
        gradient_accumulation_steps=4,
        optim="adamw_8bit",
        save_steps=100,
        logging_steps=10,
        learning_rate=2e-5,
        weight_decay=0.01,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        warmup_ratio=0.03,
        lr_scheduler_type="linear",
        seed=seed,
        report_to="wandb"
    )

    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset,
        dataset_text_field="text",
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False
    )

    print('Train!', flush=True)
    trainer.train()

    model.save_pretrained(f"{new_model}")
    tokenizer.save_pretrained(f"{new_model}")
    wandb.finish()
    model.config.use_cache = True
    print('Fine-tuning finished!', flush=True)


if __name__ == "__main__":
    main()
