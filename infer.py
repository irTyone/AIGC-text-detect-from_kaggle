import os
import torch
import argparse
import numpy as np
import pandas as pd
import json

from datasets import Dataset
from scipy.special import expit
from transformers import (
    DataCollatorWithPadding, TrainingArguments,
    AutoTokenizer, AutoModelForSequenceClassification, Trainer
)

from accelerate import Accelerator
accelerator = Accelerator()


def preprocess_function(examples, max_length, tokenizer):
    tokenized_samples = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length
    )
    return tokenized_samples


def main(args):

    # ✅ 读取 JSON（你的关键修改）
    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # list -> pandas
    test_df = pd.DataFrame(data)

    accelerator.print(f'Test data shape: {test_df.shape}')

    # 如果没有 id 字段，自动生成
    if "id" not in test_df.columns:
        test_df["id"] = np.arange(len(test_df))

    test_ds = Dataset.from_pandas(test_df)

    ## Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)

    test_tokenized_ds = test_ds.map(
        preprocess_function,
        batched=True,
        fn_kwargs={
            "max_length": args.max_length,
            "tokenizer": tokenizer
        },
        remove_columns=test_ds.column_names
    )

    for idx in range(2):
        accelerator.print(f"\n--- Sample {idx} ---\n")
        accelerator.print(
            tokenizer.decode(test_tokenized_ds[idx]["input_ids"])
        )

    ## Load Model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model_path,
        num_labels=1
    )

    model = accelerator.prepare(model)
    accelerator.print("### Loaded Model Weights ###")

    ## Trainer Setup
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="longest"
    )

    training_args = TrainingArguments(
        output_dir="tmp",
        per_device_eval_batch_size=8,
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    ## predictor
    pred_output = trainer.predict(test_tokenized_ds)

    logits = pred_output.predictions.astype(float)

    # probs = expit(logits)[:, 0]
    probs = -1 * logits[:, 0]

    sub = pd.DataFrame({
        "id": test_df['id'].values,
        "generated": probs
    })

    os.makedirs(args.save_dir, exist_ok=True)

    save_path = os.path.join(args.save_dir, f"{args.model_id}.parquet")

    sub.to_parquet(save_path)

    accelerator.print(f"Saved to {save_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument('--base_model_path', type=str, required=True)
    ap.add_argument('--max_length', type=int, required=True)
    ap.add_argument('--input_json', type=str, required=True)   # ✅ 新增
    ap.add_argument('--save_dir', type=str, default="./outputs")
    ap.add_argument('--model_id', type=str, required=True)

    args = ap.parse_args()

    main(args)