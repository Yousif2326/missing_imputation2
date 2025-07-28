
print("new")
print("top4K")
import json
import re
from datasets import Dataset
from transformers import AutoTokenizer
import os
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
import torch
import argparse
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from scipy.stats import rankdata
print("done")
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--train_epochs", type=int, default=4)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--data_num", type=int, required=3000)
# New LoRA arguments
parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank (r parameter)")
parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha parameter")
parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout rate")
parser.add_argument("--target_modules", type=str, default="q_proj,v_proj", 
                    help="Comma-separated list of target modules for LoRA")
# New batch size and gradient accumulation arguments
parser.add_argument("--batch_size", type=int, default=4, help="Per device batch size")
parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
parser.add_argument("--train_file", type=str, required=True, help="Path to train JSONL file")
parser.add_argument("--test_file", type=str, required=True, help="Path to test JSONL file")
print("without history")
args = parser.parse_args()
# Add these arguments at the top with the others
print("\n🔧 Parsed Arguments:")
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")
# Then, after parsing args, load the data:
with open(args.train_file, "r") as f:
    train_data = [json.loads(line) for line in f][:1000]  # Load only first 1000 rows
with open(args.test_file, "r") as f:
    test_data = [json.loads(line) for line in f][:500]  # Load only first 500 rows
print(f"Loaded {len(train_data)} train and {len(test_data)} test examples.")

# Load and filter examples where 'gp1' is the only masked column
# with open(args.train_file, "r") as f:
#     all_train_data = [json.loads(line) for line in f]
#     train_data = [ex for ex in all_train_data if ex.get("masked_columns", [])[0] == "gp1"]

# with open(args.test_file, "r") as f:
#     all_test_data = [json.loads(line) for line in f]
#     test_data = [ex for ex in all_test_data if ex.get("masked_columns", [])[0] == "gp1"]

#print(f"Filtered to {len(train_data)} train and {len(test_data)} test examples with gp1 masked.")
dataset = Dataset.from_list(train_data)

# Load tokenizer
local_path = "Llama-3.2-3B-Instruct"  # change this!

tokenizer = AutoTokenizer.from_pretrained(local_path, use_fast=True)
# Add special tokens
# Fix padding issue for LLaMA tokenizer
tokenizer.pad_token = tokenizer.eos_token

# Add special tokens
 
# Tokenize

for i, ex in enumerate(train_data[:10]):
    full_text = ex["input"] + "\n" + ex["output"]
    tokens = tokenizer(full_text, truncation=False)
    length = len(tokens["input_ids"])
    print(f"Example {i} has {length} tokens")
    if length > 512:
        print("⚠️ This will be truncated!\n")

import re

def tokenize(example):
    # Remove the <instructions>...</instructions> block from the input
    input_cleaned = re.sub(r"<instructions>.*?</instructions>", "", example["input"], flags=re.DOTALL).strip()

    # LLaMA 3 chat-style system message
    system_message = (
        "You are an AI assistant trained using the ReAct approach (Reason + Act). "
        "Your goal is to impute missing patient survey responses by reasoning through the most relevant predictor variables. "
        "All values range from 0 to 4: "
        "- 0 means 'not at all' "
        "- 1 means 'a little bit' "
        "- 2 means 'somewhat' "
        "- 3 means 'quite a bit' "
        "- 4 means 'very much'. "
        "First, think step-by-step and explain your reasoning inside <think>...</think> tags. "
        "Then, give your final imputed answer in JSON format inside <answer>...</answer>."
    )

    # Format into LLaMA 3 chat-style prompt
    chat_prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{system_message}<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{input_cleaned}<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
        f"{example['output'].strip()}"
    )
    print("\n🔍 Final Chat-Formatted Prompt:\n")
    print(chat_prompt)
    print("-" * 80)
    
    # Tokenize the full prompt
    tokenized = tokenizer(chat_prompt, padding="max_length", max_length=512, truncation=True)

    # Mask labels so only the assistant output is learned
    input_ids_only = tokenizer(
        chat_prompt.rsplit(example["output"].strip(), 1)[0],
        truncation=True,
        max_length=512
    )["input_ids"]

    input_len = len(input_ids_only)
    labels = [-100] * input_len + tokenized["input_ids"][input_len:]
    labels = labels[:512] + [-100] * max(0, 512 - len(labels))
    tokenized["labels"] = labels

    return tokenized

# def tokenize(example):
#     input_text = example["input"]
#     output_text = example["output"]
#     full_text = input_text + "\n" + output_text

#     # Tokenize full sequence
#     tokenized = tokenizer(full_text, padding="max_length", max_length=512, truncation=True)
    
#     # Get length of input tokens only
#     input_ids = tokenizer(input_text, truncation=True, max_length=512)["input_ids"]
#     input_len = len(input_ids)

#     # Create labels: mask input tokens with -100
#     labels = [-100] * input_len + tokenized["input_ids"][input_len:]
#     labels = labels[:512]  # truncate if needed
#     labels += [-100] * (512 - len(labels))  # pad to 512 tokens

#     tokenized["labels"] = labels
#     return tokenized

for i, example in enumerate(dataset.select(range(3))):
    tokenize(example)

tokenized_dataset = dataset.map(tokenize, batched=False)

#---------------------------------------------------------------------------------------------------------

# Load model
model = AutoModelForCausalLM.from_pretrained(local_path)

# Parse target modules from comma-separated string
target_modules_list = [module.strip() for module in args.target_modules.split(",")]

# LoRA configuration
lora_config = LoraConfig(
    r=args.lora_rank,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    target_modules=target_modules_list,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Apply LoRA
model = get_peft_model(model, lora_config)




training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    num_train_epochs=args.train_epochs,
    learning_rate=args.learning_rate,
    save_steps=200,
    save_total_limit=2,
    logging_steps=20,
    fp16=True,  # if using GPU
)
print("\n📦 LoRA Config:")
print(lora_config)

print("\n⚙️ Training Arguments:")
print(training_args)
# For autoregressive language modeling
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=collator,
    tokenizer=tokenizer
)


trainer.train()

trainer.model.eval() 



print("Training complete. Evaluating model...")


def normalize(val):
    if isinstance(val, (int, float)):
        return int(val)  # Remove decimal point (truncate)
    return val


def generate_response(prompt: str, model, tokenizer, max_new_tokens=300):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    print("hello")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)



    # Extract all <answer>...</answer> blocks
    matches = re.findall(r"<answer>(.*?)</answer>", decoded, re.DOTALL)
    answer_raw = matches[-1].strip() if matches else None
    
    if answer_raw:
        print("\n🔍 Raw extracted string (repr):\n", repr(answer_raw))
        try:
            answer_cleaned = answer_raw.strip().replace("\n", "").replace("\t", "").replace("<|eot_id|>", "")
            answer_dict = json.loads(answer_cleaned)
        except json.JSONDecodeError as e:
            print("\n⚠️ Found <answer> but failed to parse JSON:\n", answer_raw)
            print("⚠️ JSONDecodeError:", e)
            answer_dict = None
    else:
        print("\n⚠️ No <answer>...</answer> section found.")
        answer_dict = None

    return decoded, answer_raw, answer_dict

def evaluate_model_on_dataset(data, model, tokenizer):
    results = []

    for i, example in enumerate(data):
        prompt = build_llama3_chat_prompt(example["input"])        
        true_answer_block = example["output"]
        print("\n🧪 Prompt used for generation:\n", prompt)
        # Get prediction
        print(f"\n🔹 Evaluating example {i + 1}/{len(data)}")
        _, _, pred_dict = generate_response(prompt, model, tokenizer)

        # Parse true answer from <answer>{...}</answer>
        true_match = re.search(r"<answer>(.*?)</answer>", true_answer_block, re.DOTALL)
        true_json = true_match.group(1).strip() if true_match else None

        try:
            true_dict = json.loads(true_json)
        except json.JSONDecodeError:
            print("❌ Could not parse ground truth JSON.")
            true_dict = None

        # Extract values for comparison
        pred_value = list(pred_dict.values())[0] if pred_dict else None
        true_value = list(true_dict.values())[0] if true_dict else None

        # Normalize both values using rounding
        pred_value_normalized = normalize(pred_value)
        true_value_normalized = normalize(true_value)

        print(f"➡️ Predicted: {pred_value_normalized}, Ground truth: {true_value_normalized}")

        results.append({
            "index": i,
            "masked_column": example["masked_columns"][0],
            "true": true_value_normalized,
            "pred": pred_value_normalized,
            "match": pred_value_normalized == true_value_normalized,
        })
    return results

def compute_metrics(results):
    total = len(results)
    correct = sum(1 for r in results if r["match"])
    accuracy = correct / total if total > 0 else 0
    
    # Extract true and predicted values for regression metrics
    true_values = [r["true"] for r in results if r["true"] is not None and r["pred"] is not None]
    pred_values = [r["pred"] for r in results if r["true"] is not None and r["pred"] is not None]
    
    # Calculate MAE and RMSE
    mae = mean_absolute_error(true_values, pred_values) if true_values else 0
    rmse = np.sqrt(mean_squared_error(true_values, pred_values)) if true_values else 0
    
    # Calculate AUC (treating as binary classification: correct vs incorrect)
    # For AUC, we need binary labels and probabilities
    binary_labels = [1 if r["match"] else 0 for r in results]
    # Use normalized predictions as "probabilities" (closer to true = higher probability)
    # Convert to 0-1 scale by normalizing the absolute difference
    max_diff = 4  # Maximum possible difference (0 to 4 scale)
    probabilities = [1 - abs(r["pred"] - r["true"]) / max_diff if r["true"] is not None and r["pred"] is not None else 0.5 for r in results]
    
    try:
        auc = roc_auc_score(binary_labels, probabilities) if len(set(binary_labels)) > 1 else 0.5
    except ValueError:
        auc = 0.5  # Default to 0.5 if AUC cannot be computed
    
    # Calculate average rank
    # For each example, calculate the rank of the predicted value among all possible values (0-4)
    ranks = []
    for r in results:
        if r["true"] is not None and r["pred"] is not None:
            # Create array of all possible values (0-4)
            all_values = list(range(5))
            # Find the rank of the predicted value
            pred_rank = rankdata([abs(v - r["true"]) for v in all_values]).tolist()[all_values.index(r["pred"])]
            ranks.append(pred_rank)
    
    avg_rank = np.mean(ranks) if ranks else 0
    
    print(f"\n📊 Evaluation Metrics:")
    print(f"✅ Accuracy: {accuracy:.2%} ({correct}/{total})")
    print(f"📏 MAE: {mae:.3f}")
    print(f"📐 RMSE: {rmse:.3f}")
    print(f"📈 AUC: {auc:.3f}")
    print(f"🏆 Average Rank: {avg_rank:.3f}")
    
    return {
        "accuracy": accuracy,
        "mae": mae,
        "rmse": rmse,
        "auc": auc,
        "avg_rank": avg_rank,
        "total_examples": total,
        "correct_predictions": correct
    }


def build_llama3_chat_prompt(user_input: str) -> str:
    # Remove the <instructions>...</instructions> block
    input_cleaned = re.sub(r"<instructions>.*?</instructions>", "", user_input, flags=re.DOTALL).strip()

    system_message = (
        "You are an AI assistant trained using the ReAct approach (Reason + Act). "
        "Your goal is to impute missing patient survey responses by reasoning through the most relevant predictor variables. "
        "All values range from 0 to 4: "
        "- 0 means 'not at all' "
        "- 1 means 'a little bit' "
        "- 2 means 'somewhat' "
        "- 3 means 'quite a bit' "
        "- 4 means 'very much'. "
        "First, think step-by-step and explain your reasoning inside <think>...</think> tags. "
        "Then, give your final imputed answer in JSON format inside <answer>...</answer>."
    )

    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{system_message}<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{input_cleaned}<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )


# Load data


# Run evaluation
results = evaluate_model_on_dataset(test_data, trainer.model, tokenizer)

metrics = compute_metrics(results)

model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)

# Save results summary
summary = {
    "output_dir": args.output_dir,
    "learning_rate": args.learning_rate,
    "train_epochs": args.train_epochs,
    "data_num": args.data_num,
    "lora_rank": args.lora_rank,
    "lora_alpha": args.lora_alpha,
    "lora_dropout": args.lora_dropout,
    "target_modules": args.target_modules,
    "batch_size": args.batch_size,
    "gradient_accumulation_steps": args.gradient_accumulation_steps,
    "accuracy": metrics["accuracy"],
    "mae": metrics["mae"],
    "rmse": metrics["rmse"],
    "auc": metrics["auc"],
    "avg_rank": metrics["avg_rank"],
    "results": results
}

os.makedirs("results_summary", exist_ok=True)

summary_path = f"results_summary/new_results/{args.output_dir.replace('/', '_')}.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"📊 Saved summary to {summary_path}")