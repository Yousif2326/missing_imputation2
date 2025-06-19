from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
import torch
from torch.utils.data import Dataset
import json
import os
import re

# Load dataset
dataset_path = "/imputation_lora_dataset.jsonl"
with open(dataset_path, "r") as f:
    all_data = [json.loads(line) for line in f]

# Load model/tokenizer
model_id = "/llama3.2-3b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)

# Apply LoRA
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(base_model, peft_config)

class ImputationDataset(Dataset):
    def __init__(self, data):
        self.data = []
        for item in data:
            task_instruction = " Respond ONLY with a number (e.g., 2.0). Do NOT explain."

            prompt = f"<s>[INST] {item['messages'][1]['content'].strip()}{task_instruction} [/INST]"
            label = item['messages'][2]['content']
            full_text = prompt + " " + label

            enc = tokenizer(
                full_text,
                padding="max_length",
                truncation=True,
                max_length=4096,
                return_tensors="pt"
            )

            input_ids = enc["input_ids"].squeeze()
            attention_mask = enc["attention_mask"].squeeze()

            # Create labels with prompt part masked as -100
            label_ids = input_ids.clone()
            
            
            prompt_len = tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)["input_ids"].size(1)

            label_ids[:prompt_len] = -100  # ignore prompt in loss calculation

            self.data.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": label_ids
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Split
split_idx = int(0.8 * len(all_data))
train_dataset = ImputationDataset(all_data[:split_idx])
eval_dataset = ImputationDataset(all_data[split_idx:])

# Training arguments
training_args = TrainingArguments(
    output_dir="/lora-debug-output",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    fp16=True,
    num_train_epochs=1,
    logging_dir="./logs",
    report_to="none" 
)

# Accuracy metric
def extract_number(text):
    match = re.search(r"(\d+(\.\d+)?)", text)
    return match.group(1) if match else None

def compute_metrics(_):
    correct = 0
    total = len(eval_dataset)

    for i in range(total):
        example = eval_dataset[i]
        input_ids = example["input_ids"].unsqueeze(0).to(model.device)
        attention_mask = example["attention_mask"].unsqueeze(0).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=2048,
                pad_token_id=tokenizer.eos_token_id
            )

        # Trim off prompt
        gen_tokens = output_ids[0][input_ids.shape[-1]:]
        pred_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        pred_number = extract_number(pred_text)

        true_number = extract_number(all_data[8 + i]["messages"][2]["content"])
        print(f"Raw Model Output: '{pred_text.strip()}'")
        print(f"Extracted Prediction: '{pred_number}' | True Label: '{true_number}'\n")

        if pred_number == true_number:
            correct += 1

    accuracy = correct / total
    return {"accuracy": accuracy}

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
metrics = trainer.evaluate()
print("Eval Metrics:", metrics)


