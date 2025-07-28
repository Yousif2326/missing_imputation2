import os
train_file = "missing_imputation/data/train_react_imputation_500_per_col_TopK_4_all_timepoints.jsonl"
test_file = "missing_imputation/data/test_react_imputation_100_per_col_TopK_4_all_timepoints.jsonl"

learning_rates = ["3e-5"]
epochs = [2]
data_nums = [20000]

# Batch size and gradient accumulation configurations to test
batch_configs = [
    (1, 8),    # Moderate batch, moderate accumulation
]
      # Even higher rank

# LoRA configurations to test
lora_configs = [
    (32, 64, 0.1, "q_proj,v_proj")
]

for data_num in data_nums:
    for lr in learning_rates:
        for ep in epochs:
            for rank, alpha, dropout, modules in lora_configs:
                for batch_size, grad_accum in batch_configs:
                    bs = f"ep{ep}"
                    modules_short = modules.replace(",", "_").replace("_proj", "")
                    name = f"ans_test{lr}_{bs}_n{data_num}_r{rank}_a{alpha}_d{dropout}_{modules_short}_b{batch_size}_g{grad_accum}"
                    output_dir = f"LoraFineTune/FinalFinalTopK3WH_{name}"
                    cmd = f"sbatch /home/ityousif/projects/def-emohamme/ityousif/run_train.slurm {lr} {ep} {output_dir} {data_num} {rank} {alpha} {dropout} '{modules}' {batch_size} {grad_accum} {train_file} {test_file}"
                    print(f"Submitting: {cmd}")
                    os.system(cmd)  # ⬅️ Actually runs the sbatch command
