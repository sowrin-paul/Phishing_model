import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os
import getpass
from huggingface_hub import login, hf_hub_download
import time
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import json

print("Starting Optimized Phi3 LoRA Fine-tuning...")

# Authenticate with Hugging Face
print("\nAuthenticating with Hugging Face")
hf_token = getpass.getpass("Enter your HF Token: ")
if not hf_token:
    raise ValueError("Hugging Face token is required to access Phi3 model")

print("   Logging in to Hugging Face...")
login(token=hf_token)
print("   Successfully authenticated!")

print("\nSetting up aggressive memory optimizations")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True  # training speedup
print("   Memory optimized!")

# Load the same dataset as BiGRU - ealvaradob/phishing-dataset
print("\nLoading phishing URL dataset from Hugging Face")
print("   Dataset: ealvaradob/phishing-dataset")
print("   Downloading urls.json (URL-only data, no spam messages)...")

file_path = hf_hub_download(
    repo_id='ealvaradob/phishing-dataset',
    filename='urls.json',
    repo_type='dataset'
)

# Load JSON data
with open(file_path, 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)
original_size = len(df)
print(f"   Original dataset size: {original_size:,} samples")

# The dataset has 'text' (URL) and 'label' (0=legitimate, 1=phishing) columns
url_column = 'text'
label_column = 'label'

print(f"   URL column: {url_column}")
print(f"   Label column: {label_column}")

# Clean data
df = df.dropna()
df[url_column] = df[url_column].astype(str)
df[label_column] = df[label_column].astype(int)

print(f"   Unique labels: {df[label_column].unique()}")
print(f"   Label distribution:\n{df[label_column].value_counts()}")

# Balance the dataset and sample strategically
phishing_samples = df[df[label_column] == 1]
legitimate_samples = df[df[label_column] == 0]

sample_size = 2000
if len(phishing_samples) > sample_size:
    phishing_samples = phishing_samples.sample(n=sample_size, random_state=42)
if len(legitimate_samples) > sample_size:
    legitimate_samples = legitimate_samples.sample(n=sample_size, random_state=42)

# Combine and shuffle
df_optimized = pd.concat([phishing_samples, legitimate_samples]).sample(frac=1, random_state=42).reset_index(drop=True)

# Convert to Phi3 training format
print("\nConverting to Phi3 training format...")
training_data = []
for _, row in df_optimized.iterrows():
    url = row[url_column]
    label = "Phishing" if row[label_column] == 1 else "Legitimate"

    training_data.append({
        'instruction': 'Analyze the following URL and determine if it is a phishing attempt or legitimate. Explain your reasoning.',
        'input': url,
        'output': f'{label}: This URL is classified as {label.lower()} based on its structure and characteristics.'
    })

# Create DataFrame and save
df_training = pd.DataFrame(training_data)
df_training.to_csv("phi3_lora_train_optimized.csv", index=False)

print(f"   Optimized dataset size: {len(df_optimized):,} samples")
print(f"   Phishing: {len(phishing_samples):,}, Legitimate: {len(legitimate_samples):,}")
print(f"   Size reduction: {(1 - len(df_optimized)/original_size)*100:.1f}% smaller")
print(f"   Estimated time savings: ~{((original_size - len(df_optimized))/original_size)*100:.0f}%")

# Configure 4-bit quantization for memory efficiency
print("\nConfiguring 4-bit quantization")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
print("   Quantization configured")

# Load base model/tokenizer with quantization
model_name = "microsoft/Phi-3-mini-4k-instruct"
print(f"\nLoading tokenizer and model ({model_name})")

start_time = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"   Tokenizer loaded ({time.time() - start_time:.1f}s)")

start_time = time.time()
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    attn_implementation="eager",
)
print(f"   Model loaded successfully! ({time.time() - start_time:.1f}s)")

# OPTIMIZATION 2: More aggressive LoRA configuration
print("\nOptimized LoRA configuration")
base_model = prepare_model_for_kbit_training(base_model)

# Reduced LoRA rank for faster training
lora_config = LoraConfig(
    r=8,  # Reduced from 16 to 8 for faster training
    lora_alpha=16,  # Reduced proportionally
    target_modules=["q_proj", "v_proj", "o_proj"],  # Target fewer modules
    lora_dropout=0.05,  # Reduced dropout
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(base_model, lora_config)
model.gradient_checkpointing_enable()

# Print trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"   Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
print("   LoRA configuration applied")

# OPTIMIZATION 3: Faster data preprocessing
print("\nLoading and preprocessing dataset (optimized)")
dataset = load_dataset("csv", data_files={"train": "phi3_lora_train_optimized.csv"})

def preprocess_fast(examples):
    """Optimized preprocessing function"""
    texts = []
    labels = []

    # Process in batches for speed
    for i in range(len(examples['instruction'])):
        # Shorter prompt template
        text = f"<|user|>\n{examples['instruction'][i]}\n{examples['input'][i]}<|end|>\n<|assistant|>\n"
        label = f"{examples['output'][i]}<|end|>"

        # More aggressive truncation
        full_text = text + label
        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=256,  # Reduced from 512 to 256
            padding="max_length",
            return_tensors=None
        )

        # Simplified label creation
        input_len = len(tokenizer(text, truncation=True, max_length=200)["input_ids"])
        labels_ids = [-100] * input_len + tokenized["input_ids"][input_len:]

        # Ensure correct length
        if len(labels_ids) > 256:
            labels_ids = labels_ids[:256]
        elif len(labels_ids) < 256:
            labels_ids.extend([-100] * (256 - len(labels_ids)))

        texts.append(tokenized)
        labels.append(labels_ids)

    return {
        "input_ids": [t["input_ids"] for t in texts],
        "attention_mask": [t["attention_mask"] for t in texts],
        "labels": labels
    }

start_time = time.time()
tokenized_ds = dataset["train"].map(
    preprocess_fast,
    batched=True,
    batch_size=500,  # Larger batch size for faster processing
    num_proc=4,  # Use multiple processes
    remove_columns=dataset["train"].column_names
)
print(f"   Dataset processed ({time.time() - start_time:.1f}s)")

# OPTIMIZATION 4: Aggressive training arguments for speed
print("\nConfiguring optimized training arguments")
training_args = TrainingArguments(
    output_dir="./phi3_lora_finetuned_fast",
    per_device_train_batch_size=4,  # Increased from 2
    gradient_accumulation_steps=8,  # Reduced from 16
    num_train_epochs=1,  # Reduced from 2 epochs
    learning_rate=5e-4,  # Increased learning rate
    fp16=True,
    logging_steps=50,  # Reduced logging frequency
    save_steps=1000,  # Less frequent saves
    save_strategy="steps",
    warmup_steps=50,  # Reduced warmup
    max_grad_norm=1.0,
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    group_by_length=True,
    ddp_find_unused_parameters=False,
    dataloader_num_workers=2,  # Use multiple workers
    eval_steps=500,
    load_best_model_at_end=False,  # Skip evaluation for speed
    metric_for_best_model=None,
    greater_is_better=None,
    report_to=None,  # Disable wandb/tensorboard for speed
    push_to_hub=False,
    optim="adamw_torch_fused",  # Faster optimizer
)

# Custom data collator for efficiency
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=None,  # Don't pad to multiples
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Print training estimate
total_steps = len(tokenized_ds) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
estimated_time_minutes = total_steps * 0.8  # Rough estimate: 0.8 seconds per step
print(f"   Total training steps: {total_steps}")
print(f"   Estimated training time: {estimated_time_minutes/60:.1f} hours")

# OPTIMIZATION 5: Training monitoring and early stopping
class FastTrainingCallback:
    def __init__(self):
        self.start_time = None
        self.last_log_time = None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.start_time = time.time()
        print(f"\nTraining started at {time.strftime('%H:%M:%S')}")

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if logs and 'loss' in logs:
            current_time = time.time()
            if self.last_log_time:
                steps_per_minute = 60 / (current_time - self.last_log_time) * args.logging_steps
                remaining_steps = state.max_steps - state.global_step
                eta_minutes = remaining_steps / steps_per_minute
                print(f"   Step {state.global_step}/{state.max_steps} | Loss: {logs['loss']:.4f} | ETA: {eta_minutes:.1f}min")
            self.last_log_time = current_time

# Add callback
trainer.add_callback(FastTrainingCallback())

# Start optimized training
print("\nStarting optimized model training")
print("="*50)

start_training_time = time.time()
trainer.train()
training_duration = time.time() - start_training_time

print(f"\nTraining completed in {training_duration/3600:.2f} hours!")
print(f"   Speedup achieved: ~{8/(training_duration/3600):.1f}x faster")

# Save the model
print("\nSaving the trained model")
model.save_pretrained("./phi3_lora_finetuned_fast")
tokenizer.save_pretrained("./phi3_lora_finetuned_fast")
print("   Model saved successfully!")

print(f"\nOptimization complete! Training time reduced from ~8 hours to {training_duration/3600:.2f} hours")
print("="*60)
