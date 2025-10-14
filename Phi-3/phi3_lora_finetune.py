import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os
import getpass
from huggingface_hub import login
import time

# print("Starting Phi-3 LoRA Fine-tuning...")
# print("")

# Authenticate with Hugging Face
print("Authenticating with Hugging Face")
hf_token = getpass.getpass("Enter your HF Token: ")
if not hf_token:
    raise ValueError("Hugging Face token is required to access Phi-3 model")

print("   Logging in to Hugging Face...")
login(token=hf_token)
print("   Successfully authenticated!")

print("\nSetting up memory optimizations")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()
print("   Memory optimized!")

# Configure 4-bit quantization for memory efficiency
print("\nConfiguring 4-bit quantization")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
print("   Quantization configured")

# base model/tokenizer with quantization
model_name = "microsoft/Phi-3-mini-4K-instruct"
print(f"\nLoading tokenizer and model ({model_name})")
# print("   Loading tokenizer...")

start_time = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"   Tokenizer loaded ({time.time() - start_time:.1f}s)")

# print("   Loading model with 4-bit quantization...")
# print("  downloading model...")
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
print(f"  Model loaded successfully! ({time.time() - start_time:.1f}s)")

# Prepare for LoRA
print("\nPreparing LoRA configuration")
base_model = prepare_model_for_kbit_training(base_model)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(base_model, lora_config)
model.gradient_checkpointing_enable()
print("   LoRA configuration applied")

# Load and preprocess dataset
print("\nLoading and preprocessing dataset")
dataset = load_dataset("csv", data_files={"train": "phi3_lora_train.csv"})
def preprocess(examples):
    texts = []
    labels = []

    for i in range(len(examples['instruction'])):
        text = f"<|user|>\n{examples['instruction'][i]}\nURL: {examples['input'][i]}<|end|>\n<|assistant|>\n"
        label = f"{examples['output'][i]}<|end|>"

        # Tokenize input and label
        full_text = text + label
        tokenized_full = tokenizer(full_text, truncation=True, max_length=512, padding=False)
        tokenized_input = tokenizer(text, truncation=True, max_length=384, padding=False)

        # ignore input tokens, only train on output
        labels_ids = [-100] * len(tokenized_input["input_ids"]) + tokenized_full["input_ids"][len(tokenized_input["input_ids"]):]

        # Pad to max length
        if len(tokenized_full["input_ids"]) < 512:
            pad_length = 512 - len(tokenized_full["input_ids"])
            tokenized_full["input_ids"].extend([tokenizer.pad_token_id] * pad_length)
            tokenized_full["attention_mask"].extend([0] * pad_length)
            labels_ids.extend([-100] * pad_length)

        texts.append(tokenized_full)
        labels.append(labels_ids)

    return {
        "input_ids": [t["input_ids"] for t in texts],
        "attention_mask": [t["attention_mask"] for t in texts],
        "labels": labels
    }

# print("   Tokenizing and processing dataset...")
start_time = time.time()
tokenized_ds = dataset["train"].map(preprocess, batched=True, batch_size=100, remove_columns=dataset["train"].column_names)
print(f"   Dataset processed ({time.time() - start_time:.1f}s)")

# Training arguments with memory optimizations
print("\nConfiguring training arguments")
training_args = TrainingArguments(
    output_dir="./phi3_lora_finetuned",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    num_train_epochs=2,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=500,
    save_strategy="steps",
    warmup_steps=100,
    max_grad_norm=1.0,
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    group_by_length=True,
    ddp_find_unused_parameters=False,
)
# print("   Training arguments configured")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds,
    tokenizer=tokenizer
)

# Start training
print("\nStarting model training")
trainer.train()

# Save adapters
print("\nSaving the trained model")
model.save_pretrained("./phi3_lora_finetuned")
tokenizer.save_pretrained("./phi3_lora_finetuned")
print("   Model saved successfully!")
