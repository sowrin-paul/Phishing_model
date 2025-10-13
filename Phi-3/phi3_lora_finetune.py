import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import login, HfApi
import os
import getpass

# Better token handling with validation and secure input
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    print("HF_TOKEN environment variable not found")
    print("Please enter your Hugging Face token (input will be hidden):")
    hf_token = getpass.getpass("HF Token: ")
    if not hf_token.strip():
        raise ValueError("No token provided")

print("Validating token...")
try:
    # Test token validity
    api = HfApi()
    user_info = api.whoami(token=hf_token)
    print(f"Token is valid! Logged in as: {user_info['name']}")
    login(token=hf_token, add_to_git_credential=True)
except Exception as e:
    print(f"Token validation failed: {e}")
    print("Your token appears to be invalid or expired.")
    print("Please:")
    print("1. Check your token at https://huggingface.co/settings/tokens")
    print("2. Generate a new token if needed")
    print("3. Run the script again with the correct token")
    raise ValueError("Invalid HF_TOKEN - please check your token")

# base model/tokenizer
model_name = "microsoft/phi-3-mini-4k-instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    token=hf_token
)
# Add padding token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    token=hf_token,
    attn_implementation="eager"
)

# LoRA
print("Preparing model for LoRA...")
base_model = prepare_model_for_kbit_training(base_model)
lora_config = LoraConfig(
    r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.1, bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(base_model, lora_config)

# load and preprocess dataset
print("Loading dataset...")
dataset = load_dataset("csv", data_files={"train": "phi3_lora_train.csv"})
def preprocess(batch):
    text = f"{batch['instruction']}\n{batch['input']}\nAnswer:"
    labels = batch["output"]
    inputs = tokenizer(text, truncation=True, max_length=128, padding="max_length")
    label_ids = tokenizer(labels, truncation=True, max_length=128, padding="max_length")["input_ids"]
    inputs["labels"] = label_ids
    return inputs

print("Tokenizing dataset...")
tokenized_dataset = dataset["train"].map(preprocess, batched=False)

# Training Arguments
train_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    learning_rate=2e-4,
    fp16=True,
    output_dir="./phi3_lora_finetune",
    logging_steps=10,
    save_steps=100,
    eval_strategy="no",
    warmup_steps=10,
)

print("Starting training...")
trainer = Trainer(model=model, args=train_args, train_dataset=tokenized_dataset, tokenizer=tokenizer)
trainer.train()

print("Saving model...")
model.save_pretrained("./phi3_lora_finetuned")
tokenizer.save_pretrained("./phi3_lora_finetuned")
print("Training completed successfully!")
