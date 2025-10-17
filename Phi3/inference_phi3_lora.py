import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import json
import re
import warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='transformers')
warnings.filterwarnings('ignore', message='.*flash-attention.*')

class Phi3LoRAInference:
    def __init__(self, base_model_name="microsoft/Phi3-mini-4K-instruct",
                 adapter_path="./phi3_lora_finetuned", device="auto"):
        self.device = device
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path

        # Configure quantization for memory efficiency
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = None
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the base model and LoRA adapter"""
        print("Loading Phi3 model with LoRA adapter...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model with quantization - fixed deprecation warnings
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=self.quantization_config,
            device_map=self.device,
            trust_remote_code=True,
            dtype=torch.float16,  # Fixed: changed from torch_dtype to dtype
            low_cpu_mem_usage=True,
            attn_implementation="eager",  # Fixed: explicitly set to eager to avoid flash-attention issues
        )

        # Load LoRA adapter
        try:
            self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
        except Exception as e:
            print(f"Warning: Could not load LoRA adapter from {self.adapter_path}: {str(e)}")
            print("Using base model without LoRA fine-tuning")
            self.model = base_model

        self.model.eval()

        print("Model loaded successfully!")

    def predict(self, url, instruction="Decide if this is phishing and explain why."):
        """Make prediction for a single URL"""
        # Prepare input text
        input_text = f"<|user|>\n{instruction}\nURL: {url}<|end|>\n<|assistant|>\n"

        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )

        # Move to device
        if hasattr(self.model, 'device'):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode response
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the assistant's response
        assistant_response = generated_text.split("<|assistant|>\n")[-1]

        return assistant_response.strip()

    def extract_classification(self, response):
        """Extract binary classification from model response"""
        response_lower = response.lower()

        # Look for explicit classification
        if "phishing:" in response_lower or "phishing" in response_lower:
            if "legitimate:" not in response_lower[:20]:  # Check if phishing comes first
                return 1, response  # Phishing

        if "legitimate:" in response_lower or "legitimate" in response_lower:
            if "phishing:" not in response_lower[:20]:  # Check if legitimate comes first
                return 0, response  # Legitimate

        # Fallback: count occurrences
        phishing_count = response_lower.count("phishing")
        legitimate_count = response_lower.count("legitimate")

        if phishing_count > legitimate_count:
            return 1, response
        else:
            return 0, response

    def predict_batch(self, urls, instruction="Decide if this is phishing and explain why."):
        """Make predictions for multiple URLs"""
        results = []
        for url in urls:
            try:
                response = self.predict(url, instruction)
                classification, explanation = self.extract_classification(response)
                results.append({
                    'url': url,
                    'classification': classification,
                    'explanation': explanation,
                    'confidence': 1.0  # Placeholder for confidence score
                })
            except Exception as e:
                print(f"Error processing URL {url}: {str(e)}")
                results.append({
                    'url': url,
                    'classification': 0,
                    'explanation': f"Error: {str(e)}",
                    'confidence': 0.0
                })

        return results
