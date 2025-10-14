import torch
import os
import getpass
from huggingface_hub import login

print("=== Phi-3 Fine-tuning Debug Test ===")

# Test 1: Authentication
print("Step 1: Testing Hugging Face authentication...")
try:
    hf_token = getpass.getpass("HF Token: ")
    if not hf_token:
        raise ValueError("Token required")
    login(token=hf_token)
    print("✓ Authentication successful!")
except Exception as e:
    print(f"✗ Authentication failed: {e}")
    exit(1)

# Test 2: Check CUDA availability
print("Step 2: Checking CUDA availability...")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")
    print(f"Memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print(f"Memory reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")

# Test 3: Try loading tokenizer only
print("Step 3: Testing tokenizer loading...")
try:
    from transformers import AutoTokenizer
    model_name = "microsoft/Phi-3-mini-4K-instruct"
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print("✓ Tokenizer loaded successfully!")
except Exception as e:
    print(f"✗ Tokenizer loading failed: {e}")
    exit(1)

print("\nAll basic tests passed! The issue is likely with model loading.")
print("Proceeding to test model loading with different configurations...")

# Test 4: Try loading model with different configurations
print("Step 4: Testing model loading configurations...")

try:
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    # First try without quantization
    print("Trying to load model without quantization...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cpu",  # Load to CPU first
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print("✓ Model loaded on CPU successfully!")
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"✗ CPU loading failed: {e}")

    # Now try with quantization
    print("Trying to load model with 4-bit quantization...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    print("✓ Model loaded with quantization successfully!")

except Exception as e:
    print(f"✗ Model loading failed: {e}")
    import traceback
    traceback.print_exc()

print("\nDiagnostic test completed!")
