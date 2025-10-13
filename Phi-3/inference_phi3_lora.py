from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers.testing_utils import device_name

model_name = "./phi3_lora_finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
prompt = "The following URL was flagged as phishing: http://secure-bank-login.com. Explain why.\nExplanation:"
response = pipe(prompt, max_new_tokens=128,)[0]['generated_text']
print(response)