import pandas as pd
import kagglehub
import os

# Load Layer 1 dataset
dataset_path = kagglehub.dataset_download("shashwatwork/web-page-phishing-detection-dataset")

# Find and load the CSV file
csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
if not csv_files:
    raise FileNotFoundError("No CSV file found in the downloaded dataset")

# Load the first CSV file found
csv_file_path = os.path.join(dataset_path, csv_files[0])
df = pd.read_csv(csv_file_path)
print(f"Loaded dataset from: {csv_file_path}")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Check the target column values
print(f"Unique values in 'status' column: {df['status'].unique()}")
print(f"Value counts for 'status':")
print(df['status'].value_counts())

# Prepare new rows for LLM fine-tuning
rows = []
for _, row in df.iterrows():
    url = row["url"]
    status = row["status"]  # Changed from 'label' to 'status'
    instruction = "Decide if this is phishing and explain why."
    if status == "phishing":  # Assuming 'phishing' indicates phishing URLs
        output = f"Phishing: The URL '{url}' contains suspicious keywords or patterns typical for phishing attempts."
    else:
        output = f"Legitimate: The URL '{url}' does not show common phishing indicators."
    rows.append({
        "input": url,
        "instruction": instruction,
        "output": output
    })

# Save for LLM fine-tuning
out_df = pd.DataFrame(rows)
out_df.to_csv("phi3_lora_train.csv", index=False)
print("Saved LLM fine-tuning file: phi3_lora_train.csv")
