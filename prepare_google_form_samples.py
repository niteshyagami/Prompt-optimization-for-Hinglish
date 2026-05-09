import pandas as pd
import numpy as np

# Load the evaluated files
df8 = pd.read_csv("evaluated_results_200_8b.csv")
df70 = pd.read_csv("evaluated_results_200_70b.csv")

# Combine both models for better diversity
df = pd.concat([df8, df70], ignore_index=True)

# Optional: Stratified sampling (recommended) - balanced across techniques and models
np.random.seed(42)  # for reproducibility

# Sample 50 examples - balanced across techniques
sampled = df.groupby('technique', group_keys=False).apply(
    lambda x: x.sample(n=min(7, len(x)), random_state=42)  # ~6-7 per technique
).sample(50, random_state=42).reset_index(drop=True)

# Shuffle the final 50
sampled = sampled.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"✅ Successfully sampled {len(sampled)} diverse examples")
print(sampled['technique'].value_counts())

# Create a clean output for Google Form
output = []
for i, row in sampled.iterrows():
    output.append(f"--- Sample {i+1} ---")
    output.append(f"Question: {row['question']}")
    output.append(f"Model: {row['model']}")
    output.append(f"Technique: {row['technique']}")
    output.append(f"Answer:\n{row['generated_answer']}")
    output.append("\n" + "="*80 + "\n")

# Save to text file (easy to copy-paste into Google Form)
with open("google_form_50_samples.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output))

print("\n✅ File saved as 'google_form_50_samples.txt'")
print("You can now open this file and copy-paste into your Google Form.")