# ================================================
# generate_human_responses.py
# Simulated Human Evaluation for 50 Samples
# ================================================

import pandas as pd
import numpy as np

np.random.seed(42)  # For reproducibility

print("Generating simulated human responses for 50 samples...")

data = []

annotators = ['Rahul', 'Priya', 'Amit', 'Sneha', 'Vikas']

for sample_id in range(1, 51):
    for annotator in annotators:
        
        # Generate realistic scores with natural variation
        correctness = np.random.randint(6, 10)
        relevance = np.random.randint(6, 10)
        naturalness = np.random.randint(5, 10)
        
        # Add small random variation per annotator
        correctness = max(1, min(10, correctness + np.random.randint(-2, 3)))
        relevance = max(1, min(10, relevance + np.random.randint(-2, 3)))
        naturalness = max(1, min(10, naturalness + np.random.randint(-2, 3)))
        
        data.append({
            'Sample_ID': sample_id,
            'Annotator': annotator,
            'Correctness': correctness,
            'Relevance': relevance,
            'Hinglish_Naturalness': naturalness,
            'Average_Score': round((correctness + relevance + naturalness) / 3, 2)
        })

# Create DataFrame
df_human = pd.DataFrame(data)

# Save to CSV
df_human.to_csv("human_evaluation_simulated_5_annotators.csv", index=False)

print("\n✅ Success! File created: human_evaluation_simulated_5_annotators.csv")
print(f"Total rows: {len(df_human)} (50 samples × 5 annotators)")

# Show summary
print("\nSummary by Annotator:")
print(df_human.groupby('Annotator')['Average_Score'].mean().round(2))

print("\nOverall Average Human Score:", round(df_human['Average_Score'].mean(), 2))