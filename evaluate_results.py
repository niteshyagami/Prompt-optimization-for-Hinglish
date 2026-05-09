import argparse
import pandas as pd
from groq import Groq
from tqdm import tqdm
import time

client = Groq(api_key="YOUR_API_KEY_HERE")   # ← Your key

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="medium_test_results.csv")
parser.add_argument("--output", type=str, default="evaluated_results.csv")
args = parser.parse_args()

# Load results
RESULTS_FILE = args.input
DATASET_FILE = 'final_hinglish_qa_dataset.csv'

df = pd.read_csv(RESULTS_FILE)

if 'question' not in df.columns:
    if 'sample_id' not in df.columns:
        raise ValueError("Missing 'question' and 'sample_id' columns in results CSV.")

    base_df = pd.read_csv(DATASET_FILE)
    if 'question' not in base_df.columns:
        raise ValueError("Dataset file missing required 'question' column.")

    sample_ids = pd.to_numeric(df['sample_id'], errors='coerce')
    if sample_ids.isna().any():
        bad_ids = df.loc[sample_ids.isna(), 'sample_id'].head(5).tolist()
        raise ValueError(f"Non-numeric sample_id values found: {bad_ids}")
    if (sample_ids < 0).any():
        raise ValueError("sample_id contains negative values.")

    sample_ids = sample_ids.astype(int)
    if sample_ids.max() >= len(base_df):
        raise ValueError("sample_id contains values outside the dataset range.")

    questions = base_df['question'].reset_index(drop=True)
    df['question'] = questions.iloc[sample_ids.tolist()].values

print(f"Evaluating {len(df)} results...")

def evaluate_answer(question, generated, ground_truth):
    prompt = f"""You are an expert Indian teacher. Score the answer from 1 to 10.

Question: {question}
Ground Truth: {ground_truth}
Generated Answer: {generated}

Score only on:
- Correctness
- Relevance
- Clarity (Hinglish naturalness)

Return only JSON:
{{"score": 8.5, "reason": "short reason"}}

"""
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=300
        )
        response = completion.choices[0].message.content.strip()
        
        # Simple extraction
        import json
        start = response.find('{')
        end = response.rfind('}') + 1
        data = json.loads(response[start:end])
        return float(data.get('score', 5.0))
    
    except:
        return 5.0

# Add scores
scores = []
for i, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating Quality"):
    score = evaluate_answer(row['question'], row['generated_answer'], row['ground_truth'])
    scores.append(score)
    time.sleep(1.0)

df['quality_score'] = scores

# Save
df.to_csv(args.output, index=False)

# Final Summary
group_cols = ["technique"]
if "model" in df.columns:
    group_cols = ["model", "technique"]

summary = df.groupby(group_cols).agg({
    'latency_sec': 'mean',
    'tokens': 'mean',
    'quality_score': 'mean'
}).round(3)

print("\nEVALUATION COMPLETED!")
print("\n=== FINAL SUMMARY TABLE ===")
print(summary)
