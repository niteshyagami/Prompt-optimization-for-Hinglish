import pandas as pd
from groq import Groq
import time
from tqdm import tqdm

# ================== YOUR GROQ API KEY ==================
client = Groq(api_key="YOUR_API_KEY_HERE")   # ← PUT YOUR KEY HERE

# Load dataset
df = pd.read_csv('final_hinglish_qa_dataset.csv')

# ============== FAST TEST SETTINGS ==============
TEST_SAMPLES = 20

print(f"🚀 Running FAST TEST on {TEST_SAMPLES} samples...")

test_df = df.head(TEST_SAMPLES).copy()

# Prompt Templates
techniques = {
    "zero_shot": "Answer this question in natural Hinglish:\n\nQuestion: {question}\n\nAnswer:",
    
    "few_shot": "Answer in natural Hinglish.\nExample: Bhaiya gravity kya hai? → Gravity earth ki force hai...\n\nQuestion: {question}\nAnswer:",
    
    "chain_of_thought": "Think step by step and answer in natural Hinglish.\n\nQuestion: {question}\n\nStep by step thinking:",
    
    "structured_context": "Answer in Hinglish in JSON format.\nQuestion: {question}\n\nAnswer:"
}

results = []

for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Testing"):
    question = str(row['question'])
    
    for tech_name, template in techniques.items():
        prompt = template.format(question=question)
        
        start_time = time.time()
        
        try:
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=600
            )
            generated = completion.choices[0].message.content.strip()
            latency = time.time() - start_time
            tokens = completion.usage.total_tokens if hasattr(completion, 'usage') else 0
        except Exception as e:
            generated = f"ERROR: {str(e)[:100]}"
            latency = 0
            tokens = 0
        
        results.append({
            'sample_id': idx,
            'technique': tech_name,
            'question': question[:150],
            'generated_answer': generated[:400],
            'latency_sec': round(latency, 3),
            'tokens': tokens
        })
        
    time.sleep(1.0)

# Save Results
final_results = pd.DataFrame(results)
final_results.to_csv('fast_test_results.csv', index=False)

print("\n✅ FAST TEST FINISHED SUCCESSFULLY!")
print(f"Total results: {len(final_results)} rows")
print("\nAverage Latency by Technique:")
print(final_results.groupby('technique')['latency_sec'].mean().round(3))