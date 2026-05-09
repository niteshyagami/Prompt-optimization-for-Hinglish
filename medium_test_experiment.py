import argparse
import pandas as pd
from groq import Groq
import time
from tqdm import tqdm

client = Groq(api_key="YOUR_API_KEY_HERE")   # ← Your key

parser = argparse.ArgumentParser()
parser.add_argument("--samples", type=int, default=200)
parser.add_argument("--model", type=str, default="llama-3.1-8b-instant")
parser.add_argument("--output", type=str, default="medium_test_results.csv")
args = parser.parse_args()

df = pd.read_csv('final_hinglish_qa_dataset.csv')
test_df = df.head(args.samples).copy()

print(f"Running Medium Test on {len(test_df)} samples with {args.model} ...")

techniques = {
    "zero_shot": "Answer in natural Hinglish:\n\nQuestion: {question}\n\nAnswer:",
    
    "few_shot": "Answer in natural Hinglish. Example: Bhaiya gravity kya hai? → Gravity earth ki force hai jo sabko neeche khinchti hai.\n\nQuestion: {question}\nAnswer:",
    
    "chain_of_thought": "Think step by step in Hinglish and then give final answer.\n\nQuestion: {question}\n\nStep-by-step:",
    
    "structured_context": "Answer in JSON format in Hinglish.\nQuestion: {question}\n\n{{\"answer\": \"\", \"explanation\": \"\"}}",
    
    "hierarchical_context": "Context: You are teaching Class 9-11 Indian students.\n\nMain Question: {question}\n\nAnswer in natural Hinglish with clear explanation.",

    "rag_style": "Use the following knowledge to answer the question in Hinglish.\n\nKnowledge: {question}  # (We'll use question itself as knowledge for simplicity)\n\nAnswer:",

    "tree_of_thought": "Think in 2-3 different ways then give best answer in Hinglish.\n\nQuestion: {question}\n\nThinking:",

    "agentic_context": "First understand the question, then think, then answer in natural Hinglish.\n\nQuestion: {question}\n\nStep 1: Understand question\nStep 2: Recall knowledge\nStep 3: Give accurate answer\n\nAnswer:"
}

results = []

for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Medium Test"):
    question = str(row['question'])
    
    for tech_name, template in techniques.items():
        prompt = template.format(question=question)
        
        start_time = time.time()
        
        try:
            completion = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=700
            )
            generated = completion.choices[0].message.content.strip()
            latency = time.time() - start_time
            tokens = completion.usage.total_tokens if hasattr(completion, 'usage') else 0
        except:
            generated = "ERROR"
            latency = 0
            tokens = 0
        
        results.append({
            'sample_id': idx,
            'model': args.model,
            'question': question,
            'technique': tech_name,
            'generated_answer': generated[:500],
            'latency_sec': round(latency, 3),
            'tokens': tokens,
            'ground_truth': str(row['answer'])[:300]
        })
    
    time.sleep(1.2)

# Save
final_results = pd.DataFrame(results)
final_results.to_csv(args.output, index=False)

print("\nMEDIUM TEST DONE!")
print(f"Total rows: {len(final_results)}")

# Summary Table
summary = final_results.groupby('technique').agg({
    'latency_sec': ['mean', 'std'],
    'tokens': 'mean'
}).round(3)
print("\n=== SUMMARY ===")
print(summary)
