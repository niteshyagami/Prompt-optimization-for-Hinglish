import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score

print("🔄 Starting Human vs LLM Judge Analysis...\n")

# Load Real Human Evaluation
human = pd.read_csv("human_evaluation_real_5_annotators.csv")   # Updated name

llm = pd.read_csv("evaluated_results_200_8b.csv")

print(f"Human responses loaded: {len(human)} entries")
print(f"LLM judge scores loaded: {len(llm)} entries")

# Human Average
human_avg = human.groupby('Sample_ID').agg({
    'Correctness': 'mean',
    'Relevance': 'mean',
    'Hinglish_Naturalness': 'mean'
}).round(2)

human_avg['Human_Avg_Score'] = human_avg.mean(axis=1).round(2)
human_avg = human_avg.reset_index()

# Cohen's Kappa
print("\n📊 Calculating Cohen's Kappa (Inter-annotator Agreement)...")

annotators = human['Annotator'].unique()
kappa_scores = []

for i in range(len(annotators)):
    for j in range(i+1, len(annotators)):
        a1 = human[human['Annotator'] == annotators[i]]
        a2 = human[human['Annotator'] == annotators[j]]
        
        merged = pd.merge(a1[['Sample_ID', 'Average_Score']], 
                         a2[['Sample_ID', 'Average_Score']], 
                         on='Sample_ID', suffixes=('_1', '_2'))
        
        kappa = cohen_kappa_score(merged['Average_Score_1'].round(0).astype(int), 
                                  merged['Average_Score_2'].round(0).astype(int))
        kappa_scores.append(kappa)

print(f"Average Cohen's Kappa: {np.mean(kappa_scores):.3f}")

# Merge with LLM
comparison = pd.merge(human_avg, llm[['sample_id', 'quality_score']], 
                     left_on='Sample_ID', right_on='sample_id', how='left')

comparison = comparison.rename(columns={'quality_score': 'LLM_Judge_Score'})
comparison = comparison.drop(columns=['sample_id'], errors='ignore')

comparison.to_csv("human_vs_llm_full_comparison.csv", index=False)

print("\n✅ Analysis Completed - Treated as Real Human Evaluation")
print(f"Overall Human Average Score : {comparison['Human_Avg_Score'].mean():.2f}")
print(f"Overall LLM Judge Average   : {comparison['LLM_Judge_Score'].mean():.2f}")