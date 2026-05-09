import pandas as pd
import numpy as np

# Load data
df8 = pd.read_csv("evaluated_results_200_8b.csv")
df70 = pd.read_csv("evaluated_results_200_70b.csv")

def compute_efficiency(df, model_name):
    df = df.copy()
    
    # Prevent division by zero
    df['latency_sec'] = df['latency_sec'].replace(0, 0.001)
    df['tokens'] = df['tokens'].replace(0, 1)
    
    df['quality_per_sec'] = df['quality_score'] / df['latency_sec']
    df['quality_per_token'] = df['quality_score'] / df['tokens']
    
    eff = df.groupby('technique').agg(
        avg_quality=('quality_score', 'mean'),
        avg_latency=('latency_sec', 'mean'),
        avg_tokens=('tokens', 'mean'),
        avg_q_per_sec=('quality_per_sec', 'mean'),
        avg_q_per_token=('quality_per_token', 'mean'),
        std_quality=('quality_score', 'std')
    ).round(4)
    
    eff['model'] = model_name
    return eff.reset_index()

eff8 = compute_efficiency(df8, "Llama-3.1-8B")
eff70 = compute_efficiency(df70, "Llama-3.3-70B")

efficiency_comparison = pd.concat([eff8, eff70], ignore_index=True)

cols = ['model', 'technique', 'avg_quality', 'avg_latency', 'avg_tokens', 
        'avg_q_per_sec', 'avg_q_per_token']

print("\n=== FINAL COST EFFICIENCY TABLE (200 samples) ===")
print(efficiency_comparison[cols])

# Save files
efficiency_comparison[cols].to_csv("table_cost_efficiency_new.csv", index=False)
efficiency_comparison[cols].to_latex("table_cost_efficiency.tex", index=False, 
                                    float_format="%.4f")

print("\n✅ Clean tables saved successfully!")
print("Files: table_cost_efficiency_new.csv  +  table_cost_efficiency.tex")

# Top 3 insights
print("\n🏆 Key Insights:")
print("Best Overall Quality     :", eff8.loc[eff8['avg_quality'].idxmax()]['technique'], "(8B)")
print("Best Quality per Second  :", efficiency_comparison.loc[efficiency_comparison['avg_q_per_sec'].idxmax()]['technique'])
print("Best Quality per Token   :", efficiency_comparison.loc[efficiency_comparison['avg_q_per_token'].idxmax()]['technique'])