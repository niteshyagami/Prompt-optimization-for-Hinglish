import pandas as pd
import numpy as np

df8 = pd.read_csv("evaluated_results_200_8b.csv")
df70 = pd.read_csv("evaluated_results_200_70b.csv")

def summarize(df, model):
    df = df.copy()
    # Safe division
    df['latency_sec'] = df['latency_sec'].replace(0, 0.001)
    df['tokens'] = df['tokens'].replace(0, 1)
    
    summary = df.groupby('technique').agg(
        Quality=('quality_score', 'mean'),
        Q_Std=('quality_score', 'std'),
        Latency=('latency_sec', 'mean'),
        Tokens=('tokens', 'mean'),
        Q_per_Sec=('quality_score', lambda x: (x / df['latency_sec']).mean()),
        Q_per_Token=('quality_score', lambda x: (x / df['tokens']).mean())
    ).round(3)
    
    summary['Model'] = model
    return summary.reset_index()

main8 = summarize(df8, "Llama-3.1-8B")
main70 = summarize(df70, "Llama-3.3-70B")

main_table = pd.concat([main8, main70], ignore_index=True)

cols = ['Model', 'technique', 'Quality', 'Q_Std', 'Latency', 'Tokens', 'Q_per_Sec', 'Q_per_Token']
main_table = main_table[cols]

print("\n=== FINAL MAIN RESULTS TABLE (Clean) ===")
print(main_table.round(3))

main_table.to_csv("main_results_table.csv", index=False)
main_table.to_latex("main_results_table.tex", index=False, float_format="%.3f")

print("\n✅ Clean main_results_table.tex generated successfully!")