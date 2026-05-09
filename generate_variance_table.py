import pandas as pd

df8 = pd.read_csv("evaluated_results_200_8b.csv")
df70 = pd.read_csv("evaluated_results_200_70b.csv")

def get_variance(df, model_name):
    var = df.groupby('technique').agg(
        quality_std=('quality_score', 'std'),
        latency_std=('latency_sec', 'std'),
        tokens_std=('tokens', 'std')
    ).round(3)
    var['model'] = model_name
    return var.reset_index()

var8 = get_variance(df8, "Llama-3.1-8B")
var70 = get_variance(df70, "Llama-3.3-70B")

variance_table = pd.concat([var8, var70], ignore_index=True)

print("\n=== VARIANCE / STANDARD DEVIATION TABLE ===")
print(variance_table)

variance_table.to_csv("table_variance_new.csv", index=False)
variance_table.to_latex("table_variance.tex", index=False, float_format="%.3f")

print("\n✅ table_variance.tex and .csv generated successfully!")