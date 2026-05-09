import pandas as pd

df8 = pd.read_csv("evaluated_results_200_8b.csv")
df70 = pd.read_csv("evaluated_results_200_70b.csv")

def error_analysis(df, model_name):
    # Assuming you might have error patterns; for now using basic stats
    analysis = df.groupby('technique').agg(
        total_samples=('quality_score', 'count'),
        quality_mean=('quality_score', 'mean'),
        quality_std=('quality_score', 'std'),
    ).round(3)
    
    # If you have error flags in future, we can expand this
    analysis['model'] = model_name
    return analysis.reset_index()

err8 = error_analysis(df8, "Llama-3.1-8B")
err70 = error_analysis(df70, "Llama-3.3-70B")

error_table = pd.concat([err8, err70], ignore_index=True)

print("\n=== ERROR ANALYSIS / RELIABILITY TABLE ===")
print(error_table)

error_table.to_csv("table_error_analysis_new.csv", index=False)
error_table.to_latex("table_error_analysis.tex", index=False, float_format="%.3f")

print("\n✅ table_error_analysis.tex generated!")