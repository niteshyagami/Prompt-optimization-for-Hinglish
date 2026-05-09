import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ========================= LOAD DATA =========================
print("Loading latest evaluated results...")
df8 = pd.read_csv("evaluated_results_200_8b.csv")
df70 = pd.read_csv("evaluated_results_200_70b.csv")

print(f"8B  shape : {df8.shape}")
print(f"70B shape : {df70.shape}")

# ========================= SUMMARY TABLE =========================
def create_summary(df, model_name):
    summary = df.groupby('technique').agg(
        avg_quality=('quality_score', 'mean'),
        std_quality=('quality_score', 'std'),
        avg_latency=('latency_sec', 'mean'),
        avg_tokens=('tokens', 'mean'),
        count=('quality_score', 'count')
    ).round(3)
    summary['model'] = model_name
    return summary.reset_index()

summary8 = create_summary(df8, "Llama-3.1-8B")
summary70 = create_summary(df70, "Llama-3.3-70B")

comparison = pd.concat([summary8, summary70], ignore_index=True)

print("\n=== 8B vs 70B Comparison Table ===")
print(comparison[['model', 'technique', 'avg_quality', 'avg_latency', 'avg_tokens']])

# Save main summary
comparison.to_csv("results_summary_new.csv", index=False)

# ========================= DIRECT 8B vs 70B COMPARISON =========================
tech_comparison = pd.merge(
    summary8[['technique', 'avg_quality', 'avg_latency', 'avg_tokens']],
    summary70[['technique', 'avg_quality', 'avg_latency', 'avg_tokens']],
    on='technique',
    suffixes=('_8B', '_70B')
)

tech_comparison['quality_diff'] = tech_comparison['avg_quality_8B'] - tech_comparison['avg_quality_70B']
tech_comparison['latency_diff'] = tech_comparison['avg_latency_8B'] - tech_comparison['avg_latency_70B']

print("\n=== Technique-wise 8B vs 70B Direct Comparison ===")
print(tech_comparison.round(3))

tech_comparison.to_csv("table_8b_vs_70b_comparison.csv", index=False)

# ========================= STATISTICAL TESTS =========================
print("\n=== Statistical Significance Tests (Quality Score) ===")

techniques = df8['technique'].unique()

stats_results = []
for tech in techniques:
    scores8 = df8[df8['technique'] == tech]['quality_score']
    scores70 = df70[df70['technique'] == tech]['quality_score']
    
    t_stat, p_value = stats.ttest_ind(scores8, scores70, equal_var=False)
    
    stats_results.append({
        'technique': tech,
        'mean_8B': round(scores8.mean(), 3),
        'mean_70B': round(scores70.mean(), 3),
        'diff': round(scores8.mean() - scores70.mean(), 3),
        'p_value': round(p_value, 4),
        'significant': p_value < 0.05
    })

stats_df = pd.DataFrame(stats_results)
print(stats_df)

stats_df.to_csv("table_statistical_significance.csv", index=False)

# ========================= LATEX TABLES =========================
def df_to_latex(df, filename, caption=""):
    latex = df.to_latex(index=False, float_format="%.3f", caption=caption)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"LaTeX table saved: {filename}")

df_to_latex(comparison, "results_summary.tex", "Overall Performance Summary")
df_to_latex(tech_comparison, "table_8b_vs_70b.tex", "8B vs 70B Direct Comparison")
df_to_latex(stats_df, "table_statistical_significance.tex", "Statistical Significance Tests")

print("\n✅ All tables regenerated successfully!")
print("Key files created:")
print("   - results_summary_new.csv")
print("   - table_8b_vs_70b_comparison.csv")
print("   - table_statistical_significance.csv")
print("   - results_summary.tex")
print("   - table_8b_vs_70b.tex")
print("   - table_statistical_significance.tex")