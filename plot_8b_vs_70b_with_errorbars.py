import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data
techniques = ['Zero-Shot', 'Few-Shot', 'CoT', 'ToT', 'Agentic', 'Hierarchical', 'Structured', 'RAG-style']

quality_8b = [7.395, 7.292, 7.422, 7.468, 7.460, 7.152, 7.356, 7.282]
quality_70b = [6.220, 6.555, 6.445, 6.492, 6.400, 6.502, 6.572, 6.342]

q_std_8b = [2.772, 2.870, 2.829, 2.733, 2.727, 2.993, 2.769, 2.940]
q_std_70b = [3.047, 2.934, 3.020, 2.958, 2.903, 2.912, 2.975, 3.092]

latency_8b = [4.261, 2.658, 2.847, 2.752, 2.718, 2.143, 3.657, 2.660]
latency_70b = [0.236, 0.383, 0.595, 0.566, 0.483, 0.627, 0.292, 0.784]

tokens_8b = [187.25, 148.81, 256.42, 245.58, 224.41, 268.06, 147.57, 141.37]
tokens_70b = [52.03, 45.27, 83.38, 76.60, 68.15, 84.77, 41.43, 38.61]

x = np.arange(len(techniques))
width = 0.35

fig, axes = plt.subplots(1, 3, figsize=(20, 7))

# Quality with Error Bars
axes[0].bar(x - width/2, quality_8b, width, label='Llama-3.1-8B', color='#1f77b4', yerr=q_std_8b, capsize=5)
axes[0].bar(x + width/2, quality_70b, width, label='Llama-3.3-70B', color='#ff7f0e', yerr=q_std_70b, capsize=5)
axes[0].set_title('Quality Score (± Std)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Average Quality (1-10)')
axes[0].set_xticks(x)
axes[0].set_xticklabels(techniques, rotation=45, ha='right')
axes[0].legend()

# Latency
axes[1].bar(x - width/2, latency_8b, width, label='8B', color='#1f77b4')
axes[1].bar(x + width/2, latency_70b, width, label='70B', color='#ff7f0e')
axes[1].set_title('Latency (seconds)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Average Latency (s)')
axes[1].set_xticks(x)
axes[1].set_xticklabels(techniques, rotation=45, ha='right')

# Tokens
axes[2].bar(x - width/2, tokens_8b, width, label='8B', color='#1f77b4')
axes[2].bar(x + width/2, tokens_70b, width, label='70B', color='#ff7f0e')
axes[2].set_title('Tokens Used', fontsize=14, fontweight='bold')
axes[2].set_ylabel('Average Tokens')
axes[2].set_xticks(x)
axes[2].set_xticklabels(techniques, rotation=45, ha='right')

plt.suptitle('8B vs 70B Comparison on Hinglish Student QA (200 Samples)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('8b_vs_70b_comparison_with_errorbars.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Chart with error bars saved as '8b_vs_70b_comparison_with_errorbars.png'")