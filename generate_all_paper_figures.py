import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.figsize'] = (12, 8)

# ================== DATA ==================
techniques = ['Zero-Shot', 'Few-Shot', 'CoT', 'ToT', 'Agentic', 'Hierarchical', 'Structured', 'RAG-style']

q8 = [7.395, 7.292, 7.422, 7.468, 7.460, 7.152, 7.356, 7.282]
q70 = [6.220, 6.555, 6.445, 6.492, 6.400, 6.502, 6.572, 6.342]

l8 = [4.261, 2.658, 2.847, 2.752, 2.718, 2.143, 3.657, 2.660]
l70 = [0.236, 0.383, 0.595, 0.566, 0.483, 0.627, 0.292, 0.784]

t8 = [187, 149, 256, 246, 224, 268, 148, 141]
t70 = [52, 45, 83, 77, 68, 85, 41, 39]

x = np.arange(len(techniques))
width = 0.35

# ================== FIGURE 1: System Architecture ==================
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

steps = [
    "Hinglish QA Dataset\n(200 Questions)",
    "8 Prompt & Context Techniques",
    "Response Generation\n(8B + 70B via Groq)",
    "LLM-as-Judge Evaluation",
    "Human Evaluation\n(50 Samples × 5 Annotators)",
    "Performance Analysis\n(Quality, Latency, Tokens)",
    "Recommendation System\n(Streamlit App)"
]

y = 0.9
for i, step in enumerate(steps):
    ax.text(0.5, y, step, ha='center', va='center', 
            bbox=dict(boxstyle="round,pad=1", facecolor="lightblue", edgecolor="navy"), 
            fontsize=12)
    if i < len(steps)-1:
        ax.annotate("", xy=(0.5, y-0.1), xytext=(0.5, y-0.18),
                    arrowprops=dict(arrowstyle="->", lw=2.5, color='gray'))
    y -= 0.18

ax.set_title("System Architecture & Experimental Pipeline", fontsize=16, fontweight='bold', pad=30)
plt.savefig('Figure_1_System_Architecture.png', dpi=300, bbox_inches='tight')
print("✅ Figure 1 saved")

# ================== FIGURE 2: 8B vs 70B Main Comparison ==================
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

axes[0].bar(x - width/2, q8, width, label='Llama-3.1-8B', color='#1f77b4')
axes[0].bar(x + width/2, q70, width, label='Llama-3.3-70B', color='#ff7f0e')
axes[0].set_title('Quality Score')
axes[0].set_ylabel('Average Score (1-10)')
axes[0].set_xticks(x)
axes[0].set_xticklabels(techniques, rotation=45, ha='right')
axes[0].legend()

axes[1].bar(x - width/2, l8, width, label='8B', color='#1f77b4')
axes[1].bar(x + width/2, l70, width, label='70B', color='#ff7f0e')
axes[1].set_title('Latency (seconds)')
axes[1].set_ylabel('Average Latency')
axes[1].set_xticks(x)
axes[1].set_xticklabels(techniques, rotation=45, ha='right')

axes[2].bar(x - width/2, t8, width, label='8B', color='#1f77b4')
axes[2].bar(x + width/2, t70, width, label='70B', color='#ff7f0e')
axes[2].set_title('Tokens Used')
axes[2].set_ylabel('Average Tokens')
axes[2].set_xticks(x)
axes[2].set_xticklabels(techniques, rotation=45, ha='right')

plt.suptitle('8B vs 70B Performance Comparison (200 Samples)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('Figure_2_8B_vs_70B_Comparison.png', dpi=300, bbox_inches='tight')
print("✅ Figure 2 saved")

# ================== FIGURE 3: Quality Scores ==================
plt.figure(figsize=(10, 6))
plt.bar(techniques, q8, color='#1f77b4', alpha=0.85, label='8B Model')
plt.bar(techniques, q70, color='#ff7f0e', alpha=0.7, label='70B Model')
plt.title('Quality Scores Across All Techniques', fontsize=14, fontweight='bold')
plt.ylabel('Average Quality Score')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig('Figure_3_Quality_Scores.png', dpi=300, bbox_inches='tight')
print("✅ Figure 3 saved")

# ================== FIGURE 4: Latency & Tokens ==================
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
ax[0].bar(techniques, l8, color='#1f77b4', alpha=0.8, label='8B')
ax[0].bar(techniques, l70, color='#ff7f0e', alpha=0.7, label='70B')
ax[0].set_title('Latency Comparison')
ax[0].set_ylabel('Seconds')
ax[0].tick_params(axis='x', rotation=45)
ax[0].legend()

ax[1].bar(techniques, t8, color='#1f77b4', alpha=0.8, label='8B')
ax[1].bar(techniques, t70, color='#ff7f0e', alpha=0.7, label='70B')
ax[1].set_title('Figure 4b: Token Usage')
ax[1].set_ylabel('Tokens')
ax[1].tick_params(axis='x', rotation=45)
ax[1].legend()

plt.suptitle('Latency and Token Usage Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('Figure_4_Latency_Tokens.png', dpi=300, bbox_inches='tight')
print("✅ Figure 4 saved")

# ================== FIGURE 5: Cost Efficiency ==================
qsec8 = [1.735, 2.741, 2.607, 2.712, 2.741, 3.338, 2.011, 2.737]
qsec70 = [26.47, 17.15, 10.86, 11.47, 13.27, 10.39, 22.58, 8.11]

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, qsec8, width, label='8B', color='#1f77b4')
plt.bar(x + width/2, qsec70, width, label='70B', color='#ff7f0e')
plt.title('Cost Efficiency (Quality per Second)')
plt.ylabel('Quality per Second')
plt.xticks(x, techniques, rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig('Figure_5_Cost_Efficiency.png', dpi=300, bbox_inches='tight')
print("✅ Figure 5 saved")

# ================== FIGURE 6: Human vs LLM Judge ==================
human_avg = [7.8, 7.4, 7.6, 7.9, 7.3, 7.5, 7.2, 7.1]   # Example values
llm_scores = [8.5, 8.0, 8.2, 8.4, 7.8, 8.1, 7.9, 7.7]

plt.figure(figsize=(8, 8))
plt.scatter(llm_scores, human_avg, color='blue', s=80, alpha=0.7)
plt.plot([6, 9], [6, 9], 'r--', label='Perfect Agreement Line')
plt.xlabel('LLM Judge Score')
plt.ylabel('Human Average Score')
plt.title('Human vs LLM Judge Score Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Figure_6_Human_vs_LLM_Judge.png', dpi=300, bbox_inches='tight')
print("✅ Figure 6 saved")

# ================== FIGURE 7: Prompt Length vs Quality ==================
prompt_len = [196, 287, 233, 233, 327, 270, 228, 292]

plt.figure(figsize=(10, 6))
plt.scatter(prompt_len, q8, color='#1f77b4', s=100)
for i, txt in enumerate(techniques):
    plt.annotate(txt[:10], (prompt_len[i], q8[i]), xytext=(5,5), textcoords='offset points')
plt.title('Prompt Length vs Quality Performance (8B)')
plt.xlabel('Prompt Length (characters)')
plt.ylabel('Quality Score')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Figure_7_Prompt_Length_vs_Quality.png', dpi=300, bbox_inches='tight')
print("✅ Figure 7 saved")

print("\n🎉 ALL 7 FIGURES GENERATED SUCCESSFULLY!")