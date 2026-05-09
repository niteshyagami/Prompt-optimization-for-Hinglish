import pandas as pd
import json

# Robust JSONL loader
def load_jsonl(file_path):
    data = []
    skipped = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or line == 'null':
                skipped += 1
                continue
            try:
                obj = json.loads(line)
                data.append(obj)
            except Exception as e:
                skipped += 1
                if skipped < 10:  # Show only first few errors
                    print(f"⚠️ Skipped bad line {i+1}")
    
    df = pd.DataFrame(data)
    print(f"✅ Loaded {len(df)} rows | Skipped {skipped} bad lines")
    return df

# ====================== CHANGE FILE NAME HERE ======================
df = load_jsonl('merged_qa.jsonl')

# Basic Info
print("\nColumns:", df.columns.tolist())
print("\nFirst Row Example:")
print(df.iloc[0].to_dict())

# Save clean CSV
df.to_csv('student_qa_clean.csv', index=False)
print("\n✅ Saved clean dataset as 'student_qa_clean.csv'")