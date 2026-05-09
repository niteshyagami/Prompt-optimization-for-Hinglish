import pandas as pd

# Load both files
old_df = pd.read_csv('hinglish_student_qa_final.csv')      # Should have 198 rows
new_df = pd.read_csv('new_hinglish_samples_200.csv')       # Should have ~200 rows

print(f"Old dataset rows: {len(old_df)}")
print(f"New dataset rows: {len(new_df)}")

# Standardize column names (important!)
# For old dataset
old_df = old_df.rename(columns={
    'hinglish_question': 'question',
    'hinglish_answer': 'answer'
})

# For new dataset (if columns are different)
if 'question' not in new_df.columns and 'hinglish_question' in new_df.columns:
    new_df = new_df.rename(columns={'hinglish_question': 'question'})
if 'answer' not in new_df.columns and 'hinglish_answer' in new_df.columns:
    new_df = new_df.rename(columns={'hinglish_answer': 'answer'})

# Keep only important columns
old_clean = old_df[['question', 'answer']].copy()
new_clean = new_df[['question', 'answer']].copy()

# Merge both
final_df = pd.concat([old_clean, new_clean], ignore_index=True)

# Shuffle the dataset (better for experiments)
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save final dataset
final_df.to_csv('final_hinglish_qa_dataset.csv', index=False)

print(f"\n🎉 Final Dataset Created!")
print(f"Total samples: {len(final_df)}")
print("\nColumns:", final_df.columns.tolist())
print("\nFirst 3 samples:")
print(final_df.head(3))