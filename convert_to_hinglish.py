import pandas as pd
from groq import Groq
import json
from time import sleep

# ================== YOUR GROQ API KEY ==================
client = Groq(api_key="YOUR_API_KEY_HERE")   # ← Replace with your key

# Load the clean CSV
df = pd.read_csv('student_qa_clean.csv')
print(f"Original rows: {len(df)}")

# Use only 250 samples for speed (you can increase later)
df = df.head(250).copy()

def convert_to_hinglish(instruction, input_text, output):
    # Combine instruction + input to make full question
    full_question = str(instruction)
    if pd.notna(input_text) and str(input_text).strip() != "":
        full_question += " " + str(input_text)
    
    prompt = f"""Convert the following English educational question and answer into natural Hinglish (Roman script, like Indian students speak):

Question: {full_question}
Answer: {output}

Return ONLY valid JSON:
{{
  "hinglish_question": "natural hinglish question here",
  "hinglish_answer": "natural hinglish answer here"
}}
"""

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1200
        )
        response = completion.choices[0].message.content.strip()
        
        # Extract JSON
        start = response.find('{')
        end = response.rfind('}') + 1
        json_str = response[start:end]
        return json.loads(json_str)
    
    except Exception as e:
        print("Error:", e)
        return {"hinglish_question": full_question, "hinglish_answer": output}

# ============== Start Conversion ==============
results = []

for i, row in df.iterrows():
    print(f"Processing {i+1}/{len(df)} ...")
    
    converted = convert_to_hinglish(
        row['instruction'], 
        row.get('input', ''), 
        row['output']
    )
    
    results.append({
        'original_instruction': row['instruction'],
        'original_input': row.get('input', ''),
        'original_output': row['output'],
        'hinglish_question': converted.get('hinglish_question'),
        'hinglish_answer': converted.get('hinglish_answer'),
    })
    
    sleep(1.8)  # Safe rate limit

# Save Final Dataset
final_df = pd.DataFrame(results)
final_df.to_csv('hinglish_student_qa_final.csv', index=False)

print("\n🎉 DONE!")
print(f"Final dataset saved with {len(final_df)} samples")
print(final_df.head())