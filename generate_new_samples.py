import pandas as pd
from groq import Groq
import json
from time import sleep

# ================== YOUR NEW API KEY ==================
client = Groq(api_key="YOUR_NEW_API_KEY_HERE")   # ← NEW KEY HERE

def generate_qa_batch(batch_size=20):
    system_prompt = """You are an experienced Indian CBSE NCERT teacher (Class 8-12).
Create natural Hinglish student question-answer pairs.

Rules:
- Use natural Hinglish (Roman script)
- Questions should sound like real Indian students
- Topics: Science, Math, History, Geography, Physics, Chemistry, Biology, GK
- Answer should be helpful and accurate (2-5 sentences)

Output ONLY JSON array like this:
[
  {"question": "...", "answer": "...", "topic": "Science", "class_level": "10"}
]"""

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate {batch_size} fresh diverse QA pairs. Make them different from common topics."}
            ],
            temperature=0.85,
            max_tokens=4000
        )
        response = completion.choices[0].message.content
        start = response.find('[')
        end = response.rfind(']') + 1
        return json.loads(response[start:end])
    except Exception as e:
        print("Error:", e)
        return []

# Generate 200 new samples
all_data = []
for i in range(10):   # 10 batches * 20 = 200
    print(f"🔄 Generating batch {i+1}/10 ...")
    batch = generate_qa_batch(20)
    all_data.extend(batch)
    sleep(3.5)

new_df = pd.DataFrame(all_data)
new_df.to_csv('new_hinglish_samples_200.csv', index=False)

print(f"\n✅ Successfully generated {len(new_df)} new samples!")
print(new_df.head())