# ================== PROMPT TEMPLATES ==================

prompt_templates = {
    "zero_shot": """Answer the following question in Hinglish:

Question: {question}

Answer:""",

    "few_shot": """Answer the following question in Hinglish. Here are some examples:

Example 1:
Question: Bhaiya, gravity kya hota hai?
Answer: Gravity wo force hai jo earth har cheez ko apni taraf khinchta hai...

Example 2:
Question: Photosynthesis kaise hota hai?
Answer: Plants sunlight use karke apna food banate hain...

Now answer this question:
Question: {question}

Answer:""",

    "chain_of_thought": """Answer the following question step by step in Hinglish.

Question: {question}

Let's think step by step:
1. 
2. 
3. 

Final Answer:""",

    "structured_context": """You are a helpful Indian teacher. Answer in clear Hinglish.

{question}

Provide answer in this JSON format:
{{
  "answer": "your answer here",
  "explanation": "short explanation"
}}""",

    "hierarchical_context": """Context: You are teaching Class 9-11 Indian students.

Main Question: {question}

Answer in natural Hinglish with clear explanation.""",

    "rag_style": """Use the following knowledge to answer the question in Hinglish.

Knowledge: {question}  # (We'll use question itself as knowledge for simplicity)

Answer:""",

    "tree_of_thought": """Think about the question in multiple ways and then give the best answer in Hinglish.

Question: {question}

Explore different angles:
- Angle 1:
- Angle 2:
- Best Answer:""",

    "agentic_context": """First understand the question, then think, then answer in natural Hinglish.

Question: {question}

Step 1: Understand question
Step 2: Recall knowledge
Step 3: Give accurate answer

Answer:"""
}

if __name__ == "__main__":
    print("All 8 Prompt Templates Ready!")
