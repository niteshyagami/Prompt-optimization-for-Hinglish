prompt_templates = {
    "zero_shot": """Answer the following question in {language_label}:

Question: {question}

Answer:""",

    "few_shot": """Answer the following question in {language_label}. Here are some examples:

Example 1:
Question: {example1_q}
Answer: {example1_a}

Example 2:
Question: {example2_q}
Answer: {example2_a}

Now answer this question:
Question: {question}

Answer:""",

    "chain_of_thought": """Answer the following question step by step in {language_label}.

Question: {question}

Let's think step by step:
1.
2.
3.

Final Answer:""",

    "structured_context": """You are a helpful Indian teacher. Answer in clear {language_label}.

{question}

Provide answer in this JSON format:
{
  "answer": "your answer here",
  "explanation": "short explanation"
}""",

    "hierarchical_context": """Context: You are teaching Class 9-11 Indian students.

Main Question: {question}

Answer in natural {language_label} with clear explanation.""",

    "rag_style": """Use the following knowledge to answer the question in {language_label}.

Knowledge: {question}  # (We'll use question itself as knowledge for simplicity)

Answer:""",

    "tree_of_thought": """Think about the question in multiple ways and then give the best answer in {language_label}.

Question: {question}

Explore different angles:
- Angle 1:
- Angle 2:
- Best Answer:""",

    "agentic_context": """First understand the question, then think, then answer in natural {language_label}.

Question: {question}

Step 1: Understand question
Step 2: Recall knowledge
Step 3: Give accurate answer

Answer:""",
}
