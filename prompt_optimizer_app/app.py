import os
import pandas as pd
import streamlit as st

try:
    from groq import Groq
except Exception:
    Groq = None

from prompt_templates import prompt_templates


APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(APP_DIR)
RESULTS_SUMMARY = os.path.join(PROJECT_DIR, "results_summary.csv")
EVALUATED_RESULTS = os.path.join(PROJECT_DIR, "evaluated_results.csv")


def _minmax(series):
    min_v = series.min()
    max_v = series.max()
    if max_v == min_v:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - min_v) / (max_v - min_v)


def load_summary():
    if os.path.exists(RESULTS_SUMMARY):
        df = pd.read_csv(RESULTS_SUMMARY)
    elif os.path.exists(EVALUATED_RESULTS):
        raw = pd.read_csv(EVALUATED_RESULTS)
        df = raw.groupby("technique").agg({
            "quality_score": ["mean", "std"],
            "latency_sec": ["mean", "std"],
            "tokens": ["mean", "std"],
        }).round(3)
        df.columns = [f"{a}_{b}" for a, b in df.columns]
        df = df.reset_index()
    else:
        return None

    needed = {"technique", "quality_score_mean", "latency_sec_mean", "tokens_mean"}
    if not needed.issubset(set(df.columns)):
        return None
    return df


def choose_technique(summary, objective):
    summary = summary.copy()
    quality_norm = _minmax(summary["quality_score_mean"])
    latency_norm = _minmax(summary["latency_sec_mean"])
    tokens_norm = _minmax(summary["tokens_mean"])

    summary["balanced_score"] = (
        0.6 * quality_norm + 0.2 * (1 - latency_norm) + 0.2 * (1 - tokens_norm)
    ).round(4)

    if objective == "Highest quality":
        row = summary.loc[summary["quality_score_mean"].idxmax()]
    elif objective == "Lowest latency":
        row = summary.loc[summary["latency_sec_mean"].idxmin()]
    elif objective == "Lowest tokens":
        row = summary.loc[summary["tokens_mean"].idxmin()]
    else:
        row = summary.loc[summary["balanced_score"].idxmax()]

    return row, summary.sort_values(by="balanced_score", ascending=False)


def render_prompt(template, user_input, language_params):
    try:
        return template.format(question=user_input, **language_params)
    except KeyError:
        return template


def _insert_guidelines(prompt, guidelines):
    markers = [
        "Provide answer in this JSON format:",
        "Final Answer:",
        "Best Answer:",
        "Answer:",
    ]
    for marker in markers:
        idx = prompt.find(marker) if marker == "Provide answer in this JSON format:" else prompt.rfind(marker)
        if idx != -1:
            return prompt[:idx].rstrip() + "\n\n" + guidelines + "\n" + prompt[idx:]
    return prompt.rstrip() + "\n\n" + guidelines


def offline_enhance(prompt, language_label):
    guidelines = (
        "Guidelines:\n"
        f"- Use {language_label}.\n"
        "- Be concise (2-5 sentences).\n"
        "- Keep explanations student-friendly and factual.\n"
        "- If steps are asked, use short bullet points.\n"
        "- If the question is ambiguous, state one brief assumption.\n"
    )
    if "JSON format" in prompt:
        guidelines += "- Return valid JSON only in the specified format.\n"
    return _insert_guidelines(prompt, guidelines)


def refine_with_groq(raw_prompt, api_key, model, language_label):
    client = Groq(api_key=api_key)
    system = (
        "You optimize prompts for student QA. Keep the user question intact, "
        f"use {language_label}, and target 2-5 sentence answers."
    )
    user = (
        "Improve this prompt for clarity and effectiveness. Return only the final prompt.\n\n"
        f"PROMPT:\n{raw_prompt}"
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
        max_tokens=400,
    )
    return completion.choices[0].message.content.strip()


st.set_page_config(page_title="Hinglish Prompt Optimizer", layout="wide")

st.title("Hinglish Prompt Optimizer")
st.write(
    "Enter a question or rough prompt. The app selects the best-performing prompt template "
    "from your experiments and generates an optimized prompt."
)

summary = load_summary()
if summary is None:
    st.error("Missing results_summary.csv or evaluated_results.csv. Run evaluation first.")
    st.stop()

language_options = {
    "Hinglish (Roman mix)": {
        "language_label": "Hinglish (Roman script, natural mix of Hindi and English)",
        "use_examples": True,
        "example1_q": "Bhaiya, gravity kya hota hai?",
        "example1_a": "Gravity wo force hai jo earth har cheez ko apni taraf khinchta hai.",
        "example2_q": "Photosynthesis kaise hota hai?",
        "example2_a": "Plants sunlight use karke apna food banate hain, is process ko photosynthesis bolte hain.",
    },
    "English": {
        "language_label": "English",
        "use_examples": True,
        "example1_q": "What is gravity?",
        "example1_a": "Gravity is the force that pulls objects toward the Earth.",
        "example2_q": "How does photosynthesis work?",
        "example2_a": "Plants use sunlight to make food; this process is called photosynthesis.",
    },
    "Hindi (Devanagari)": {
        "language_label": "Hindi (Devanagari script)",
        "use_examples": False,
        "example1_q": "",
        "example1_a": "",
        "example2_q": "",
        "example2_a": "",
    },
}

language_choice = st.selectbox(
    "Output language",
    list(language_options.keys()),
)

objective = st.selectbox(
    "Optimization objective",
    ["Balanced (quality/cost)", "Highest quality", "Lowest latency", "Lowest tokens"],
)

user_input = st.text_area(
    "Your question or prompt",
    height=120,
    placeholder="e.g., Photosynthesis kaise hota hai?",
)

use_groq = st.checkbox("Refine with Groq (optional)", value=False)
model = st.selectbox(
    "Groq model",
    ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
    index=0,
    disabled=not use_groq,
)

if st.button("Generate optimized prompt", type="primary"):
    if not user_input.strip():
        st.warning("Please enter a question or prompt.")
        st.stop()

    chosen, ranked = choose_technique(summary, objective)
    technique = chosen["technique"]
    template = prompt_templates.get(technique)
    if not template:
        st.error(f"No template found for technique: {technique}")
        st.stop()

    language_params = language_options[language_choice]
    if technique == "few_shot" and not language_params["use_examples"]:
        template = "Answer the following question in {language_label}:\n\nQuestion: {question}\n\nAnswer:"

    optimized = render_prompt(template, user_input, language_params)

    if use_groq:
        if Groq is None:
            st.error("Groq SDK not installed. Install requirements.txt.")
            st.stop()
        api_key = ""
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            api_key = ""
        if not api_key:
            api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            st.error("GROQ_API_KEY is not set in secrets or environment.")
            st.stop()
        optimized = refine_with_groq(optimized, api_key, model, language_params["language_label"])
    else:
        optimized = offline_enhance(optimized, language_params["language_label"])

    st.subheader("Optimized prompt")
    st.code(optimized, language="text")

    st.subheader("Selected technique and expected metrics")
    metric_cols = [
        "quality_score_mean",
        "latency_sec_mean",
        "tokens_mean",
        "balanced_score",
    ]
    display_row = chosen[["technique"] + metric_cols].to_frame().T
    st.dataframe(display_row, use_container_width=True)

    st.subheader("Top techniques (balanced score)")
    st.dataframe(
        ranked[["technique", "quality_score_mean", "latency_sec_mean", "tokens_mean", "balanced_score"]]
        .head(5),
        use_container_width=True,
    )

st.caption("Tip: Change the objective to compare how the selected prompt template changes.")
