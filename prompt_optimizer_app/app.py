import os
import uuid
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

TECHNIQUE_LABELS = {
    "zero_shot": "Zero-shot",
    "few_shot": "Few-shot",
    "chain_of_thought": "Chain-of-Thought",
    "structured_context": "Structured Context",
    "hierarchical_context": "Hierarchical Context",
    "rag_style": "RAG-style",
    "tree_of_thought": "Tree-of-Thought",
    "agentic_context": "Agentic Context",
}


# ----------------------------------------------------------------------------
# Backend logic (unchanged from original experiment pipeline)
# ----------------------------------------------------------------------------

def _minmax(series):
    min_v = series.min()
    max_v = series.max()
    if max_v == min_v:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - min_v) / (max_v - min_v)


@st.cache_data
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


def get_llm_answer(prompt_text, api_key, model):
    """Actually answers a prompt (used for the live naive-vs-optimized comparison)."""
    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt_text}],
        temperature=0.7,
        max_tokens=400,
    )
    return completion.choices[0].message.content.strip()


def get_groq_api_key():
    manual = st.session_state.get("manual_groq_key", "")
    if manual:
        return manual
    api_key = ""
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = ""
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")
    return api_key


def estimate_cost(tokens, rate_per_million):
    return (tokens / 1_000_000) * rate_per_million


def process_question(user_input, language_choice, language_options, objective, use_groq, model, summary):
    """Runs the full pipeline for one question and returns a result dict."""
    chosen, ranked = choose_technique(summary, objective)
    technique = chosen["technique"]
    template = prompt_templates.get(technique)
    if not template:
        return {"error": f"No template found for technique: {technique}"}

    language_params = language_options[language_choice]
    if technique == "few_shot" and not language_params["use_examples"]:
        template = "Answer the following question in {language_label}:\n\nQuestion: {question}\n\nAnswer:"

    optimized = render_prompt(template, user_input, language_params)
    engine_used = "offline"

    if use_groq:
        if Groq is None:
            return {"error": "Groq SDK not installed. Install requirements.txt."}
        api_key = get_groq_api_key()
        if not api_key:
            return {"error": "GROQ_API_KEY is not set in secrets or environment."}
        try:
            optimized = refine_with_groq(optimized, api_key, model, language_params["language_label"])
            engine_used = f"groq:{model}"
        except Exception as e:
            return {"error": f"Groq refinement failed: {e}"}
    else:
        optimized = offline_enhance(optimized, language_params["language_label"])

    return {
        "optimized": optimized,
        "technique": technique,
        "chosen": chosen,
        "ranked": ranked,
        "engine_used": engine_used,
        "objective": objective,
        "language": language_choice,
        "question": user_input,
    }


# ----------------------------------------------------------------------------
# Page config + styling — minimal, ChatGPT-like
# ----------------------------------------------------------------------------

st.set_page_config(
    page_title="Prompt Optimizer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    }

    #MainMenu {visibility: hidden;}

    .stApp {
        background: #FFFFFF;
    }

    header[data-testid="stHeader"] {
        background: #FFFFFF;
        height: 3rem;
    }

    div[data-testid="stBottomBlockContainer"],
    div[data-testid="stBottom"] > div {
        background: #FFFFFF;
    }

    div[data-testid="stChatInput"] {
        background: #FFFFFF;
        border: 1px solid #D9D9E3;
        border-radius: 1.5rem;
    }

    div[data-testid="stChatInput"] textarea {
        color: #0D0D0D;
    }

    div[data-testid="stChatMessageAvatarUser"],
    div[data-testid="stChatMessageAvatarCustom"],
    div[data-testid="stChatMessageAvatarAssistant"] {
        display: none !important;
    }

    [data-testid="stSidebarCollapsedControl"] svg,
    [data-testid="stSidebarHeader"] svg,
    button[kind="header"] svg {
        color: #0D0D0D !important;
        fill: #0D0D0D !important;
    }

    [data-testid="stSidebarCollapsedControl"] {
        background: #FFFFFF;
        border: 1px solid #E5E5E5;
        border-radius: 6px;
    }

    div[data-baseweb="select"] > div {
        background: #FFFFFF;
        border-color: #D9D9E3;
        color: #0D0D0D;
    }

    ul[data-testid="stSelectboxVirtualDropdown"] {
        background: #FFFFFF;
    }

    div[data-testid="stExpander"] p,
    div[data-testid="stExpander"] summary {
        color: #0D0D0D !important;
    }

    .block-container {
        max-width: 760px;
        padding-top: 2rem;
        padding-bottom: 8rem;
    }

    section[data-testid="stSidebar"] {
        background: #F9F9F9;
        border-right: 1px solid #E5E5E5;
    }

    section[data-testid="stSidebar"] .block-container {
        max-width: none;
        padding-top: 1.25rem;
    }

    .app-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #0D0D0D;
        margin-bottom: 0.9rem;
    }

    .sidebar-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        color: #A0A0A0;
        margin: 1.1rem 0 0.4rem 0;
    }

    div[data-testid="stChatMessage"] {
        background: transparent;
        padding: 0.5rem 0;
        border: none;
    }

    div[data-testid="stChatMessage"]:has(div.user-bubble) {
        display: flex;
        justify-content: flex-end;
    }

    .user-bubble {
        background: #F4F4F4;
        color: #0D0D0D;
        padding: 0.6rem 1rem;
        border-radius: 1.1rem;
        display: inline-block;
        max-width: 85%;
        font-size: 0.95rem;
        line-height: 1.5;
    }

    .prompt-label {
        color: #0D0D0D;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }

    .prompt-sublabel {
        color: #8E8E8E;
        font-size: 0.78rem;
        margin-bottom: 0.6rem;
    }

    .meta-line {
        color: #8E8E8E;
        font-size: 0.78rem;
        margin: 0.6rem 0 0.1rem 0;
    }

    .meta-line b {
        color: #4B4B4B;
        font-weight: 600;
    }

    .cost-box {
        background: #F7F9F7;
        border: 1px solid #E1EAE1;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin: 0.7rem 0;
        font-size: 0.83rem;
        color: #2E4A2E;
    }

    .cost-box b {
        color: #1B3B1B;
    }

    .answer-col-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-bottom: 0.4rem;
    }

    .answer-col-naive {
        color: #A05A2C;
    }

    .answer-col-optimized {
        color: #1E7A4F;
    }

    .answer-box {
        border: 1px solid #E5E5E5;
        border-radius: 10px;
        padding: 0.7rem 0.9rem;
        font-size: 0.85rem;
        line-height: 1.5;
        color: #0D0D0D;
        background: #FAFAFA;
        height: 100%;
    }

    .empty-state {
        text-align: center;
        padding: 3rem 1rem 1.5rem 1rem;
    }

    .empty-state h3 {
        color: #0D0D0D;
        font-weight: 600;
        font-size: 1.15rem;
        margin-bottom: 0.35rem;
    }

    .empty-state p {
        color: #8E8E8E;
        font-size: 0.88rem;
    }

    .stButton button {
        border-radius: 8px;
        border: 1px solid #E5E5E5;
        color: #0D0D0D;
        background: #FFFFFF;
        font-size: 0.85rem;
    }

    .stButton button:hover {
        border-color: #C7C7C7;
        background: #FAFAFA;
    }

    div[class*="st-key-chatrow_"] {
        margin-bottom: 1px;
    }

    div[class*="st-key-chatrow_"] div[data-testid="stButton"] button {
        border: none;
        background: transparent;
        min-height: 1.9rem;
        height: 1.9rem;
        padding: 0 0.5rem;
        font-size: 0.82rem;
        font-weight: 400;
        border-radius: 6px;
        text-align: left;
        justify-content: flex-start;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        display: block;
    }

    div[class*="st-key-chatrow_"] div[data-testid="stButton"] button:hover {
        background: #EFEFEF;
    }

    div[class*="st-key-chatrow_active_"] div[data-testid="stButton"] button {
        background: #ECECEC;
        font-weight: 600;
    }

    div[class*="st-key-delbtn_"] div[data-testid="stButton"] button {
        border: none;
        background: transparent;
        min-height: 1.9rem;
        height: 1.9rem;
        width: 100%;
        padding: 0;
        color: #B0B0B0;
        font-size: 0.85rem;
        border-radius: 6px;
    }

    div[class*="st-key-delbtn_"] div[data-testid="stButton"] button:hover {
        color: #E23636;
        background: #FBEAEA;
    }

    pre {
        border: 1px solid #E5E5E5 !important;
        border-radius: 10px !important;
        background: #F7F7F7 !important;
    }

    hr {
        border-color: #ECECEC;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Chat-session state (multi-conversation support)
# ----------------------------------------------------------------------------

def new_chat():
    cid = str(uuid.uuid4())
    st.session_state.chats[cid] = {"title": "New chat", "messages": []}
    st.session_state.chat_order.insert(0, cid)
    st.session_state.current_chat_id = cid
    return cid


if "chats" not in st.session_state:
    st.session_state.chats = {}
    st.session_state.chat_order = []
    st.session_state.current_chat_id = None
    new_chat()

if not st.session_state.chat_order:
    new_chat()


# ----------------------------------------------------------------------------
# Sidebar — settings + chat history + cost assumptions
# ----------------------------------------------------------------------------

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

with st.sidebar:
    st.markdown('<div class="app-title">Prompt Optimizer</div>', unsafe_allow_html=True)

    if st.button("＋ New chat", use_container_width=True):
        new_chat()
        st.rerun()

    st.markdown('<div class="sidebar-label">Chats</div>', unsafe_allow_html=True)

    for cid in st.session_state.chat_order:
        chat = st.session_state.chats[cid]
        is_active = cid == st.session_state.current_chat_id
        row_key = f"chatrow_active_{cid}" if is_active else f"chatrow_{cid}"

        with st.container(key=row_key):
            col1, col2 = st.columns([6, 1])
            with col1:
                if st.button(chat["title"], key=f"chatbtn_{cid}", use_container_width=True):
                    st.session_state.current_chat_id = cid
                    st.rerun()
            with col2:
                with st.container(key=f"delbtn_{cid}"):
                    if st.button("✕", key=f"delraw_{cid}", use_container_width=True):
                        del st.session_state.chats[cid]
                        st.session_state.chat_order.remove(cid)
                        if st.session_state.current_chat_id == cid:
                            st.session_state.current_chat_id = (
                                st.session_state.chat_order[0] if st.session_state.chat_order else None
                            )
                        if not st.session_state.chat_order:
                            new_chat()
                        st.rerun()

    st.markdown('<div class="sidebar-label">Output language</div>', unsafe_allow_html=True)
    language_choice = st.selectbox(
        "Output language", list(language_options.keys()), label_visibility="collapsed"
    )

    st.markdown('<div class="sidebar-label">Optimize for</div>', unsafe_allow_html=True)
    objective = st.selectbox(
        "Optimize for",
        ["Balanced (quality/cost)", "Highest quality", "Lowest latency", "Lowest tokens"],
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-label">Groq refinement</div>', unsafe_allow_html=True)
    use_groq = st.toggle("Refine with live Groq call", value=False)
    model = "llama-3.1-8b-instant"
    if use_groq:
        model = st.selectbox(
            "Groq model",
            ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
            index=0,
        )
        with st.expander("API key (optional)"):
            manual_key = st.text_input("GROQ_API_KEY override", type="password")
            if manual_key:
                st.session_state["manual_groq_key"] = manual_key
    else:
        st.caption("Off — uses fast offline template enhancement instead of a live API call.")

    st.markdown('<div class="sidebar-label">Cost assumptions</div>', unsafe_allow_html=True)
    with st.expander("Illustrative $ / 1M tokens"):
        frontier_rate = st.number_input(
            "Frontier model API (blended)", value=7.50, min_value=0.0, step=0.5,
            help="Rough blended input+output rate for a proprietary frontier model API."
        )
        resource_rate = st.number_input(
            "Groq-hosted 8B model (blended)", value=0.08, min_value=0.0, step=0.01,
            help="Rough blended rate for a small open-weight model on a fast inference provider."
        )
        st.caption("Defaults are illustrative — edit to match current provider pricing before citing.")

    with st.expander("Why not just use ChatGPT?"):
        st.write(
            "This tool isn't meant to replace ChatGPT or Claude for a student "
            "asking a one-off question — those already answer well on their own. "
            "It's built for **developers running their own low-cost model** "
            "(e.g. an 8B model, for an EdTech app serving thousands of Hinglish "
            "queries a day) who can't afford frontier-model API pricing at scale. "
            "It tells them which prompt strategy gets the best quality out of "
            "that cheap model — backed by 3,600 benchmarked responses — instead "
            "of guessing."
        )


# ----------------------------------------------------------------------------
# Main area
# ----------------------------------------------------------------------------

summary = load_summary()
if summary is None:
    st.error("Missing `results_summary.csv` or `evaluated_results.csv` in the project folder. Run the evaluation pipeline first.")
    st.stop()

current_chat = st.session_state.chats[st.session_state.current_chat_id]
messages = current_chat["messages"]

if not messages:
    st.markdown(
        """
        <div class="empty-state">
            <h3>Ask a student question</h3>
            <p>You'll get a ready-to-use optimized prompt — built from the best-benchmarked strategy — not a direct answer.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    suggestions = [
        "Photosynthesis kaise hota hai?",
        "Newton's laws of motion samjhao",
        "What is the difference between speed and velocity?",
        "Trigonometry ke formulas yaad rakhne ka tarika batao",
    ]
    cols = st.columns(2)
    for i, s in enumerate(suggestions):
        with cols[i % 2]:
            if st.button(s, key=f"sugg_{i}", use_container_width=True):
                st.session_state.pending_input = s
                st.rerun()

for idx, msg in enumerate(messages):
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        with st.chat_message("assistant"):
            if msg.get("error"):
                st.error(msg["content"])
                continue

            st.markdown('<div class="prompt-label">Optimized prompt</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="prompt-sublabel">Copy this into ChatGPT, Claude, or your own LLM to get the answer.</div>',
                unsafe_allow_html=True,
            )
            st.code(msg["content"], language="text")

            meta = msg["meta"]
            chosen = meta["chosen"]
            technique_label = TECHNIQUE_LABELS.get(meta["technique"], meta["technique"])
            engine_label = "Groq-refined" if meta["engine_used"].startswith("groq") else "Offline"

            st.markdown(
                f"""
                <div class="meta-line">
                    <b>{technique_label}</b> &nbsp;·&nbsp;
                    Quality {chosen['quality_score_mean']:.2f}/10 &nbsp;·&nbsp;
                    {chosen['latency_sec_mean']:.2f}s &nbsp;·&nbsp;
                    {chosen['tokens_mean']:.0f} tokens &nbsp;·&nbsp;
                    {engine_label}
                </div>
                """,
                unsafe_allow_html=True,
            )

            dcol1, dcol2 = st.columns([1, 5])
            with dcol1:
                st.download_button(
                    "Download .txt",
                    data=msg["content"],
                    file_name=f"optimized_prompt_{idx}.txt",
                    mime="text/plain",
                    key=f"dl_{st.session_state.current_chat_id}_{idx}",
                )

            # --- Cost at scale ---
            tokens_est = chosen["tokens_mean"]
            cost_frontier = estimate_cost(tokens_est, frontier_rate)
            cost_resource = estimate_cost(tokens_est, resource_rate)
            savings_pct = (1 - cost_resource / cost_frontier) * 100 if cost_frontier > 0 else 0
            daily_queries = 100_000
            st.markdown(
                f"""
                <div class="cost-box">
                    💰 <b>Cost at scale</b> (illustrative, ~{tokens_est:.0f} tokens/answer, {daily_queries:,} queries/day)<br>
                    Frontier model API: <b>${cost_frontier * daily_queries:,.0f}/day</b> &nbsp;→&nbsp;
                    This approach on a small model: <b>${cost_resource * daily_queries:,.0f}/day</b>
                    &nbsp;(~{savings_pct:.0f}% cheaper)
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("Compare top techniques"):
                ranked = meta["ranked"].copy()
                ranked["technique"] = ranked["technique"].map(lambda t: TECHNIQUE_LABELS.get(t, t))
                top5 = ranked.head(5).set_index("technique")

                st.caption("Quality score by technique")
                st.bar_chart(top5[["quality_score_mean"]].rename(
                    columns={"quality_score_mean": "Quality"}
                ))

                display = ranked.rename(columns={
                    "technique": "Technique",
                    "quality_score_mean": "Quality",
                    "latency_sec_mean": "Latency (s)",
                    "tokens_mean": "Tokens",
                    "balanced_score": "Balanced score",
                })
                st.dataframe(
                    display.reset_index()[["Technique", "Quality", "Latency (s)", "Tokens", "Balanced score"]].head(5),
                    use_container_width=True,
                    hide_index=True,
                )

            with st.expander("See it in action: naive vs. optimized answer"):
                st.caption(
                    "Runs the raw question and the optimized prompt through the same small model "
                    "live, so you can see the actual quality difference."
                )
                comparison = msg.get("comparison")
                if comparison is None:
                    if st.button("Generate live comparison", key=f"cmp_{st.session_state.current_chat_id}_{idx}"):
                        api_key = get_groq_api_key()
                        if Groq is None:
                            st.error("Groq SDK not installed.")
                        elif not api_key:
                            st.error("No GROQ_API_KEY found. Add one in the Groq refinement section in the sidebar.")
                        else:
                            with st.spinner("Calling the model twice (naive + optimized)..."):
                                try:
                                    naive_answer = get_llm_answer(meta["question"], api_key, model)
                                    optimized_answer = get_llm_answer(msg["content"], api_key, model)
                                    msg["comparison"] = {"naive": naive_answer, "optimized": optimized_answer}
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Live comparison failed: {e}")
                else:
                    ccol1, ccol2 = st.columns(2)
                    with ccol1:
                        st.markdown('<div class="answer-col-label answer-col-naive">Naive prompt</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="answer-box">{comparison["naive"]}</div>', unsafe_allow_html=True)
                    with ccol2:
                        st.markdown('<div class="answer-col-label answer-col-optimized">Optimized prompt</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="answer-box">{comparison["optimized"]}</div>', unsafe_allow_html=True)

pending = st.session_state.pop("pending_input", None)
user_input = st.chat_input("Message Prompt Optimizer...")
final_input = pending or user_input

if final_input:
    messages.append({"role": "user", "content": final_input})

    if current_chat["title"] == "New chat":
        current_chat["title"] = final_input[:28] + ("…" if len(final_input) > 28 else "")

    with st.spinner("Building optimized prompt..."):
        result = process_question(
            final_input, language_choice, language_options, objective, use_groq, model, summary
        )

    if "error" in result:
        messages.append({"role": "assistant", "content": result["error"], "error": True})
    else:
        messages.append({
            "role": "assistant",
            "content": result["optimized"],
            "meta": result,
        })

    st.rerun()