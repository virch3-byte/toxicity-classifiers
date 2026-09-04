"""
Toxicity Classifier Showcase
-----------------------------
Streamlit app to demo the two fine-tuned OPT toxicity classifiers
(facebook/opt-1.3b and facebook/opt-2.7b) produced in nlpasgm.ipynb.

Run with:  streamlit run app.py
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
st.set_page_config(page_title="Toxicity Classifier Showcase", page_icon="🛡️", layout="wide")

ID2LABEL = {0: "non-toxic", 1: "toxic"}

MODEL_OPTIONS = {
    "OPT-1.3B": "virch3/toxicity-models/opt1.3b",
    "OPT-2.7B": "virch3/toxicity-models/opt2.7b",
}

HISTORY_FILES = {
    "OPT-1.3B": "virch3/toxicity-models/hist/trainer_state13.json",
    "OPT-2.7B": "virch3/toxicity-models/hist/trainer_state27.json",
}

STOPWORDS_CACHE = None


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_stopwords():
    """Load NLTK English stopwords, downloading them on first run if needed."""
    import nltk
    from nltk.corpus import stopwords

    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    return set(stopwords.words("english"))


def preprocess(text: str) -> str:
    """Mirror the exact preprocessing used in the notebook's training/inference cells."""
    stop_words = get_stopwords()
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = " ".join(word for word in text.split(" ") if word not in stop_words)
    return text


@st.cache_resource(show_spinner=True)
def load_model(model_path: str):
    """Load a fine-tuned classifier + tokenizer from a local folder."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=2,
        id2label=ID2LABEL,
        label2id={v: k for k, v in ID2LABEL.items()},
        torch_dtype=torch.float16,  # halves memory usage vs. default float32
        low_cpu_mem_usage=True,
    )
    model.eval()
    return tokenizer, model


def predict(text: str, tokenizer, model):
    processed = preprocess(text)
    inputs = tokenizer(processed, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze().numpy()
    pred_id = int(np.argmax(probs))
    return {
        "processed_text": processed,
        "label": ID2LABEL[pred_id],
        "probs": {ID2LABEL[i]: float(p) for i, p in enumerate(probs)},
    }


def model_dir_ready(path: str) -> bool:
    p = Path(path)
    return p.exists() and any(p.iterdir())


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.title("🛡️ Settings")

st.sidebar.markdown("**Model folders** (edit if your saved models live elsewhere)")
model_paths = {}
for name, default_path in MODEL_OPTIONS.items():
    model_paths[name] = st.sidebar.text_input(f"{name} path", value=default_path)

selected_models = st.sidebar.multiselect(
    "Models to load",
    options=list(MODEL_OPTIONS.keys()),
    default=list(MODEL_OPTIONS.keys()),
)

st.sidebar.divider()
show_history = st.sidebar.checkbox("Show training history plots", value=True)

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
st.title("🛡️ Toxicity Classifier Showcase")
st.caption("Fine-tuned OPT-1.3B / OPT-2.7B sequence classifiers — toxic vs. non-toxic text detection")

tab_predict, tab_history = st.tabs(["🔍 Try it out", "📈 Training history"])

# ---- Prediction tab -------------------------------------------------------
with tab_predict:
    text_input = st.text_area(
        "Enter a sentence to classify",
        placeholder="Type or paste a sentence here...",
        height=100,
    )
    run = st.button("Classify", type="primary", use_container_width=False)

    if run:
        if not text_input.strip():
            st.warning("Please enter some text first.")
        elif not selected_models:
            st.warning("Select at least one model in the sidebar.")
        else:
            cols = st.columns(len(selected_models))
            for col, name in zip(cols, selected_models):
                path = model_paths[name]
                with col:
                    st.subheader(name)
                    if not model_dir_ready(path):
                        st.error(f"No model files found at `{path}`. Update the path in the sidebar.")
                        continue
                    try:
                        with st.spinner(f"Loading {name}..."):
                            tokenizer, model = load_model(path)
                        result = predict(text_input, tokenizer, model)
                    except Exception as e:
                        st.error(f"Failed to load/run {name}: {e}")
                        continue

                    label = result["label"]
                    badge = "🔴 TOXIC" if label == "toxic" else "🟢 NON-TOXIC"
                    st.markdown(f"### {badge}")

                    probs_df = pd.DataFrame(
                        {"label": list(result["probs"].keys()), "probability": list(result["probs"].values())}
                    ).set_index("label")
                    st.bar_chart(probs_df)

                    with st.expander("Preprocessed text"):
                        st.code(result["processed_text"] or "(empty after preprocessing)")

# ---- History tab ------------------------------------------------------
with tab_history:
    if not show_history:
        st.info("Training history plots are turned off. Enable them in the sidebar.")
    else:
        histories = {}
        for name, hist_path in HISTORY_FILES.items():
            p = Path(hist_path)
            if p.exists():
                with open(p) as f:
                    log_history = json.load(f)["log_history"][1::2]
                histories[name] = {
                    "auc": [entry.get("eval_auc") for entry in log_history],
                    "f1": [entry.get("eval_f1") for entry in log_history],
                    "loss": [entry.get("eval_loss") for entry in log_history],
                }

        if not histories:
            st.info(
                "No trainer_state JSON files found. Place them at "
                "`models/hist/trainer_state13.json` and `models/hist/trainer_state27.json` "
                "(copy from your Google Drive `models/hist/` folder) to see plots here."
            )
        else:
            metric_labels = {"auc": "AUC", "f1": "F1 score", "loss": "Loss"}
            for metric, label in metric_labels.items():
                st.subheader(f"{label} by epoch")
                max_len = max(len(h[metric]) for h in histories.values())
                chart_df = pd.DataFrame(
                    {name: h[metric] + [None] * (max_len - len(h[metric])) for name, h in histories.items()},
                    index=range(1, max_len + 1),
                )
                st.line_chart(chart_df)
