import os
from pathlib import Path
import numpy as np
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ==========================================
# 1. إعداد الصفحة والهوية البصرية
# ==========================================
st.set_page_config(
    page_title="6-Class Emotion Classifier",
    page_icon="✨",
    layout="wide",
)

CUSTOM_CSS = """
<style>
    .stApp {
        background: radial-gradient(circle at top right, #eef2ff 0%, #fdf4ff 45%, #f0fdf4 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .block-container {
        max-width: 1000px !important;
        padding-top: 2rem !important;
        padding-bottom: 2.5rem !important;
    }
    #hero-banner {
        background: linear-gradient(120deg, #312e81 0%, #6d28d9 48%, #db2777 100%);
        color: #ffffff;
        padding: 24px 28px;
        border-radius: 18px;
        margin-bottom: 20px;
        box-shadow: 0 12px 28px rgba(109, 40, 217, 0.20);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    #hero-banner h1 {
        margin: 0 0 8px 0;
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: 0.2px;
        color: #ffffff;
    }
    #hero-banner p {
        margin: 0 0 12px 0;
        font-size: 0.98rem;
        color: #f5d0fe;
    }
    .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    .emotion-pill {
        background: rgba(255, 255, 255, 0.16);
        backdrop-filter: blur(6px);
        padding: 4px 11px;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.25);
    }
    .dominant-box {
        background: linear-gradient(135deg, #eef2ff 0%, #fdf4ff 100%);
        border: 1.5px solid #c4b5fd;
        border-radius: 12px;
        padding: 14px 16px;
        font-size: 1.15rem;
        font-weight: 700;
        color: #1e1b4b;
        margin-bottom: 14px;
    }
    .warning-box {
        background: #fffbeb;
        border: 1.5px solid #fcd34d;
        border-radius: 12px;
        padding: 12px 14px;
        color: #92400e;
        font-weight: 600;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #7c3aed 0%, #ec4899 100%) !important;
        border: none !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        box-shadow: 0 6px 16px rgba(219, 39, 119, 0.25) !important;
        transition: filter 0.15s ease;
    }
    div.stButton > button[kind="primary"]:hover {
        filter: brightness(1.06);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ==========================================
# 2. تحديد مصدر الموديل والفئات
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
HF_MODEL_REPO = os.getenv(
    "HF_MODEL_REPO", "mo123essam/emotion-roberta-large-6class"
)

LOCAL_CANDIDATES = [
    BASE_DIR / "emotion_6class_merged_model",
    BASE_DIR / "emotion_model",
    BASE_DIR,
]

local_model_dir = next(
    (p for p in LOCAL_CANDIDATES if (p / "config.json").exists()), None
)
MODEL_SOURCE = str(local_model_dir) if local_model_dir else HF_MODEL_REPO

EMOJI_LABELS = {
    "joy": "😄 Joy",
    "sadness": "😢 Sadness",
    "anger": "😠 Anger",
    "fear": "😨 Fear",
    "surprise": "😲 Surprise",
    "disgust": "🤢 Disgust",
}

FALLBACK_ORDER = ["joy", "sadness", "anger", "fear", "surprise", "disgust"]

SAMPLE_SENTENCES = [
    "I finally got the job offer I've been dreaming about for months!",
    (
        "It hurts to realize how much effort I put into something that didn't"
        " work out."
    ),
    (
        "It makes me furious that the system crashed right before I saved my"
        " work."
    ),
    (
        "I'm terrified that I won't be ready in time for tomorrow's"
        " presentation."
    ),
    "I never expected to run into you all the way out here!",
    "The smell coming from that trash bin is honestly revolting.",
    "Oh great, another delay right when we were about to wrap things up.",
]


def get_ordered_labels(config, num_classes: int):
  id2label = getattr(config, "id2label", None)
  if isinstance(id2label, dict) and len(id2label) == num_classes:
    parsed = []
    has_real_names = True
    for i in range(num_classes):
      val = id2label.get(i, id2label.get(str(i), ""))
      clean_val = str(val).strip().lower()
      if not clean_val or clean_val.startswith("label_"):
        has_real_names = False
        break
      parsed.append(clean_val)
    if has_real_names:
      return parsed
  return FALLBACK_ORDER[:num_classes]


def format_label(label_key: str) -> str:
  return EMOJI_LABELS.get(label_key.lower(), label_key.capitalize())


# ==========================================
# 3. تحميل Tokenizer والموديل مرة واحدة
# ==========================================
@st.cache_resource(show_spinner="⏳ Loading RoBERTa-Large emotion model...")
def load_model_and_tokenizer(model_source: str):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  tok = AutoTokenizer.from_pretrained(model_source)
  mdl = AutoModelForSequenceClassification.from_pretrained(model_source).to(
      device
  )
  mdl.eval()
  labels = get_ordered_labels(mdl.config, mdl.config.num_labels)
  return tok, mdl, labels, device


tokenizer, model, LABEL_ORDER, device = load_model_and_tokenizer(MODEL_SOURCE)


# ==========================================
# 4. دالة التنبؤ
# ==========================================
def predict_emotion(text: str):
  encoded = tokenizer(
      text.strip(),
      return_tensors="pt",
      truncation=True,
      max_length=128,
  ).to(device)

  with torch.no_grad():
    logits = model(**encoded).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

  label_to_prob = {
      LABEL_ORDER[i]: float(probs[i]) for i in range(len(LABEL_ORDER))
  }

  sorted_scores = {
      format_label(k): v
      for k, v in sorted(
          label_to_prob.items(), key=lambda item: item[1], reverse=True
      )
  }

  best_idx = int(np.argmax(probs))
  best_key = LABEL_ORDER[best_idx]
  dominant = f"{format_label(best_key)}  •  {probs[best_idx] * 100:.1f}%"

  return dominant, sorted_scores


# ==========================================
# 5. تهيئة الحالة (Session State) وواجهة التطبيق
# ==========================================
if "text_input" not in st.session_state:
  st.session_state.text_input = ""
if "prediction_result" not in st.session_state:
  st.session_state.prediction_result = None


def set_sample_text(sample: str):
  st.session_state.text_input = sample
  st.session_state.prediction_result = predict_emotion(sample)


def clear_all():
  st.session_state.text_input = ""
  st.session_state.prediction_result = None


st.markdown(
    """
    <div id="hero-banner">
        <h1>✨ 6-Class Emotion Classification Studio</h1>
        <p>RoBERTa-Large (LoRA) Fine-Tuned on GoEmotions • Checkpoint-4310 (Acc: 82.88% | Macro F1: 0.7512)</p>
        <div class="badge-row">
            <span class="emotion-pill">😄 Joy</span>
            <span class="emotion-pill">😢 Sadness</span>
            <span class="emotion-pill">😠 Anger</span>
            <span class="emotion-pill">😨 Fear</span>
            <span class="emotion-pill">😲 Surprise</span>
            <span class="emotion-pill">🤢 Disgust</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
  st.subheader("✍️ Input Text")
  st.text_area(
      label="Enter an English sentence",
      key="text_input",
      height=140,
      placeholder="Type or paste an English sentence here...",
      label_visibility="collapsed",
  )

  btn_col1, btn_col2 = st.columns([1, 1])
  with btn_col1:
    st.button("🧹 Clear", on_click=clear_all, use_container_width=True)
  with btn_col2:
    analyze_clicked = st.button(
        "🚀 Analyze Emotion",
        type="primary",
        use_container_width=True,
    )

with col_right:
  st.subheader("🏆 Dominant Emotion")

  if analyze_clicked:
    current_text = st.session_state.text_input
    if not current_text or not current_text.strip():
      st.markdown(
          '<div class="warning-box">⚠️ Please enter a sentence first.</div>',
          unsafe_allow_html=True,
      )
      st.session_state.prediction_result = None
    else:
      with st.spinner("Analyzing emotion..."):
        st.session_state.prediction_result = predict_emotion(current_text)

  if st.session_state.prediction_result is not None:
    dominant_text, scores_dict = st.session_state.prediction_result
    st.markdown(
        f'<div class="dominant-box">{dominant_text}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("**📊 Confidence Distribution**")
    for label_name, prob_val in scores_dict.items():
      pct = prob_val * 100.0
      st.markdown(f"**{label_name}** — `{pct:.1f}%`")
      st.progress(min(max(float(prob_val), 0.0), 1.0))
  else:
    st.info("Enter text or click a sample below to view emotion probabilities.")

# ==========================================
# 6. أمثلة سريعة للاختبار
# ==========================================
st.markdown("---")
st.subheader("💡 Quick Test Samples (Click any sentence to load & analyze)")

for i in range(0, len(SAMPLE_SENTENCES), 2):
  row_cols = st.columns(2, gap="medium")
  for j in range(2):
    idx = i + j
    if idx < len(SAMPLE_SENTENCES):
      sample_str = SAMPLE_SENTENCES[idx]
      short_label = (
          sample_str[:78] + "..." if len(sample_str) > 78 else sample_str
      )
      with row_cols[j]:
        st.button(
            short_label,
            key=f"sample_{idx}",
            on_click=set_sample_text,
            args=(sample_str,),
            use_container_width=True,
        )
