import os
from pathlib import Path
import gradio as gr
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
  import spaces

  gpu_decorator = spaces.GPU
except ImportError:

  def gpu_decorator(fn):
    return fn


# ==========================================
# 1. تحديد مصدر الموديل (محلي أولاً ثم Hugging Face Hub)
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

# ==========================================
# 2. تعريف الفئات وضبط التوافق مع id2label
# ==========================================
EMOJI_LABELS = {
    "joy": "😄 Joy",
    "sadness": "😢 Sadness",
    "anger": "😠 Anger",
    "fear": "😨 Fear",
    "surprise": "😲 Surprise",
    "disgust": "🤢 Disgust",
}

FALLBACK_ORDER = ["joy", "sadness", "anger", "fear", "surprise", "disgust"]


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
# 3. تحميل Tokenizer والموديل
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_SOURCE)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_SOURCE)
model.eval()

LABEL_ORDER = get_ordered_labels(model.config, model.config.num_labels)


# ==========================================
# 4. دالة التنبؤ
# ==========================================
@gpu_decorator
def predict_emotion(text: str):
  if not text or not text.strip():
    return "⚠️ Please enter a sentence first.", {}

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)

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
# 5. التنسيق البصري والألوان
# ==========================================
CUSTOM_CSS = """
.gradio-container {
    max-width: 1000px !important;
    margin: auto !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: radial-gradient(circle at top right, #eef2ff 0%, #fdf4ff 45%, #f0fdf4 100%);
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
.input-card {
    background: linear-gradient(180deg, #ffffff 0%, #f5f3ff 100%);
    border: 1.5px solid #ddd6fe;
    border-radius: 16px;
    padding: 16px;
    box-shadow: 0 6px 18px rgba(79, 70, 229, 0.07);
}
.output-card {
    background: linear-gradient(180deg, #ffffff 0%, #ecfeff 100%);
    border: 1.5px solid #bae6fd;
    border-radius: 16px;
    padding: 16px;
    box-shadow: 0 6px 18px rgba(14, 165, 233, 0.08);
}
.examples-card {
    background: linear-gradient(180deg, #ffffff 0%, #fffbeb 100%);
    border: 1.5px solid #fde68a;
    border-radius: 16px;
    padding: 14px 16px;
    margin-top: 14px;
    box-shadow: 0 6px 18px rgba(245, 158, 11, 0.08);
}
#analyze-btn {
    background: linear-gradient(90deg, #7c3aed 0%, #ec4899 100%) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    box-shadow: 0 6px 16px rgba(219, 39, 119, 0.25) !important;
}
#analyze-btn:hover {
    filter: brightness(1.06);
}
"""

theme = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="amber",
    neutral_hue="slate",
    radius_size="lg",
)

SAMPLE_SENTENCES = [
    ["I finally got the job offer I've been dreaming about for months!"],
    [
        "It hurts to realize how much effort I put into something that didn't"
        " work out."
    ],
    [
        "It makes me furious that the system crashed right before I saved my"
        " work."
    ],
    [
        "I'm terrified that I won't be ready in time for tomorrow's"
        " presentation."
    ],
    ["I never expected to run into you all the way out here!"],
    ["The smell coming from that trash bin is honestly revolting."],
    ["Oh great, another delay right when we were about to wrap things up."],
]

with gr.Blocks(title="6-Class Emotion Classifier") as demo:
  gr.HTML("""
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
    """)

  with gr.Row(equal_height=False):
    with gr.Column(scale=1, elem_classes="input-card"):
      text_input = gr.Textbox(
          lines=4,
          placeholder="Type or paste an English sentence here...",
          label="✍️ Input Text",
      )
      with gr.Row():
        clear_btn = gr.ClearButton(components=[text_input], value="🧹 Clear")
        submit_btn = gr.Button(
            "🚀 Analyze Emotion", variant="primary", elem_id="analyze-btn"
        )

    with gr.Column(scale=1, elem_classes="output-card"):
      dominant_out = gr.Textbox(
          label="🏆 Dominant Emotion",
          interactive=False,
      )
      scores_out = gr.Label(
          num_top_classes=6,
          label="📊 Confidence Distribution",
      )

  with gr.Column(elem_classes="examples-card"):
    gr.Examples(
        examples=SAMPLE_SENTENCES,
        inputs=text_input,
        label="💡 Quick Test Samples (Click any sentence to load)",
    )

  submit_btn.click(
      fn=predict_emotion,
      inputs=text_input,
      outputs=[dominant_out, scores_out],
  )
  text_input.submit(
      fn=predict_emotion,
      inputs=text_input,
      outputs=[dominant_out, scores_out],
  )

if __name__ == "__main__":
  is_spaces = bool(os.getenv("SPACE_ID"))
  if is_spaces:
    demo.launch(theme=theme, css=CUSTOM_CSS)
  else:
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme=theme,
        css=CUSTOM_CSS,
    )
