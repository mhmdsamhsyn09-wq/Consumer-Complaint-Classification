from __future__ import annotations

import html
import re
import threading
import time
from typing import Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import torch

# =============================================================================
# 1. CONFIGURATION  (حط هنا مسار/اسم الموديل بتاعك على Hugging Face Hub)
# =============================================================================

# صيغة المسار على Hugging Face Hub: "USERNAME/REPO_NAME"
# مثال: "mshmdsamhsyn09-wq/consumer-complaint-distilbert"
HF_MODEL_REPO = "mo123essam/Consumer-Complaint-Classification"

CLASS_IDS: List[str] = [
    "credit_card",
    "credit_reporting",
    "debt_collection",
    "mortgages_and_loans",
    "retail_banking",
]
DISPLAY_NAMES = {
    "credit_card": "Credit Card",
    "credit_reporting": "Credit Reporting",
    "debt_collection": "Debt Collection",
    "mortgages_and_loans": "Mortgages and Loans",
    "retail_banking": "Retail Banking",
}
CLASS_ICONS = {
    "credit_card": "💳",
    "credit_reporting": "📊",
    "debt_collection": "📞",
    "mortgages_and_loans": "🏠",
    "retail_banking": "🏦",
}
CLASS_COLORS = {
    "credit_card": "#6366f1",
    "credit_reporting": "#0ea5e9",
    "debt_collection": "#f97316",
    "mortgages_and_loans": "#10b981",
    "retail_banking": "#a855f7",
}

MAX_TOKENS = 512
MIN_WORDS_WARNING = 8

BENCHMARK = {
    "DistilBERT (Full)": {
        "accuracy": None,
        "f1": None,
        "params": "~66M (all trainable)",
    },
    "DistilBERT (LoRA)": {
        "accuracy": None,
        "f1": None,
        "params": "~66M (adapters only trainable)",
    },
    "LSTM": {"accuracy": None, "f1": None, "params": "from scratch"},
    "GRU": {"accuracy": None, "f1": None, "params": "from scratch"},
    "Simple RNN": {"accuracy": None, "f1": None, "params": "from scratch"},
}

EXAMPLES = [
    [
        (
            "I was charged an annual fee on my credit card that I was never"
            " told about, and the issuer refuses to reverse the interest and"
            " late fees on my statement."
        )
    ],
    [
        (
            "There is an account on my credit report that does not belong to"
            " me. I disputed it with the bureau twice and they keep verifying"
            " the incorrect information."
        )
    ],
    [
        (
            "A collection agency keeps calling me at work about a debt I do not"
            " owe. They threatened legal action and will not send written"
            " validation of the debt."
        )
    ],
    [
        (
            "My mortgage servicer misapplied my payments and raised my escrow"
            " without explanation. I have asked for a loan modification but they"
            " lost my paperwork."
        )
    ],
    [
        (
            "My checking account was overdrawn because the bank held my deposit"
            " for days, then charged multiple overdraft fees and closed my"
            " savings account without notice."
        )
    ],
]

# =============================================================================
# 2. DEVICE + MODEL LOADER
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_LOAD_LOCK = threading.Lock()
_CACHED_PREDICTOR: Optional["BasePredictor"] = None


class BasePredictor:
  is_demo = False
  device_label = "cpu"
  max_len = MAX_TOKENS

  def predict(self, text: str) -> Tuple[np.ndarray, bool]:
    raise NotImplementedError


class HFPredictor(BasePredictor):
  """Loads full fine-tuned DistilBERT directly from Hugging Face Hub."""

  def __init__(self, repo_id: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not repo_id or "YOUR_USERNAME" in repo_id:
      raise ValueError("Set HF_MODEL_REPO to your Hugging Face Hub repo id.")

    self.tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForSequenceClassification.from_pretrained(repo_id)
    self.model = model.to(DEVICE).eval()
    self.device_label = DEVICE.type
    self.labels = self._resolve_labels(model.config)
    self.max_len = min(
        MAX_TOKENS,
        getattr(self.tokenizer, "model_max_length", MAX_TOKENS) or MAX_TOKENS,
    )

  @staticmethod
  def _resolve_labels(config) -> List[str]:
    id2label = getattr(config, "id2label", None) or {}
    names = [str(id2label[i]) for i in sorted(id2label)] if id2label else []
    if names and set(names) == set(CLASS_IDS):
      return names
    return list(CLASS_IDS)

  @torch.no_grad()
  def predict(self, text: str) -> Tuple[np.ndarray, bool]:
    n_tokens = len(
        self.tokenizer(text, truncation=False, verbose=False)["input_ids"]
    )
    enc = self.tokenizer(
        text,
        truncation=True,
        max_length=self.max_len,
        padding=False,
        return_tensors="pt",
    ).to(DEVICE)
    logits = self.model(**enc).logits[0]
    probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    ordered = np.array([probs[self.labels.index(c)] for c in CLASS_IDS])
    return ordered, n_tokens > self.max_len


class DemoPredictor(BasePredictor):
  is_demo = True
  device_label = "demo"
  KEYWORDS = {
      "credit_card": [
          "credit card",
          "annual fee",
          "interest rate",
          "statement",
          "card",
          "apr",
          "rewards",
          "chargeback",
      ],
      "credit_reporting": [
          "credit report",
          "credit score",
          "bureau",
          "equifax",
          "experian",
          "transunion",
          "dispute",
          "inaccurate",
      ],
      "debt_collection": [
          "collection",
          "collector",
          "debt",
          "calling",
          "harass",
          "validation",
          "owe",
          "lawsuit",
      ],
      "mortgages_and_loans": [
          "mortgage",
          "loan",
          "escrow",
          "servicer",
          "modification",
          "foreclosure",
          "refinance",
          "student loan",
      ],
      "retail_banking": [
          "bank account",
          "checking",
          "savings",
          "overdraft",
          "deposit",
          "atm",
          "branch",
          "debit",
      ],
  }

  def predict(self, text: str) -> Tuple[np.ndarray, bool]:
    low = text.lower()
    scores = np.array(
        [sum(low.count(k) for k in self.KEYWORDS[c]) for c in CLASS_IDS],
        dtype=np.float64,
    )
    logits = scores * 1.5 + np.random.default_rng(len(text)).normal(
        0, 0.05, len(CLASS_IDS)
    )
    e = np.exp(logits - logits.max())
    return e / e.sum(), False


def get_predictor() -> Tuple[BasePredictor, Optional[str]]:
  global _CACHED_PREDICTOR
  with _LOAD_LOCK:
    if _CACHED_PREDICTOR is not None:
      return _CACHED_PREDICTOR, None
    try:
      _CACHED_PREDICTOR = HFPredictor(HF_MODEL_REPO)
      return _CACHED_PREDICTOR, None
    except Exception as exc:
      warning = (
          "⚠️ **Could not load model from Hugging Face** — running in **demo"
          f" mode**.\n`{type(exc).__name__}: {exc}`"
      )
      _CACHED_PREDICTOR = DemoPredictor()
      return _CACHED_PREDICTOR, warning


_startup_predictor, _startup_warning = get_predictor()
_STARTUP_NOTICE = _startup_warning or ""

# =============================================================================
# 3. INFERENCE + RENDERING HELPERS
# =============================================================================


def counter_html(text: str) -> str:
  text = text or ""
  words = len(text.split())
  return (
      f"<div class='counter'>{len(text):,} characters · {words:,}"
      " words</div>"
  )


def empty_card() -> str:
  return (
      "<div class='result-card empty'>"
      "<div class='rc-icon'>🔎</div>"
      "<div class='rc-title'>No prediction yet</div>"
      "<div class='rc-sub'>Enter a complaint and press <b>Classify"
      " Complaint</b>.</div>"
      "</div>"
  )


def result_card(class_id: str, confidence: float) -> str:
  color = CLASS_COLORS[class_id]
  return (
      f"<div class='result-card' style='--accent:{color}'>"
      f"<div class='rc-icon'>{CLASS_ICONS[class_id]}</div>"
      f"<div class='rc-label'>TOP PREDICTED CATEGORY</div>"
      f"<div class='rc-title'>{html.escape(DISPLAY_NAMES[class_id])}</div>"
      f"<div class='rc-conf'>{confidence * 100:.1f}%"
      " <span>confidence</span></div>"
      f"<div class='rc-bar'><div style='width:{confidence * 100:.1f}%'></div></div>"
      "<div class='rc-sub'>via DistilBERT (Full)</div>"
      "</div>"
  )


def latency_html(ms: Optional[float], device: str = "") -> str:
  if ms is None:
    return "<div class='latency'>⏱ Inference time: —</div>"
  dev = f" · device: {html.escape(device)}" if device else ""
  return (
      f"<div class='latency'>⏱ Inference time: <b>{ms:.0f} ms</b>{dev}</div>"
  )


def classify(text: str):
  if text is None or not text.strip():
    return (
        empty_card(),
        None,
        latency_html(None),
        "⚠️ Please enter a complaint narrative before classifying.",
    )

  predictor, load_warning = get_predictor()
  notices: List[str] = []
  if load_warning:
    notices.append(load_warning)
  elif predictor.is_demo:
    notices.append(
        "⚠️ **Demo mode** — no trained model is loaded; results are a keyword"
        " heuristic."
    )

  clean = re.sub(r"\s+", " ", text).strip()
  if len(clean.split()) < MIN_WORDS_WARNING:
    notices.append(
        "ℹ️ Very short input — predictions are less reliable. Add more detail"
        " if you can."
    )

  try:
    start = time.perf_counter()
    probs, truncated = predictor.predict(clean)
    elapsed_ms = (time.perf_counter() - start) * 1000
  except Exception as exc:
    return (
        empty_card(),
        None,
        latency_html(None),
        f"❌ Inference failed: `{type(exc).__name__}: {exc}`",
    )

  if truncated:
    notices.append(
        "✂️ Narrative exceeds the model's input limit and was **truncated**"
        f" (first {getattr(predictor, 'max_len', MAX_TOKENS)} tokens used)."
    )

  probs = np.asarray(probs, dtype=np.float64)
  probs = probs / probs.sum()
  top = int(np.argmax(probs))
  top_id = CLASS_IDS[top]

  label_dict = {DISPLAY_NAMES[c]: float(p) for c, p in zip(CLASS_IDS, probs)}
  return (
      result_card(top_id, float(probs[top])),
      label_dict,
      latency_html(elapsed_ms, predictor.device_label),
      "\n\n".join(notices),
  )


def clear_all():
  return "", counter_html(""), empty_card(), None, latency_html(None), ""


def benchmark_markdown() -> str:
  def fmt(v):
    return (
        "—"
        if v is None
        else f"{v:.2%}" if isinstance(v, float) and v <= 1 else str(v)
    )

  rows = "\n".join(
      f"| {'⭐ ' if n == 'DistilBERT (Full)' else ''}{n} | {fmt(b['accuracy'])} |"
      f" {fmt(b['f1'])} | {b['params']} |"
      for n, b in BENCHMARK.items()
  )
  return f"""
| Model | Accuracy | Macro F1 | Parameters |
|---|---|---|---|
{rows}

*(Fill in `BENCHMARK` at the top of `app.py` with your test-set metrics.)*

**Why DistilBERT (Full) was selected**

- **Contextual understanding:** pretrained on large corpora, it captures word meaning in context
  (e.g. *"card"* in a credit-card vs. debit/banking complaint), which embeddings learned from scratch struggle with.
- **Vs. LoRA:** updating *all* weights lets the model fully adapt to financial-complaint vocabulary,
  whereas LoRA trades some accuracy for far fewer trainable parameters and smaller checkpoints.
- **Vs. LSTM / GRU / Simple RNN:** recurrent models read sequentially, lose long-range information in
  lengthy narratives, and (Simple RNN especially) suffer from vanishing gradients; they also start with no language knowledge.
- **Trade-off:** the RNNs are lighter and faster on CPU, but DistilBERT gave the best accuracy/F1
  on the evaluation split while remaining fast enough for interactive use.
"""


# =============================================================================
# 4. UI
# =============================================================================

CSS = """
.gradio-container {max-width: 1200px !important; margin: auto;}
.hero {
  background: linear-gradient(135deg, #4338ca 0%, #6366f1 45%, #0ea5e9 100%);
  color: #fff; border-radius: 18px; padding: 28px 32px; margin-bottom: 14px;
  box-shadow: 0 10px 30px rgba(67,56,202,.25);
}
.hero h1 {margin: 0 0 6px 0; font-size: 1.9rem; font-weight: 800; color: #fff;}
.hero p {margin: 0 0 14px 0; opacity: .92; font-size: 1rem; color: #fff;}
.chips {display: flex; flex-wrap: wrap; gap: 8px;}
.chip {background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.35);
       padding: 5px 12px; border-radius: 999px; font-size: .85rem; backdrop-filter: blur(4px);}
.counter {text-align: right; font-size: .8rem; opacity: .7; margin-top: -4px;}
.latency {display: inline-block; padding: 8px 14px; border-radius: 10px; font-size: .9rem;
          background: var(--background-fill-secondary); border: 1px solid var(--border-color-primary);}
.result-card {
  --accent: #6366f1; text-align: center; padding: 22px 18px; border-radius: 16px;
  border: 2px solid var(--accent); background: color-mix(in srgb, var(--accent) 9%, transparent);
}
.result-card.empty {--accent: #94a3b8; border-style: dashed;}
.rc-icon {font-size: 2.2rem; line-height: 1;}
.rc-label {font-size: .72rem; letter-spacing: .12em; font-weight: 700; opacity: .7; margin-top: 6px;}
.rc-title {font-size: 1.6rem; font-weight: 800; color: var(--accent); margin: 4px 0;}
.rc-conf {font-size: 1.25rem; font-weight: 700;}
.rc-conf span {font-size: .85rem; font-weight: 500; opacity: .7;}
.rc-bar {height: 8px; border-radius: 99px; background: rgba(148,163,184,.3); margin: 10px auto 8px; max-width: 320px; overflow: hidden;}
.rc-bar div {height: 100%; background: var(--accent); border-radius: 99px;}
.rc-sub {font-size: .85rem; opacity: .7;}
footer {visibility: hidden;}
"""

HEADER_HTML = """
<div class="hero">
  <h1>🏦 Consumer Complaint Classifier</h1>
  <p>Paste a consumer complaint narrative and the model will route it to the most likely
     financial product category, with a full confidence breakdown.</p>
  <div class="chips">
    <span class="chip">💳 Credit Card</span>
    <span class="chip">📊 Credit Reporting</span>
    <span class="chip">📞 Debt Collection</span>
    <span class="chip">🏠 Mortgages and Loans</span>
    <span class="chip">🏦 Retail Banking</span>
  </div>
</div>
"""

theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="sky",
    neutral_hue="slate",
    font=[
        gr.themes.GoogleFont("Inter"),
        "ui-sans-serif",
        "system-ui",
        "sans-serif",
    ],
)

with gr.Blocks(
    theme=theme, css=CSS, title="Consumer Complaint Classifier"
) as demo:
  gr.HTML(HEADER_HTML)

  with gr.Row(equal_height=False):
    with gr.Column(scale=5):
      complaint_tb = gr.Textbox(
          label="Complaint narrative",
          placeholder="Paste or type the consumer complaint here…",
          lines=10,
          max_lines=20,
      )
      counter = gr.HTML(counter_html(""))
      with gr.Row():
        classify_btn = gr.Button(
            "Classify Complaint", variant="primary", scale=3
        )
        clear_btn = gr.Button("Clear", variant="secondary", scale=1)

    with gr.Column(scale=5):
      result_html = gr.HTML(empty_card())
      label_out = gr.Label(label="Confidence distribution", num_top_classes=5)
      latency_out = gr.HTML(latency_html(None))
      notice_md = gr.Markdown(_STARTUP_NOTICE)

      with gr.Accordion("Model Comparison & Project Benchmark", open=False):
        gr.Markdown(benchmark_markdown())

  outputs = [result_html, label_out, latency_out, notice_md]

  gr.Examples(
      examples=EXAMPLES,
      inputs=[complaint_tb],
      outputs=outputs,
      fn=classify,
      run_on_click=True,
      cache_examples=False,
      label="Try an example (one per category)",
  )

  complaint_tb.change(
      counter_html, complaint_tb, counter, show_progress="hidden"
  )
  classify_btn.click(classify, complaint_tb, outputs)
  complaint_tb.submit(classify, complaint_tb, outputs)
  clear_btn.click(clear_all, None, [complaint_tb, counter, *outputs])

if __name__ == "__main__":
  demo.queue().launch()