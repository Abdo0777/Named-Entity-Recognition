import json, torch, gradio as gr
from transformers import AutoTokenizer, AutoModelForTokenClassification

MODEL_DIR = "distilbert-ner"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
bert_model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR).to(device)
with open(f"{MODEL_DIR}/id2label.json") as f:
    id2label = {int(k): v for k, v in json.load(f).items()}

LABEL_COLORS = {"PER": "#ff6b6b", "ORG": "#4ecdc4", "LOC": "#ffd93d", "MISC": "#a78bfa"}
LABEL_NAMES = {"PER": "Person", "ORG": "Organization", "LOC": "Location", "MISC": "Miscellaneous"}

def predict_entities(text):
    if not text.strip():
        return [], "Enter some text above to see detected entities."
    tokens = text.split()
    inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        logits = bert_model(**inputs).logits
    preds = logits.argmax(-1)[0].cpu().numpy()

    word_ids = inputs.word_ids(batch_index=0)
    word_preds = {}
    for idx, word_id in enumerate(word_ids):
        if word_id is not None and word_id not in word_preds:
            word_preds[word_id] = id2label[preds[idx]]

    highlighted, entity_count = [], {"PER": 0, "ORG": 0, "LOC": 0, "MISC": 0}
    for i, tok in enumerate(tokens):
        tag = word_preds.get(i, "O")
        etype = tag[2:] if tag != "O" else None
        if etype:
            entity_count[etype] += 1
        highlighted.append((tok, etype))

    summary = "  ".join(f"{LABEL_NAMES[k]}: {v}" for k, v in entity_count.items() if v > 0) or "No entities detected."
    return highlighted, summary

CUSTOM_CSS = """
.gradio-container { max-width: 900px !important; margin: auto; }
.legend-badge { display: inline-block; padding: 4px 12px; border-radius: 16px; font-size: 13px; font-weight: 600; margin: 0 6px 6px 0; color: #1a1a1a; }
"""
legend_html = "".join(f'<span class="legend-badge" style="background:{c}">{LABEL_NAMES[t]}</span>' for t, c in LABEL_COLORS.items())

with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Soft(primary_hue="teal")) as demo:
    gr.Markdown("# Named Entity Recognition")
    gr.Markdown("Fine-tuned DistilBERT · trained on CoNLL-2003 · live detection")
    gr.HTML(f'<div style="text-align:center; margin-bottom:20px">{legend_html}</div>')
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(label="Input text", lines=4,
                placeholder="e.g. Elon Musk visited Berlin last July with a Tesla team.")
            gr.Examples(examples=[
                "Barack Obama met Angela Merkel in Berlin on Monday.",
                "Apple Inc. announced record profits in California.",
                "The United Nations held a summit in Geneva last March.",
            ], inputs=text_input)
        with gr.Column():
            output = gr.HighlightedText(label="Detected entities (live)", color_map=LABEL_COLORS)
            summary_box = gr.Textbox(label="Summary", interactive=False)

    text_input.change(fn=predict_entities, inputs=text_input, outputs=[output, summary_box], show_progress="hidden")

demo.launch()