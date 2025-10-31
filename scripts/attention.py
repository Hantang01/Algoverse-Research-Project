import sys, os
import torch
import circuitsvis as cv
from IPython.display import display

from functions import load_model_fp32, format_comparison_prompt, save_attention

from generate_data import Zeropad_pair, Misleading_pair, baseline_pair
torch.set_default_device("cuda")

tokenizer, model = load_model_fp32()
print(f"Number of attention layers: {model.config.num_hidden_layers}")
model.set_attn_implementation("eager")

a, b, label = baseline_pair()
prompt = format_comparison_prompt(a, b)

messages = [
    {"role": "system", "content": "You are a helpful assistant that compares numbers"},
    {"role": "user", "content": prompt}
]

formatted_prompt = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)

inputs = tokenizer(formatted_prompt, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}


with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)

layer_idx = 1
for layer_idx in range(model.config.num_hidden_layers):
    attention_pattern = outputs.attentions[layer_idx][0].detach().cpu().numpy()  # (H, T, T)

    tokens_full = tokenizer.convert_ids_to_tokens(
        inputs["input_ids"][0].tolist(),
        skip_special_tokens=False
    )


    num_heads = attention_pattern.shape[0]
    head_names = [f"L{layer_idx}H{i}" for i in range(num_heads)]

    def pretty(tok: str) -> str:
        # common whitespace markers 
        t = tok.replace("Ġ", " ")   
        t = t.replace("▁", " ")     
        t = t.replace("Ċ", "\\n")   
        t = t.replace("<0x0A>", "\\n")  
        if t.strip() == "":
            return "·"              
        return t


    tokens_display = [pretty(t) for t in tokens_full]




    viz = cv.attention.attention_patterns(
            tokens=tokens_display,
            attention=attention_pattern,
        )

    html = viz._repr_html_() if hasattr(viz, "_repr_html_") else str(viz)
    save_attention(html, "attention_analysis_1", layer_idx)

    print("Saved visualization layer ", layer_idx)

