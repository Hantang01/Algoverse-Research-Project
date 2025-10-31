import sys, os
import torch
import numpy as np
from functions import load_model_fp32, format_comparison_prompt
from generate_data import Zeropad_pair, Misleading_pair, baseline_pair

torch.set_default_device("cuda")

tokenizer, model = load_model_fp32()
print(f"Number of attention layers: {model.config.num_hidden_layers}")
model.set_attn_implementation("eager")

a, b, label = baseline_pair()
print(f"\nAnalyzing: {a} vs {b}")
print("-" * 60)

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

# Get tokens
tokens_full = tokenizer.convert_ids_to_tokens(
    inputs["input_ids"][0].tolist(),
    skip_special_tokens=False
)

def pretty(tok: str) -> str:
    t = tok.replace("Ġ", " ")   
    t = t.replace("▁", " ")     
    t = t.replace("Ċ", "\\n")   
    t = t.replace("<0x0A>", "\\n")  
    if t.strip() == "":
        return "·"              
    return t

tokens_display = [pretty(t) for t in tokens_full]



a_indices = []
b_indices = []

a_str = str(a)
b_str = str(b)

for i, token in enumerate(tokens_display):
    token_clean = token.strip()
    if a_str in token_clean or token_clean in a_str:
        a_indices.append(i)
    if b_str in token_clean or token_clean in b_str:
        b_indices.append(i)

print(f"\nNumber A ({a}) found at token position: {a_indices}")
print(f"Number B ({b}) found at token position: {b_indices}")


attention_scores = []

for layer_idx in range(model.config.num_hidden_layers):
    attention_pattern = outputs.attentions[layer_idx][0].detach().cpu().numpy()  # (H, T, T)
    num_heads = attention_pattern.shape[0]
    
    for head_idx in range(num_heads):
        for b_pos in b_indices:
            for a_pos in a_indices:

                attention_score = attention_pattern[head_idx, b_pos, a_pos]
                
                attention_scores.append({
                    'layer': layer_idx,
                    'head': head_idx,
                    'b_pos': b_pos,
                    'a_pos': a_pos,
                    'b_token': tokens_display[b_pos],
                    'a_token': tokens_display[a_pos],
                    'attention': attention_score
                })

attention_scores.sort(key=lambda x: x['attention'], reverse=True)

print("\n" + "=" * 80)
print("top 20 head_layer of B attending to A")
print("=" * 80)


for rank, score in enumerate(attention_scores[:20], 1):
    b_to_a = f"[{score['b_pos']}]→[{score['a_pos']}]"
    details = f"'{score['b_token']}'→'{score['a_token']}'"
    print(f"{rank:<6} L{score['layer']:<6} H{score['head']:<5} {b_to_a:<15} {score['attention']:<12.6f} {details}")


