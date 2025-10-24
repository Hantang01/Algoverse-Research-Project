import sys, os
import torch
import circuitsvis as cv
from IPython.display import display

from functions import load_model, format_comparison_prompt

from generate_data import Zeropad_pair, Misleading_pair, baseline_pair
torch.set_default_device("cuda")

tokenizer, model = load_model()



model.set_attn_implementation("eager")


if getattr(model, "config", None) and getattr(model.config, "attn_implementation", None) not in ("eager","eager_paged","flex_attention"):
    from transformers import AutoModelForCausalLM
    model_name_or_path = getattr(model, "name_or_path", None)
    if model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            attn_implementation="eager",        
        )


a, b, label = Misleading_pair()
prompt = format_comparison_prompt(a, b)

inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)

layer_idx = 0
attention_pattern = outputs.attentions[layer_idx][0].detach().cpu().numpy()  # (H, T, T)

tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist(), skip_special_tokens=True)
print(tokens)
num_heads = attention_pattern.shape[0]
head_names = [f"L{layer_idx}H{i}" for i in range(num_heads)]

viz = cv.attention.attention_patterns(
        tokens=tokens,
        attention=attention_pattern,
    )

html = viz._repr_html_() if hasattr(viz, "_repr_html_") else str(viz)
with open("attention_viz.html", "w", encoding="utf-8") as f:
    f.write(html)

print("Saved visualization")
