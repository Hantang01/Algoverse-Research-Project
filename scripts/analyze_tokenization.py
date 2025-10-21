from generate_data import generate_int_with_decimals
from functions import load_model
import matplotlib.pyplot as plt
import numpy as np

tokenizer, model = load_model()

text_lengths = []
token_counts = []

data = [generate_int_with_decimals(1, 9, 20) for _ in range(100)]

for text in data:
    tokens = tokenizer.tokenize(text)
    number_of_tokens = len(tokens)
    length = len(text)
    
    text_lengths.append(length)
    token_counts.append(number_of_tokens)
    
    print(f"Text: {text} | Text length {length} | Number of tokens: {number_of_tokens} | Tokens: {tokens}")

plt.figure(figsize=(10, 6))
plt.scatter(text_lengths, token_counts, alpha=0.6, s=50)
plt.xlabel('Text Length (characters)', fontsize=12)
plt.ylabel('Number of Tokens', fontsize=12)
plt.title('Text Length vs Number of Tokens for Generated Numbers', fontsize=14)
plt.grid(True, alpha=0.3)

z = np.polyfit(text_lengths, token_counts, 1)
p = np.poly1d(z)
plt.plot(text_lengths, p(text_lengths), "r--", alpha=0.8, label=f'Trend line (slope: {z[0]:.2f})')
plt.legend()

correlation = np.corrcoef(text_lengths, token_counts)[0, 1]
plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=plt.gca().transAxes, 
         bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('outputs/tokenization_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("graph saved to outputs/tokenization_analysis.png")