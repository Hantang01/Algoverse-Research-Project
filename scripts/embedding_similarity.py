from functions import load_model
from generate_data import Zeropad_pair, Misleading_pair, baseline_pair
import torch
import torch.nn.functional as F

tokenizer, model = load_model()

embedding_matrix = model.get_input_embeddings().weight  

def get_embedding(text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    vectors = embedding_matrix[torch.tensor(ids)]
    return vectors.mean(dim=0)  

similarities = []

for i in range(10000):
    num1, num2, label = Misleading_pair()

    embed_num1 = get_embedding(num1)
    embed_num2   = get_embedding(num2)

    similarity = F.cosine_similarity(embed_num1.unsqueeze(0), embed_num2.unsqueeze(0))
    similarities.append(similarity.item())
    
    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1}/1000 pairs...")

similarities_tensor = torch.tensor(similarities)
average_similarity = similarities_tensor.mean().item()
std_similarity = similarities_tensor.std().item()
min_similarity = similarities_tensor.min().item()
max_similarity = similarities_tensor.max().item()

print(f"Average cosine similarity: {average_similarity:.4f}")
print(f"Standard deviation: {std_similarity:.4f}")
print(f"Minimum similarity: {min_similarity:.4f}")
print(f"Maximum similarity: {max_similarity:.4f}")


#graph X:difference
#Y: cosine similarity