"""
Embedding Similarity Analysis for Numerical Comparison Tasks

This module analyzes cosine similarity between embeddings of mathematically 
equivalent numbers with different representations (e.g., 6.8 vs 6.800).
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import torch
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import os

def get_number_embedding(tokenizer, model, number_str):
    """
    Get the embedding representation for a single number.
    
    Args:
        tokenizer: HuggingFace tokenizer
        model: HuggingFace model
        number_str: String representation of the number
        
    Returns:
        numpy array: Embedding vector for the number
    """
    # Tokenize the number
    tokens = tokenizer.encode(number_str, add_special_tokens=False, return_tensors="pt")
    
    # Get embeddings from the model
    with torch.no_grad():
        # Get the last hidden states (embeddings)
        outputs = model(tokens, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # Last layer
        
        # Average pool across tokens for this number
        embedding = hidden_states.mean(dim=1).squeeze().cpu().numpy()
    
    return embedding

def compute_pair_similarity(tokenizer, model, num1, num2):
    """
    Compute cosine similarity between embeddings of two numbers.
    
    Args:
        tokenizer: HuggingFace tokenizer
        model: HuggingFace model
        num1, num2: String representations of numbers to compare
        
    Returns:
        dict: Analysis results including similarity score and metadata
    """
    # Get embeddings
    emb1 = get_number_embedding(tokenizer, model, num1)
    emb2 = get_number_embedding(tokenizer, model, num2)
    
    # Compute cosine similarity
    similarity = cosine_similarity([emb1], [emb2])[0][0]
    
    # Get tokenization info for context
    tokens1 = tokenizer.tokenize(num1)
    tokens2 = tokenizer.tokenize(num2)
    
    return {
        'num1': num1,
        'num2': num2,
        'cosine_similarity': float(similarity),
        'tokens1': tokens1,
        'tokens2': tokens2,
        'token_count1': len(tokens1),
        'token_count2': len(tokens2),
        'mathematically_equal': float(num1) == float(num2),
        'embedding_dim': len(emb1)
    }

def generate_zero_padding_pairs(n_pairs=20):
    """
    Generate pairs of mathematically equivalent numbers with different zero-padding.
    
    Args:
        n_pairs: Number of pairs to generate
        
    Returns:
        list: Pairs of (original, zero_padded) numbers
    """
    from generate_data import Zeropad_pair
    
    pairs = []
    for _ in range(n_pairs):
        # Generate a zero-padded pair
        a, b, label = Zeropad_pair()
        
        # Since Zeropad_pair returns mathematically equal numbers,
        # we know these should be equivalent
        pairs.append((a, b))
    
    return pairs

def generate_precision_variation_pairs(n_pairs=20):
    """
    Generate pairs with systematic precision variations.
    
    Args:
        n_pairs: Number of pairs to generate
        
    Returns:
        list: Pairs of numbers with different decimal precision
    """
    import random
    
    pairs = []
    for _ in range(n_pairs):
        # Generate base number
        base_int = random.randint(1, 100)
        decimal_part = random.randint(1, 9)
        
        # Create variations with different zero-padding
        base = f"{base_int}.{decimal_part}"
        padded = f"{base_int}.{decimal_part}" + "0" * random.randint(1, 3)
        
        pairs.append((base, padded))
    
    return pairs

def analyze_embedding_similarities(tokenizer, model, pairs, analysis_name="embedding_similarity"):
    """
    Analyze cosine similarities for a list of number pairs.
    
    Args:
        tokenizer: HuggingFace tokenizer
        model: HuggingFace model
        pairs: List of (num1, num2) tuples
        analysis_name: Name for this analysis
        
    Returns:
        dict: Complete analysis results
    """
    results = {
        'analysis_name': analysis_name,
        'timestamp': datetime.now().isoformat(),
        'total_pairs': len(pairs),
        'pair_analyses': [],
        'summary_statistics': {}
    }
    
    similarities = []
    
    print(f"Analyzing {len(pairs)} pairs...")
    for i, (num1, num2) in enumerate(pairs):
        if i % 5 == 0:
            print(f"  Processing pair {i+1}/{len(pairs)}")
        
        pair_result = compute_pair_similarity(tokenizer, model, num1, num2)
        results['pair_analyses'].append(pair_result)
        similarities.append(pair_result['cosine_similarity'])
    
    # Calculate summary statistics
    similarities = np.array(similarities)
    results['summary_statistics'] = {
        'mean_similarity': float(np.mean(similarities)),
        'std_similarity': float(np.std(similarities)),
        'min_similarity': float(np.min(similarities)),
        'max_similarity': float(np.max(similarities)),
        'median_similarity': float(np.median(similarities))
    }
    
    return results

def plot_similarity_analysis(results, save_path="outputs/similarity_plot.png"):
    """
    Create a line plot of cosine similarities across pairs.
    
    Args:
        results: Results from analyze_embedding_similarities
        save_path: Path to save the plot
    """
    # Extract data for plotting
    pair_indices = range(1, len(results['pair_analyses']) + 1)
    similarities = [pair['cosine_similarity'] for pair in results['pair_analyses']]
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Line plot of similarities
    plt.plot(pair_indices, similarities, 'b-o', linewidth=2, markersize=6, alpha=0.7)
    
    # Add statistical lines
    mean_sim = results['summary_statistics']['mean_similarity']
    plt.axhline(y=mean_sim, color='r', linestyle='--', alpha=0.7, 
                label=f'Mean: {mean_sim:.4f}')
    
    # Formatting
    plt.xlabel('Pair Index', fontsize=12)
    plt.ylabel('Cosine Similarity', fontsize=12)
    plt.title(f'Embedding Cosine Similarity: {results["analysis_name"]}\n'
              f'Mean: {mean_sim:.4f} ± {results["summary_statistics"]["std_similarity"]:.4f}', 
              fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add range information
    plt.ylim(min(similarities) - 0.01, max(similarities) + 0.01)
    
    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Similarity plot saved to {save_path}")
    
    plt.show()

def save_similarity_analysis(results, filename="outputs/embedding_similarity_analysis.json"):
    """Save similarity analysis results to JSON file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Similarity analysis saved to {filename}")

def create_detailed_comparison_table(results):
    """
    Create a detailed table showing individual pair comparisons.
    
    Args:
        results: Results from analyze_embedding_similarities
    """
    print("\n=== DETAILED PAIR COMPARISON ===")
    print(f"{'Pair':<4} {'Number 1':<12} {'Number 2':<12} {'Similarity':<12} {'Math Equal':<10} {'Tokens 1':<10} {'Tokens 2':<10}")
    print("-" * 80)
    
    for i, pair in enumerate(results['pair_analyses']):
        print(f"{i+1:<4} {pair['num1']:<12} {pair['num2']:<12} "
              f"{pair['cosine_similarity']:<12.6f} {pair['mathematically_equal']:<10} "
              f"{pair['token_count1']:<10} {pair['token_count2']:<10}")

def run_zero_padding_similarity_analysis(n_pairs=20):
    """
    Run complete embedding similarity analysis for zero-padded numbers.
    
    Args:
        n_pairs: Number of pairs to analyze
        
    Returns:
        dict: Analysis results
    """
    from functions import load_model
    
    print("Loading model and tokenizer...")
    tokenizer, model = load_model()
    
    print(f"Generating {n_pairs} zero-padded pairs...")
    pairs = generate_zero_padding_pairs(n_pairs)
    
    print("Analyzing embedding similarities...")
    results = analyze_embedding_similarities(
        tokenizer, model, pairs, 
        analysis_name="Zero-Padded Number Similarity"
    )
    
    # Save results
    save_similarity_analysis(results, "outputs/zero_padding_similarity.json")
    
    # Create visualizations
    plot_similarity_analysis(results, "outputs/zero_padding_similarity_plot.png")
    
    # Print summary
    print("\n=== SIMILARITY ANALYSIS SUMMARY ===")
    stats = results['summary_statistics']
    print(f"Analysis: {results['analysis_name']}")
    print(f"Total pairs analyzed: {results['total_pairs']}")
    print(f"Mean cosine similarity: {stats['mean_similarity']:.6f}")
    print(f"Standard deviation: {stats['std_similarity']:.6f}")
    print(f"Range: {stats['min_similarity']:.6f} - {stats['max_similarity']:.6f}")
    
    # Show detailed comparison
    create_detailed_comparison_table(results)
    
    return results

def run_precision_variation_analysis(n_pairs=20):
    """
    Run embedding similarity analysis for precision variations.
    
    Args:
        n_pairs: Number of pairs to analyze
        
    Returns:
        dict: Analysis results
    """
    from functions import load_model
    
    print("Loading model and tokenizer...")
    tokenizer, model = load_model()
    
    print(f"Generating {n_pairs} precision variation pairs...")
    pairs = generate_precision_variation_pairs(n_pairs)
    
    print("Analyzing embedding similarities...")
    results = analyze_embedding_similarities(
        tokenizer, model, pairs, 
        analysis_name="Precision Variation Similarity"
    )
    
    # Save results
    save_similarity_analysis(results, "outputs/precision_similarity.json")
    
    # Create visualizations
    plot_similarity_analysis(results, "outputs/precision_similarity_plot.png")
    
    # Print summary
    print("\n=== PRECISION VARIATION ANALYSIS ===")
    stats = results['summary_statistics']
    print(f"Mean cosine similarity: {stats['mean_similarity']:.6f}")
    print(f"Standard deviation: {stats['std_similarity']:.6f}")
    
    return results

if __name__ == "__main__":
    # Run zero-padding analysis
    print("=== ZERO-PADDING SIMILARITY ANALYSIS ===")
    zero_results = run_zero_padding_similarity_analysis(20)
    
    print("\n" + "="*60 + "\n")
    
    # Run precision variation analysis
    print("=== PRECISION VARIATION ANALYSIS ===")
    precision_results = run_precision_variation_analysis(20)