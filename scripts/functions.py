
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch 
import csv
import os
from datetime import datetime

def format_comparison_prompt(num1, num2):
    return f"""Which is bigger?

                Options:
                    1. {num1}
                    2. {num2}
                    3. Both are equal

                Answer with **only** the option number (1, 2, or 3). Do not write anything else."""


def load_model():
    model_name = "./my_llama_model"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'left' 
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    return tokenizer, model

def ask_model_batch(prompts, tokenizer, model, system_prompt):

    """
    Generates model responses for a batch of input prompts.

    arguments:
        prompts (List[str]): A list of user input strings (questions or prompts) to query the model.

    returns:
        A list of decoded model responses, in the same order as the input prompts.

    Notes:
        MUST USE do_sample=False

    """


    messages_batch = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        messages_batch.append(messages)

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    input_ids = tokenizer.apply_chat_template(
        messages_batch,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
    
    responses = []
    for i in range(outputs.shape[0]):
        generated_tokens = outputs[i][input_ids.shape[-1]:]
        reply = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        responses.append(reply)
    
    return responses
def prompt_accuracy_with_batching(
        batch_size,
        comparisons,
        tokenizer,
        model,
        system_prompt
    ):

    """Evaluate model accuracy on the provided comparisons list using batching.

    Args:
        batch_size (int): number of examples per batch
        comparisons (List[tuple]): list of (a, b, label) tuples
        
    Returns:
        float: accuracy as a decimal (0.0 to 1.0)
    """

    total_cases = len(comparisons)
    wrong = 0

    for batch_start in range(0, total_cases, batch_size):
        batch_end = min(batch_start + batch_size, total_cases)
        batch_cases = comparisons[batch_start:batch_end]

        # Progress indicator every 5 batches
        if batch_start % (batch_size * 5) == 0:
            print(f"Processing cases {batch_start + 1}-{batch_end} ({(batch_end/total_cases)*100:.1f}%)")

        batch_prompts = [format_comparison_prompt(a, b) for a, b, label in batch_cases]

        batch_responses = ask_model_batch(batch_prompts, tokenizer, model, system_prompt)

        for i, (a, b, expected_label) in enumerate(batch_cases):
            response = batch_responses[i]
            # responses are strings like '1' or '2' or '3', expected_label may be str or int
            if str(response).strip() != str(expected_label).strip():
                wrong += 1
    
    accuracy = (total_cases - wrong) / total_cases
    print(f"Testing completed. Total cases: {total_cases}, Wrong predictions: {wrong}, Accuracy: {accuracy:.2%}")
    return accuracy


def load_data(file_path):


    """
    Loads comparison data from a CSV file.

    Arguments:
    file_path -- path to the CSV file containing comparison data

    Returns:
    A list of tuples, each containing two numbers as strings and the correct answer
    (num1, num2, answer).

    """

    comparison_cases = []
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            comparison_cases.append((row[0], row[1], row[2]))  

    return comparison_cases
def report_result(model_name, tested_method, data_size, accuracy):
    """
    Reports test results to a CSV file in the outputs directory.
    
    Args:
        model_name (str): Name of the model tested
        tested_method (str): Method tested (e.g., 'Zeropad_pair', 'Misleading_pair')
        accuracy (float): Accuracy as a decimal (0.0 to 1.0)
    """
    
    # Ensure outputs directory exists
    os.makedirs('outputs', exist_ok=True)
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Check if results file exists, if not create with headers
    results_file = 'outputs/results.csv'
    file_exists = os.path.exists(results_file)
    
    with open(results_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header if file is new
        if not file_exists:
            writer.writerow(['model_used', 'method_tested','data_size', 'accuracy', 'time_tested'])
        
        # Write the result
        writer.writerow([model_name, tested_method, data_size, f"{accuracy:.4f}", timestamp])
    
    print(f"Results saved to {results_file}")

if __name__ == "__main__":
    tokenizer, model = load_model()
