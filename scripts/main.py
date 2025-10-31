from generate_data import write_file, Zeropad_pair, Misleading_pair, baseline_pair
from functions import load_model, format_prompt, ask_model_batch, load_data, prompt_accuracy_with_batching, report_result
import torch


data_size = 100
system_prompt = "You are a helpful assistant that compares numbers"

if __name__ == "__main__":
    torch.set_default_device("cuda")
    print("Loading model..")
    tokenizer, model = load_model()
    print("Model loaded.")

    print(model)

    # for i in range(1):


    #     path = write_file(Misleading_pair, data_size)
    #     print(f"Generated 1000 Misleading_pair data at: {path}")
    #     comparisons = load_data(path)
    #     print("Data loaded")

    #     accuracy = prompt_accuracy_with_batching(20, comparisons, tokenizer, model, system_prompt)
        
    #     model_name = "my_llama_model"  
    #     method_name = Misleading_pair.__name__
    #     report_result(model_name, method_name, data_size, accuracy)
        
    # print("done")