import random
from decimal import Decimal, getcontext
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import csv
import os

def Zeropad_pair(low_bound=0, high_bound=20, swap_probability=0.5):

    """
    Generates integes with zero padded decimal pairs for comparison.
    example: 5.7 and 5.700

    Arguments: 
    low_bound -- the lower bound for the random integers 
    high_bound -- the upper bound for the random integers 
    swap_probability -- the probability of swapping the order of the pair
    
    Returns
    a -- first number as string
    b -- second number as string
    label -- 3 (1 if a > b, 2 if b > a, 3 if equal)
    
        """

    i = random.randint(low_bound, high_bound)
    decimal = random.randint(0,9)
    pad_len = random.randint(1, 3)

    a = str(i)+"."+str(decimal)
    b = str(i)+"."+str(decimal)+"0"*pad_len

    pair = [a, b]
    if random.random() < swap_probability:
        random.shuffle(pair)

    return str(pair[0]), str(pair[1]), 3

def Misleading_pair(low_bound=1, high_bound=20, swap_probability=0.5):

    """
    Generates misleading pairs of decimal numbers for comparison.
    example: 5.7 and 5.12

    Arguments: 
    low_bound -- the lower bound for the random integers 
    high_bound -- the upper bound for the random integers 
    swap_probability -- the probability of swapping the order of the pair
    
    Returns
    a -- first number as string
    b -- second number as string
    label -- which number is larger (1 if a > b, 2 if b > a, 3 if equal)
    
    """

    i = random.randint(low_bound, high_bound)

    d1 = Decimal(str(random.uniform(0.6, 0.9))).quantize(Decimal("0.0"))   # one decimal place
    d2 = Decimal(str(random.uniform(0.11, 0.13))).quantize(Decimal("0.00")) # two decimal places

    a = Decimal(i) + d1
    b = Decimal(i) + d2

    pair = [a, b]
    if random.random() < swap_probability:
        random.shuffle(pair)

    if pair[0] > pair[1]:
        label = 1

    elif pair[1] > pair[0]:
        label = 2

    else:
        label = 3

    return str(pair[0]), str(pair[1]), label


def baseline_pair(low_bound=0, high_bound=999, swap_probability=0.5):

    """
    Generates base line pairs of integers for comparison.    

    Arguments: 
    low_bound -- the lower bound for the random integers 
    high_bound -- the upper bound for the random integers 
    swap_probability -- the probability of swapping the order of the pair
    
    Returns
    a -- first integer as string
    b -- second integer as string
    label -- which number is larger (1 if a > b, 2 if b > a, 3 if equal)
    
    """

    i = random.randint(low_bound, high_bound)
    j = random.randint(low_bound, high_bound)
    pair = [i, j]
    if random.random() < swap_probability:
        random.shuffle(pair)

    if pair[0] > pair[1]:
        label = 1

    elif pair[1] > pair[0]:
        label = 2
    else:
        label = 3
    return str(pair[0]), str(pair[1]), label

def write_file(method, size=1000):

    """
    Writes a genereted dataset to a CSV file.
    
    Arguments:
    method -- the data generation method to use(Zero_pad_pair, Misleading_pair, base_line_pair)
    size -- number of data points to generate
    
    Returns:
    The path to the generated CSV file.
    
    """

    existing_files = [f for f in os.listdir("src/datasets/") if f.endswith(".csv")]

    # Filter files by the specific method name
    method_files = [f for f in existing_files if method.__name__ in f]
    
    indexs = [0]
    
    for i in method_files:
        # Split by "_" and take the last part before ".csv", then remove ".csv"
        index_part = i.split("_")[-1].replace(".csv", "")
        indexs.append(int(index_part))

    max_index = int(max(indexs)) + 1


    with open(f'src/datasets/{method.__name__}_{max_index}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['a', 'b', 'answer']) 
        for _ in range(size):
            a, b, label = method()
            writer.writerow([a, b, label])
    return f"src/datasets/{method.__name__}_{max_index}.csv"


