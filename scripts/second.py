import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# SETTING UP THE ENVIRONMENT
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import pandas as pd
import time
import argparse

try:
    import accelerate
except ImportError:
    print("Accelerate is not installed.")

script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the tokenizer and model from the local directory
gemma_path = "/home/ltnghia02/models/gemma_model"

tokenizer = AutoTokenizer.from_pretrained(gemma_path)
model = AutoModelForCausalLM.from_pretrained(
    gemma_path,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# UTILITIES
import chardet

def read_txt_file(file_path: str) -> str:
    """
    Reads the content of a text file with automatic encoding detection.
    """
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']

    with open(file_path, 'r', encoding=encoding) as file:
        return file.read().strip()

def write_file(file_path, content):
    """
    Writes content to a text file with UTF-8 encoding.
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

def generate_text(input_text, max_new_tokens=5000):
    """
    Generates text using the loaded model given an input string.
    """
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

    outputs = model.generate(
        input_ids=input_ids['input_ids'],
        max_new_tokens=max_new_tokens,
        do_sample=True
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    start_index = len(input_text)
    return generated_text[start_index:].strip()

def process_step_2(caption: str, prompt: str):
    """
    Processes step 2: generates questions based on the given caption and prompt.
    """
    detailed_prompt = prompt.format(caption=caption)
    generated_text = generate_text(detailed_prompt)

    return detailed_prompt, generated_text

def process_step_3(caption: str, questions: str, knowledge: str, prompt: str):
    """
    Processes step 3: generates answers based on the given caption, questions, knowledge, and prompt.
    """
    detailed_prompt = prompt.format(caption=caption, questions=questions, knowledge=knowledge)
    generated_text = generate_text(detailed_prompt)

    return detailed_prompt, generated_text

def process_step_4(knowledge: str, answers: str, caption: str, prompt: str):
    """
    Processes step 4: generates labels based on the given knowledge, answers, caption, and prompt.
    """
    detailed_prompt = prompt.format(knowledge=knowledge, answers=answers, caption=caption)
    generated_text = generate_text(detailed_prompt)

    return detailed_prompt, generated_text

# PROCESS LABELING

def read_prompt(prompt_path: str):
    """
    Reads and returns a list of prompts from the given path.
    """
    prompt_files = [
        "prompt_step_1.txt",
        "prompt_step_2.txt",
        "prompt_step_3.txt",
        "prompt_step_4.txt"
    ]

    # Read all prompt files
    prompts = []
    for prompt_file in prompt_files:
        prompt_file_path = os.path.join(prompt_path, prompt_file)
        prompt_content = read_txt_file(prompt_file_path)
        prompts.append(prompt_content)

    return prompts

def process_all_steps(dataset_path, prompt_path, img_index, begin_index=None, end_index=None):
    """
    Processes all steps for the given dataset and prompts.
    Filters directories by indices and applies all processing steps.
    Only processes subdirectories ending with .img_index.
    """
    csv_path = os.path.join(dataset_path, "dataset.csv")
    df = pd.read_csv(csv_path)

    prompts = read_prompt(prompt_path)
    normalized_dataset_path = os.path.normpath(dataset_path)
    base_depth = normalized_dataset_path.count(os.sep)

    for root, dirs, files in os.walk(dataset_path):
        normalized_root = os.path.normpath(root)
        current_depth = normalized_root.count(os.sep)

        if current_depth == base_depth:
            # Sort directories numerically and filter based on begin_index and end_index
            dirs.sort(key=lambda d: int(d) if d.isdigit() else d)

            if begin_index is not None and end_index is not None:
                dirs[:] = [d for d in dirs if d.isdigit() and begin_index <= int(d) <= end_index]

        elif current_depth == base_depth + 1:
            # Sort sub-subfolders numerically and filter by img_index
            dirs.sort(key=lambda d: [int(part) if part.isdigit() else part for part in d.split('.')])
            dirs[:] = [d for d in dirs if d.endswith(f".{img_index}")]

            for subdir in dirs:
                subdir_path = os.path.join(root, subdir)
                print(f"Processing {subdir_path}", flush=True)

                # Paths to the input files
                image_path = os.path.join(subdir_path, "image.jpg")
                short_path = os.path.join(subdir_path, "short.txt")
                caption_path = os.path.join(subdir_path, "molmo_dense_cap.txt")

                if not os.path.exists(image_path):
                    print(f"Image file missing: {image_path}", flush=True)
                    continue
                if not os.path.exists(short_path):
                    print(f"Short file missing: {short_path}", flush=True)
                    continue
                if not os.path.exists(caption_path):
                    print(f"Caption file missing: {caption_path}", flush=True)
                    continue

                # Read caption directly from the file
                caption = read_txt_file(caption_path)
                                                                                                                                                                                                                  
                # Remove the first line from the caption
                # caption = "\n".join(caption.splitlines()[1:])

                # Get knowledge from the dataset CSV
                subfolder_name = os.path.basename(root)
                df['index'] = df['index'].astype(str)
                subfolder_name = str(subfolder_name)
                matched_rows = df[df['index'] == subfolder_name]
                if not matched_rows.empty:
                    knowledge = matched_rows['text'].values[0]
                else:
                    print(f"No article text found for index {subfolder_name}")
                    knowledge = ""

                start_time = time.time()

                # Process all steps
                all_steps = ""

                # Step 2: Questioning
                prompt, question = process_step_2(caption, prompts[1])
                all_steps += "STEP 2: QUESTIONING\n\n"
                # all_steps += f"- Prompt:\n {prompt}\n\n"
                all_steps += f"- Response: \n{question}\n\n"

                # Step 3: Answering
                prompt, answer = process_step_3(caption, question, knowledge, prompts[2])
                all_steps += "STEP 3: ANSWERING\n\n"
                # all_steps += f"- Prompt: \n {prompt}\n\n"
                all_steps += f"- Response: \n{answer}\n\n"

                # Step 4: Labeling
                prompt, label = process_step_4(knowledge, answer, caption, prompts[3])
                all_steps += "STEP 4: LABELING\n\n"
                # all_steps += f"- Prompt: \n{prompt}\n\n"
                all_steps += f"- Response: \n{label}\n\n"

                end_time = time.time()
                elapsed_time = end_time - start_time

                # Add elapsed time to output
                all_steps += f"TIME: {elapsed_time:.2f}s\n"

                # Save results
                label_path = os.path.join(subdir_path, "caption.txt")
                write_file(label_path, label)

                all_steps_path = os.path.join(subdir_path, "all_steps.txt")
                write_file(all_steps_path, all_steps)
                print(f"DONE: {elapsed_time:.2f}s")

if __name__ == "__main__":
    print("Starting...")
    parser = argparse.ArgumentParser(description="Process all steps for the dataset.")
    parser.add_argument('dataset_path', type=str, help='The path to the dataset directory')
    parser.add_argument('prompt_path', type=str, help='The path to the prompt file')
    parser.add_argument('--begin_index', type=int, help='The starting index of directories to process', default=None)
    parser.add_argument('--end_index', type=int, help='The ending index of directories to process', default=None)
    parser.add_argument('--img_index', type=int, help='The ending index of image to process', default=None)

    args = parser.parse_args()
    print(f"Index: b = {args.begin_index}, e = {args.end_index}, i = {args.img_index}")

    print("Processing dataset...")
    process_all_steps(args.dataset_path, args.prompt_path, args.img_index, args.begin_index, args.end_index)


