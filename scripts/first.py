# Standard libraries
import os
import time
import logging
import traceback
import argparse
from pathlib import Path
from typing import List, Optional, Union, Mapping
from concurrent.futures import ThreadPoolExecutor

# Third-party libraries
import torch
import torchvision.transforms as T
import numpy as np
import pandas as pd
import cv2
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    GenerationConfig,
    AutoTokenizer,
)
from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ImageInput,
    is_valid_image,
)

# For file encoding detection (if used later)
import chardet


def set_visible_gpus(gpu_ids):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    print(f"Demand GPUs: {gpu_ids}")

    if torch.cuda.is_available():
        print(f"CUDA available, using GPU(s): {gpu_ids}")
    else:
        print("CUDA not available. Running on CPU.")


molmo_processor = None
molmo_model = None
log_success_files_path = '/home/ltnghia02/vischronos/logs/success_files.txt'

def setup(): 
    global molmo_processor, molmo_model

    logging.basicConfig(filename='/home/ltnghia02/vischronos/logs/script_execution1.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("Loading processor and model")
    try:
        custom_model_path = "/home/ltnghia02/models/molmo_model/models--allenai--Molmo-72B-0924/snapshots/2ca845922396b7a5f7086bfda3fca6b8ecd1c8f3"
        molmo_processor = AutoProcessor.from_pretrained(
            custom_model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"  # Chỉ định 4 GPU cho processor (nếu cần)
        )

        # Load model với việc phân phối trên 4 GPU
        molmo_model = AutoModelForCausalLM.from_pretrained(
            custom_model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto" # Chỉ định 4 GPU cho model
        )
    except Exception as e:
        logging.error(f"Error loading processor or model: {str(e)}")
        print(traceback.format_exc())
        raise


# %%
# Custom Dataset
class ImageTextDataset(Dataset):
    def __init__(self, root_dir, df, bidx: int, eidx: int, iidx: int):
        self.root_dir = root_dir
        self.df = df

        root_p = Path(root_dir)
        self.samples = sorted(
            (
                str(file)
                for folder in root_p.iterdir()
                if folder.is_dir() and folder.name.isdigit() and bidx <= int(folder.name) <= eidx
                for subfolder in folder.iterdir()
                if subfolder.is_dir() and subfolder.name.endswith(f'.{iidx}')
                for file in subfolder.glob("image.jpg")
            ),
            key=lambda x: float(Path(x).parts[-2])
        )

        self.target_size = (512, 512)

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize(self.target_size),
            T.ToTensor(),  # Converts to tensor and scales [0, 1]
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Example normalization
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            img_path = self.samples[idx]
            data_path = Path(img_path).parent
            # print(data_path)
            short_caption = open(data_path / 'short.txt').read()
            if (len(short_caption) == 0):
                short_caption = "."
            article = self.df[self.df['index'] == int(data_path.parent.stem)]['text'].values[0]
            image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            image_tensor = (self.transform(image)).permute(1, 2, 0)

            return image_tensor, short_caption, article, str(img_path)
        except Exception as e:
            logging.error(f"Error in __getitem__ for index {idx}: {str(e)}")
            print(f"Skipping sample {idx} due to error.")
            return None  # Return None to skip this sample


def collate_fn(batch):
    try:
        # Remove None values from the batch
        batch = [item for item in batch if item is not None]

        # Now process the batch
        images = [item[0] for item in batch]
        short_captions = [item[1] for item in batch]
        articles = [item[2] for item in batch]
        img_paths = [item[3] for item in batch]

        return images, short_captions, articles, img_paths
    except Exception as e:
        logging.error(f"Error in collate_fn: {str(e)}")
        print(traceback.format_exc())
        raise

def read_txt_file(file_path: str) -> str:
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']

    with open(file_path, 'r', encoding=encoding) as file:
        return file.read().strip()

def read_prompt_from_file(prompt_path: str):
    # Ensure prompt_path ends with a separator
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

# %%
def get_prompt(step, short_captions, dense_captions, questions, articles, questions_and_answers, template_prompt):
    try:
        if step == 1:  # Captioning
            prompts_list = [template_prompt[0].format(short_caption) for short_caption in short_captions]
        elif step == 2:  # Questioning
            prompts_list = [template_prompt[1].format(dense_caption) for dense_caption in dense_captions]
        elif step == 3:  # Answering
            prompts_list = [template_prompt[2].format(dense_caption, question, article) for dense_caption, question, article in
                            zip(dense_captions, questions, articles)]
        elif step == 4:  # Labeling
            prompts_list = [template_prompt[3].format(article, questions_and_answers_subset, dense_caption) for
                            article, questions_and_answers_subset, dense_caption in
                            zip(articles, questions_and_answers, dense_captions)]
        return prompts_list
    except Exception as e:
        logging.error(f"Error in get_prompt for step {step}: {str(e)}")
        print(traceback.format_exc())
        raise


# %%
# VLM pipeline
@torch.inference_mode()
def process_batch(
        batch,
        step,
        dense_captions=None,
        questions=None,
        questions_and_answers=None,
):
    try:
        images, short_captions, articles, img_paths = batch

        if None in images or None in short_captions or None in articles:
            print("Skipping batch due to errors in sample data.")
            return [], []

        prompts_list = []
        print(img_paths)
        with open(log_success_files_path, 'a') as log_success_files:
            log_success_files.write(f"Currently processing: {str(img_paths)}\n")
        template_prompt = read_prompt_from_file(args.prompt_path)
        prompts_list = get_prompt(step, short_captions, dense_captions, questions, articles, questions_and_answers,
                                  template_prompt)
        generated_texts = []

        input_list = [
            molmo_processor.process(images=[image.numpy()], text=prompt) for image, prompt in zip(images, prompts_list)
        ]

        inputs = {}

        padding_token_id = molmo_processor.tokenizer.pad_token_id

        # Extract input_ids from the input_list
        input_ids_list = [item['input_ids'] for item in input_list]

        # Find the maximum length of all sequences
        max_len = max(len(ids) for ids in input_ids_list)

        # Pre-pad each sequence (in the front) to the maximum length
        padded_input_ids = [
            torch.cat([torch.full((max_len - len(ids),), padding_token_id, dtype=torch.int64), ids])
            for ids in input_ids_list
        ]

        # Stack the padded sequences into a batch tensor
        padded_input_ids = torch.stack(padded_input_ids)

        # Send the padded input IDs to the model's device
        inputs = {}
        inputs['input_ids'] = padded_input_ids.to(molmo_model.device)

        for k in input_list[0].keys():

            if (k in inputs):
                continue
            inputs[k] = torch.stack([input[k] for input in input_list]).to(molmo_model.device)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = molmo_model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=1024, stop_strings="<|endoftext|>"),
                tokenizer=molmo_processor.tokenizer
            )

            for i in range(outputs.size(0)):
                generated_tokens = outputs[i, inputs['input_ids'].size(1):]
                generated_text = molmo_processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                generated_texts.append(generated_text)

        return generated_texts, img_paths

    except Exception as e:
        logging.error(f"Error in process_batch for step {step}: {str(e)}")
        print(f"Skipping sample due to error: {e}")
        return [], []  # Return empty lists to skip this sample



# %%
def write_results(response, data_path, step):
    try:
        parent_path = Path(data_path).parent
        if step == 1:
            print("STEP 1: CAPTIONING")
            with open(parent_path / 'molmo_dense_cap.txt', 'w') as f:
                f.write(response)
        elif step == 2:
            print("STEP 2: QUESTIONING")
            with open(parent_path / 'molmo_questions.txt', 'w') as f:
                f.write(response)
        elif step == 3:
            print("STEP 3: ANSWERING")
            with open(parent_path / 'molmo_answers.txt', 'w') as f:
                f.write(response)
        elif step == 4:
            print("STEP 4: LABELLING")
            with open(parent_path / 'molmo_caption.txt', 'w') as f:
                f.write(response)
        else:
            logging.error(f"Invalid step number: {step}")

        print("DONE")
    except Exception as e:
        logging.error(f"Error in write_results for step {step}, data_path {data_path}: {str(e)}")
        print(traceback.format_exc())

def write_results_batch(responses, data_paths, step):
    try:
        for response, data_path in zip(responses, data_paths):
            write_results(response, data_path, step)
    except Exception as e:
        logging.error(f"Error in write_results_batch: {str(e)}")
        print(f"Skipping batch due to error: {e}")

def process_dataset(dataset):
    num_workers = 4
    dataloader = DataLoader(dataset, batch_size=1, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)

    for batch in dataloader:
        try:
            dense_captions, img_paths = process_batch(batch, 1)
            if dense_captions:  # Only write results if the batch processed successfully
                write_results_batch(dense_captions, img_paths, 1)
            else:
                print(f"Skipping batch due to previous error.")
        except Exception as e:
            logging.error(f"Error processing batch: {str(e)}")
            print(traceback.format_exc())

# %%
import os

if __name__ == '__main__':
    try:
        print("Starting...")

        parser = argparse.ArgumentParser(description="Process all steps for the dataset.")
        parser.add_argument('dataset_path', type=str, help='The path to the dataset directory')
        parser.add_argument('prompt_path', type=str, help='The path to the prompt file')
        parser.add_argument('--begin_index', type=int, help='The starting index of directories to process',default=None)
        parser.add_argument('--end_index', type=int, help='The ending index of directories to process', default=None)
        parser.add_argument('--img_index', type=int, help='Index of image in an article', default=None)
        parser.add_argument('--gpu', type=str, default="0", help="Comma-separated list of GPU indices to be used (e.g., '0,1,2').")
        args = parser.parse_args()

        print(f"Index: b = {args.begin_index}, e = {args.end_index}, i = {args.img_index}")
        print("Preprocessing...")
        
        set_visible_gpus(args.gpu)
        setup()

        root_dir = args.dataset_path
        df = pd.read_csv(os.path.join(root_dir, "dataset.csv"))
        dataset = ImageTextDataset(root_dir, df, args.begin_index, args.end_index, args.img_index)

        print("Processing dataset...")
        start_time = time.time()
        process_dataset(dataset)
        end_time = time.time()
        print(f"Total time taken: {end_time - start_time} seconds")
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        print(traceback.format_exc())
        raise

