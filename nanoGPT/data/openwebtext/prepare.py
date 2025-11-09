# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets
import ipdb
st = ipdb.set_trace
# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    # takes 54GB in huggingface .cache dir, about 8M documents (8,013,769)
    # dataset = load_dataset("openwebtext", num_proc=num_proc_load_dataset)
    import pickle
    import os
    root_data_dir = os.environ["DATA_DIR"]
    dataset = pickle.load(open(os.path.join(root_data_dir, "stories_cache_500000.pkl"), "rb"))
    val_dataset = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
    # st()
    # Convert the list of stories to a dataset-like structure for processing
    train_stories = [{'text': story} for story in dataset]
    
    # Convert validation dataset to list
    val_stories = []
    for i, story in enumerate(val_dataset):
        if i >= 5000:  # Limit validation set size
            break
        val_stories.append({'text': story['text']})
    
    # Create split dataset structure
    split_dataset = {
        'train': train_stories,
        'val': val_stories
    }

    # we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
    def process(example):
        ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        out = {'ids': ids, 'len': len(ids)}
        return out

    # tokenize the dataset
    tokenized = {}
    for split_name, stories in split_dataset.items():
        print(f"Tokenizing {split_name} split with {len(stories)} stories...")
        tokenized_stories = []
        for story in tqdm(stories, desc=f"tokenizing {split_name}"):
            tokenized_story = process(story)
            tokenized_stories.append(tokenized_story)
        tokenized[split_name] = tokenized_stories
    os.environ["DATA_DIR"]
    root_data_dir = os.environ["DATA_DIR"]
    data_dir = os.path.join(root_data_dir, "nanoGPT-tinystories")
    os.makedirs(data_dir, exist_ok=True)
    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum([item['len'] for item in dset], dtype=np.uint64)
        print(f"{split} arr_len: {arr_len}")
        filename = os.path.join(data_dir, f'{split}.bin')
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        
        idx = 0
        for item in tqdm(dset, desc=f'writing {filename}'):
            arr_batch = np.array(item['ids'], dtype=dtype)
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

    # train.bin is ~17GB, val.bin ~8.5MB
    # train has ~9B tokens (9,035,582,198)
    # val has ~4M tokens (4,434,897)

    # to read the bin files later, e.g. with numpy:
    # m = np.memmap('train.bin', dtype=np.uint16, mode='r')
