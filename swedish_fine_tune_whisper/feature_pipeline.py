

from datasets import load_dataset, DatasetDict, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer
import sys
import huggingface_hub
import os
from utils import *


def load_common_voice(path=None, save_path=None):
    """function that loads or downloads and edits

    Args:
        path (string, optional): path to the dataset to load. Defaults to None. (directory)
        save_path (string, optional): path where to save the loaded/downloaded dataset. Defaults to None. (directory)

    Returns:
        datasets.DatasetDict: common voice dataset
    """
    
    # if the save path already exists, ask the user whether they want to overwrite it
    if save_path and os.path.exists(save_path):
        if not query_yes_no(f"{save_path} already exists and will be overwritten. Continue?"):
            return
        
    # if the save path is same to load path (and they exist), we may want to load it instead
    if path == save_path and path:
        if query_yes_no(f"{save_path} already exists. Do you want to load it instead?"):
            return DatasetDict.load_from_disk(save_path)
    
    print("Dataset loading started")
    
    if path:
        print(f"Loading dataset from {path}...")
        return DatasetDict.load_from_disk(path)

    print("Loading dataset from huggingface...")
    common_voice = DatasetDict()

    common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", "sv-SE", split="train+validation", use_auth_token=True)
    common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "sv-SE", split="test", use_auth_token=True)
    
    print("Raw dataset loaded.")

    common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])
    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
    
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Swedish", task="transcribe")
    
    def prepare_dataset(batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array 
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # encode target text to label ids 
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch

    print("Mapping the dataset...")
    common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=2)
    
    print("Dataset ready for training.")
    
    if SAVE_DATASET_PATH:
        print(f"Saving dataset to {save_path}...")
        common_voice.save_to_disk(save_path)
        
        return common_voice
        
        
if __name__ == "__main__":
    print("Feature pipeline started...")
    
    # path to read the common voice dataset from (directory)
    # if None, common voice will be downloaded from huggingface
    LOAD_DATASET_PATH = None

    # path to save the common voice dataset to after downloading (directory)
    # if None, it will not be saved
    SAVE_DATASET_PATH = "common_voice"

    with open("huggingface_token.txt", mode='r') as f:
        token = f.read()
        
    huggingface_hub.login(token=token, add_to_git_credential=True)
    
    if not LOAD_DATASET_PATH and not SAVE_DATASET_PATH:
        print("Error: no paths were specified to load nor save, so this code will have no effect. Aborting.")
        sys.exit()

    load_common_voice(path=LOAD_DATASET_PATH, save_path=SAVE_DATASET_PATH)