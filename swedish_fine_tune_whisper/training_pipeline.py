import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate 
from feature_pipeline import load_common_voice

LOAD_PRETRAINED = "common_voice/whisper-small-weights"
TRAINING_PARAMS = "cpu"
LOAD_DATASET_PATH = "common_voice"
SAVE_WEIGHTS = "common_voice/whisper-small-weights"

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
    

processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Swedish", task="transcribe")
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Swedish", task="transcribe")

metric = evaluate.load("wer")
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

def load_model(from_pretrained="openai/whisper-small", save_path=SAVE_WEIGHTS):
    """function that returns the model to be trained on

    Args:
        from_pretrained (str, optional): pretrained weights to use. Defaults to "openai/whisper-small".
        save_path (str, optional): path to save the weights so they don't need to be downloaded. If left none, they will not be saved. Defaults to None. 

    Returns:
        transformers.WhisperForConditionalGeneration: huggingface transformer model
    """

    model = WhisperForConditionalGeneration.from_pretrained(from_pretrained)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    
    if not from_pretrained.split("/")[0] == 'openai':
        print("Weights loaded from local source.")
        return model
    
    if save_path:
        print(f"Saving downloaded weights to {save_path}...")
        model.save_pretrained(save_path)
    
    return model

def load_training_args(params_key='training_config_05_12_22_v1'):
    """loads the training config

    Args:
        params_key (str, optional): key in the json config of parameters to use. Defaults to 'training_config_05_12_22_v1'.

    Returns:
        transformers.Seq2SeqTrainingArguments: training arguments
    """

    import json

    with open("training_config.json") as f:
        training_params = json.load(f)
        
    training_params = training_params[params_key]
        
    training_args = Seq2SeqTrainingArguments(
        num_train_epochs=training_params['num_train_epochs'],
        output_dir=training_params['output_dir'], 
        per_device_train_batch_size=training_params['per_device_train_batch_size'],
        gradient_accumulation_steps=training_params['gradient_accumulation_steps'], 
        learning_rate=training_params['learning_rate'],
        warmup_steps=training_params['warmup_steps'],
        max_steps=training_params['max_steps'],
        gradient_checkpointing=training_params['gradient_checkpointing'],
        fp16=training_params['fp16'],
        evaluation_strategy=training_params['evaluation_strategy'],
        per_device_eval_batch_size=training_params['per_device_eval_batch_size'],
        predict_with_generate=training_params['predict_with_generate'],
        generation_max_length=training_params['generation_max_length'],
        save_steps=training_params['save_steps'],
        eval_steps=training_params['eval_steps'], 
        logging_steps=training_params['logging_steps'],
        report_to=training_params['report_to'],
        load_best_model_at_end=training_params['load_best_model_at_end'],
        metric_for_best_model=training_params['metric_for_best_model'],
        greater_is_better=training_params['greater_is_better'],
        push_to_hub=training_params['push_to_hub'],
    )

    return training_args    

if __name__ == "__main__":
    print("Started training pipeline.")

    print(f"Loading model with pretrained {LOAD_PRETRAINED}...")
    model = load_model(LOAD_PRETRAINED)
    print("Model loaded.")
    
    print(f"Loading training params from the config file, {TRAINING_PARAMS}...")
    training_args = load_training_args(TRAINING_PARAMS)
    print("Training params loaded.")

    if not LOAD_DATASET_PATH:
        print("Creating and loading the common voice dataset...")
    else:
        print(f"Loading the common voice dataset from {LOAD_DATASET_PATH}...")
    common_voice = load_common_voice(path=LOAD_DATASET_PATH)
    print("Common voice loaded.")

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=common_voice["train"],
        eval_dataset=common_voice["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    processor.save_pretrained(training_args.output_dir)

    print("Training starting...")
    trainer.train()

    kwargs = {
        "dataset_tags": "mozilla-foundation/common_voice_11_0",
        "dataset": "Common Voice 11.0",  # a 'pretty' name for the training dataset
        "dataset_args": "config: sv, split: test",
        "language": "sv",
        "model_name": "Whisper Small Sv - Swedish",  # a 'pretty' name for our model
        "finetuned_from": "openai/whisper-small",
        "tasks": "automatic-speech-recognition",
        "tags": "hf-asr-leaderboard",
    }

    print("Pushing the model...")
    trainer.push_to_hub(**kwargs)
    print("Training pipeline finished")