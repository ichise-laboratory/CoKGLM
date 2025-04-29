#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    # AutoModelForSeq2SeqLM,
    AutoTokenizer,
    # DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
    HfArgumentParser,
    # Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
    EarlyStoppingCallback,
    TrainerCallback,
    get_linear_schedule_with_warmup,
    AdamW
)

from trainer_seq2seq_baseline import Seq2SeqTrainer
from modeling_bart_baseline import BartForConditionalGeneration
from prefix_modeling_bart import PrefixBartForConditionalGeneration


from utils import *
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
import ast
import pickle
import torch
import wandb
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, train_test_split
from datasets import DatasetDict

torch.autograd.set_detect_anomaly(True)
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.7.0")

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    mode: str = field(
        metadata={"help": "the mode to merge knowlege"}
    )
    memory_bank_mode: str = field(
        metadata={"help": "the mode to merge memory_bank"}
    )
    max_memory_bank: int = field(
        metadata={"help": "the maximum number of retrieved triples in memory bank"}
    )
    
    use_memory_bank: bool = field(
        metadata={"help": "Whether to use memory bank."},
    )
    use_kg_embedding: bool = field(
        metadata={"help": "Whether to use memory bank."},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
        # default: 入力がない時に使う値
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    early_stop: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to do early stopping in the traning process."}
    )
    early_stopping_patience: Optional[int] = field(
        default=1,
        metadata={"help": "`metric_for_best_model` to stop training when the specified metric worsens for `early_stopping_patience` evaluation calls."}
    )
    ####################prefix####################
    prefix_tuning: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to do P-tuning v2 in the traning process."}
    )
    freeze_model: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to freeze model"}
    )
    pre_seq_len: int = field(
        default=4,
        metadata={
            "help": "The length of prompt"
        }
    )
    prefix_projection: bool = field(
        default=False,
        metadata={
            "help": "Apply a two-layer MLP head over the prefix embeddings"
        }
    ) 
    prefix_hidden_size: int = field(
        default=512,
        metadata={
            "help": "only used when prefix_projection is True. The hidden size of the MLP projection head in Prefix Encoder if prefix projection is used"
        }
    )
    prefix_hidden_dropout_prob: float = field(
        default=0.1,
        metadata={
            "help": "The dropout probability used in the models"
        }
    )



@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    entity_relation_embedding_path: str = field(default=None)
    
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    #要約するコラム：会話履歴
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge) on "
            "(a jsonlines or csv file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on " "(a jsonlines or csv file)."
        },
    )
    
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    do_sample: Optional[bool] = field(default=None,)
    top_k: Optional[int] = field(default=None,)
    top_p: Optional[float] = field(default=None,)
    generation_file: Optional[str] = field(default=None)
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    
    report_to_wandb: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the WandB project to log training runs (if using WandB)."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1] 
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length

@dataclass
class CustomTrainingArguments(Seq2SeqTrainingArguments):
    warmup_ratio: Optional[float] = field(
        default=0.3,
        metadata={
            "help": "The ratio of total training steps to use for warmup steps. Default is 0.3."
        },
    )

# map text and summary column names for different dataset
summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
}


def main():
    
    os.environ["WANDB_PROJECT"]="hallucination-detection"
    os.environ["WANDB_LOG_MODEL"]="false"
    os.environ["WANDB_WATCH"]="false"
    
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"): 
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)], 
    )
    logger.setLevel(logging.INFO if training_args.should_log else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    logger.info(f"Training/evaluation parameters {training_args}")

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
            

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the full texts and the second column for the
    # summaries (unless you specify column names for this with the `text_column` and `summary_column` arguments).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    
    entity_relation_embedding = np.array(pickle.load(open(data_args.entity_relation_embedding_path, "rb")))
    original_graph_emb_dim = entity_relation_embedding.shape[1]
    entity_relation_embedding = np.array(list(np.zeros((1, original_graph_emb_dim))) + list(entity_relation_embedding))
    
                                
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
        extension = data_args.train_file.split(".")[-1]
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
        extension = data_args.validation_file.split(".")[-1]
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
        extension = data_args.test_file.split(".")[-1]
    datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    #tokenizer.add_tokens(['<sep>','<triple>','<user>','<assistant>','<response>'])
    tokenizer.add_tokens(['<sep>','<triple>','<user>','<assistant>','<response>','<ent>','</ent>']) # for LLM only detection
    def model_init_function():
        config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        )
    
        assert model_args.mode in ['input', "last_one"]
        config.update({'mode': model_args.mode})
        
        assert model_args.memory_bank_mode in ['all', "entity"]
        config.update({'memory_bank_mode': model_args.memory_bank_mode})
        config.update({'use_memory_bank': model_args.use_memory_bank})
        config.update({'use_kg_embedding': model_args.use_kg_embedding})
        
        
        config.update({'prefix_tuning': model_args.prefix_tuning})
        config.update({'freeze_model': model_args.freeze_model})
        config.update({'pre_seq_len': model_args.pre_seq_len})
        config.update({'prefix_projection': model_args.prefix_projection})
        config.update({'prefix_hidden_size': model_args.prefix_hidden_size})
        config.update({'prefix_hidden_dropout_prob': model_args.prefix_hidden_dropout_prob})
        
        if model_args.prefix_tuning:
            model = PrefixBartForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                entity_relation_weight=entity_relation_embedding,
            )
        else:
            model = BartForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                entity_relation_weight=entity_relation_embedding,
            )
              
        model.resize_token_embeddings(len(tokenizer))
        #assert model.vocab_size == model.config.vocab_size ==50270
        assert model.vocab_size == model.config.vocab_size ==50272 # for LLM only detection

        
        """
        if model.config.decoder_start_token_id is None:
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
        """
        
        return model
    
    # Load the model with the highest f1 score in the first cross-validation.
    def load_best_model():
        config = AutoConfig.from_pretrained(best_model_save_path)
        
        if model_args.prefix_tuning:
            model_class = PrefixBartForConditionalGeneration
        else:
            model_class = BartForConditionalGeneration
        
        model = model_class.from_pretrained(
            best_model_save_path,
            from_tf=bool(".ckpt" in best_model_save_path),
            config=config,
            entity_relation_weight=entity_relation_embedding if entity_relation_embedding is not None else None,
        )
              
        model.resize_token_embeddings(len(tokenizer))
        #assert model.vocab_size == model.config.vocab_size ==50270
        assert model.vocab_size == model.config.vocab_size ==50272 # for LLM only detection
        return model

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = datasets["train"].column_names 
    elif training_args.do_eval:
        column_names = datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`.")
        return
    
    column_names.remove('memory_bank')
    
    # Get the column names for input/target.
    
    text_column = data_args.text_column
    if text_column not in column_names:
        raise ValueError(
            f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
        )
        
    summary_column = data_args.summary_column
    if summary_column not in column_names:
        raise ValueError(
            f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
        )

    
    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False
    """
    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )
    """    

    def preprocess_function(examples):
        
        inputs = examples[text_column]
        targets = examples[summary_column]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)
        targets = [int(l) for l in targets]
        model_inputs["labels"] = targets      
        
        model_inputs['input_entity_relation_ids'] = []
        for entity_relation_ids in examples['entity_relation_ids']:
            entity_relation_ids = ast.literal_eval(entity_relation_ids)[:data_args.max_source_length]
            entity_relation_ids += (data_args.max_source_length-len(entity_relation_ids))*[0]
            model_inputs['input_entity_relation_ids'].append(entity_relation_ids)

        model_inputs['memory_bank'] = []
        model_inputs['memory_bank_attention_mask'] = []
        
        max_memory_bank = model_args.max_memory_bank  # need to be adjusted based on the number of retrieved triples
    
    
        for memory_bank_str in examples['memory_bank']:
            memory_bank = ast.literal_eval(memory_bank_str)

            valid_triples = [triple for triple in memory_bank if len(triple) == 3]
            num_triples = len(valid_triples) 

            if num_triples <= max_memory_bank:
                padding_needed = max_memory_bank - num_triples
                memory_bank.extend(padding_needed * [[0, 0, 0]])
                attention_mask = [1] * num_triples + [0] * padding_needed
                model_inputs['memory_bank'].append(memory_bank)
                model_inputs['memory_bank_attention_mask'].append(attention_mask)    
            
            else:
                assert False

        return model_inputs

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits[0], axis=-1)
        print("=============predictions===============")
        for i in predictions:
            print(i)
        print("=============labels===============")
        for i in labels:
            print(i)
        
        conf_matrix = confusion_matrix(labels, predictions)
        tn, fp, fn, tp = conf_matrix.ravel()
        
        precision = precision_score(labels, predictions)
        accuracy = accuracy_score(labels, predictions)
        recall = recall_score(labels, predictions)
        f1 = f1_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'TP': tp,
            'FP': fp,
            'TN': tn,
            'FN': fn,
        }
    

    
    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        
        # for weight&bias
        project_name = "hallucination-detection"
        group_name = "extract_5_perentity"  # group name for wandb
        model_folder = "extract_5_perentity"  # folder name for test data and best model
        
        # initialize GroupKFold
        outer_gkf = GroupKFold(n_splits=5)
        inner_gkf = GroupKFold(n_splits=5)
        
        # Split based on 'data_id' column
        groups = datasets["train"]['data_id']
        
        test_accuracy_results = []
        test_precision_results = []
        test_recall_results = []
        test_f1_results = []
        
        
        for split_idx, (train_val_idx, test_idx) in enumerate(outer_gkf.split(X=datasets["train"], groups=groups)):
            test_dataset = datasets["train"].select(test_idx)    
            test_dataset = test_dataset.shuffle(seed=52)
            
            # Save the split test data
            df_test = test_dataset.to_pandas()
            file_path = f'./nest_cross_val/{model_folder}/test_{split_idx + 1}.csv'
    
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok=True)
            df_test.to_csv(file_path, index=False)
            
            test_dataset = test_dataset.map(preprocess_function, batched=True, num_proc=data_args.preprocessing_num_workers, remove_columns=column_names, load_from_cache_file=not data_args.overwrite_cache)
            
            # the secondary cross-validation (further split into training and validation data to select the model parameters with best performance)
            train_val_dataset = datasets["train"].select(train_val_idx)
            train_val_groups = train_val_dataset['data_id'] # Split based on 'data_id' column
        
        
            eval_accuracy_results = []
            eval_precision_results = []
            eval_recall_results = []
            eval_f1_results = []
            
            # to select the best model 
            best_score = 0
            best_fold = -1
            best_model_save_path = None
            
            # start the secondary cross-validation
            for fold, (train_idx, val_idx) in enumerate(inner_gkf.split(X=np.zeros(len(train_val_groups)), groups=train_val_groups)):
                # Start a new wandb run
                run = wandb.init(project=project_name, group=group_name, name=f"split_{split_idx + 1}_fold_{fold + 1}", reinit=True)
                run_id = run.id
                
                # training data and validation data for the current fold
                fold_train_dataset = train_val_dataset.select(train_idx)
                fold_val_dataset = train_val_dataset.select(val_idx)
                fold_train_dataset = fold_train_dataset.shuffle(seed=52)
                fold_val_dataset = fold_val_dataset.shuffle(seed=52)
                
                fold_train_dataset = fold_train_dataset.map(preprocess_function, batched=True, num_proc=data_args.preprocessing_num_workers, remove_columns=column_names, load_from_cache_file=not data_args.overwrite_cache)
                fold_val_dataset = fold_val_dataset.map(preprocess_function, batched=True, num_proc=data_args.preprocessing_num_workers, remove_columns=column_names, load_from_cache_file=not data_args.overwrite_cache)
                
                # Data collator
                #label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
                
                data_collator = DataCollatorWithPadding(
                    tokenizer,
                    pad_to_multiple_of=8 if training_args.fp16 else None,
                )
                
                # settings for warm-up learning rate
                model = model_init_function()
                optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)
                train_dataset_size = len(fold_train_dataset)
                steps_per_epoch = train_dataset_size // training_args.per_device_train_batch_size // training_args.gradient_accumulation_steps
                total_steps = steps_per_epoch * training_args.num_train_epochs
                warmup_steps = int(training_args.warmup_ratio * total_steps) # set the warm-up steps as a percentage of the total training steps
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_steps
                )

                # Initialize early stop callbacks
                callbacks = []
                if model_args.early_stop:
                    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=model_args.early_stopping_patience)
                    callbacks.append(early_stopping_callback)
                
                # Initialize Trainer
                trainer = Seq2SeqTrainer(
                    model=model,
                    #model_init=model_init_function,  # for the experiments not using warm-up learning rate
                    args=training_args,
                    train_dataset=fold_train_dataset if training_args.do_train else None,
                    eval_dataset=fold_val_dataset,
                    compute_metrics=compute_metrics, 
                    tokenizer=tokenizer,
                    data_collator=data_collator,
                    callbacks=callbacks,
                    optimizers=(optimizer, scheduler), # comment out for the experiments not using warm-up learning rate
                )
                
                # training
                train_result = trainer.train()

                metrics = train_result.metrics
                max_train_samples = (
                    data_args.max_train_samples if data_args.max_train_samples is not None else len(fold_train_dataset)
                )
                metrics["train_samples"] = min(max_train_samples, len(fold_train_dataset))
                trainer.log_metrics("train", metrics)
                trainer.save_metrics("train", metrics)

                logger.info("*** Validation ***")
                max_target_length = data_args.val_max_target_length
                eval_result = trainer.evaluate(metric_key_prefix="eval") # add the prefix "eval" to the metrics 
                max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(fold_val_dataset)
                eval_result["eval_samples"] = min(max_eval_samples, len(fold_val_dataset))
                trainer.log_metrics("eval", eval_result)
                trainer.save_metrics("eval", eval_result)
                
                logger.info(f"split_idx: {split_idx + 1} Eval_metrics Fold {fold + 1} - Accuracy: {eval_result['eval_accuracy']}, Precision: {eval_result['eval_precision']}, Recall: {eval_result['eval_recall']}, F1: {eval_result['eval_f1']}")
                wandb.log({"eval_metrics": eval_result}, step=fold)
                
                # check if the current model is the best model based on f1 score
                current_score = eval_result['eval_f1']
                if current_score > best_score:
                    best_score = current_score
                    best_fold = fold + 1
                    best_run_id = run_id
                    # Save the best model
                    best_model_save_path = os.path.join(training_args.output_dir, model_folder, f"split_{split_idx  + 1}_fold_{best_fold}")
                    trainer.save_model(best_model_save_path)  
                
                eval_accuracy_results.append(eval_result['eval_accuracy'])
                eval_precision_results.append(eval_result['eval_precision'])
                eval_recall_results.append(eval_result['eval_recall'])
                eval_f1_results.append(eval_result['eval_f1'])

                run.finish()
                
            # calculate the average performance of the current split
            eval_average_accuracy = sum(eval_accuracy_results) / len(eval_accuracy_results)
            eval_average_precision = sum(eval_precision_results) / len(eval_precision_results)
            eval_average_recall = sum(eval_recall_results) / len(eval_recall_results)
            eval_average_f1 = sum(eval_f1_results) / len(eval_f1_results)
            
            logger.info(f"split_idx: {split_idx + 1}  Best fold is {best_fold} with best F1 score: {best_score}  Average Eval Performance  Accuracy: {eval_average_accuracy}, Precision: {eval_average_precision}, Recall: {eval_average_recall}, F1: {eval_average_f1}")
            if best_model_save_path:
                print("best_model_save_path:", best_model_save_path)
            else:
                print("No best model was saved.")
            print("best_model_save_path:", best_model_save_path)
            
            wandb.init(id=best_run_id, resume="allow")
            # load the best model
            trainer = Seq2SeqTrainer(
                model_init=load_best_model,  
                args=training_args,
                compute_metrics=compute_metrics,
                tokenizer=tokenizer,
                data_collator=data_collator,
            )
            
            logger.info("*** Test ***")
            # Test the best model
            test_results = trainer.predict(
                test_dataset,
                metric_key_prefix="test",
            )
            metrics = test_results.metrics
            max_test_samples = (
                data_args.max_predict_samples if data_args.max_predict_samples is not None else len(test_dataset)
            )
            metrics["test_samples"] = min(max_test_samples, len(test_dataset))
            trainer.log_metrics("test", metrics)
            trainer.save_metrics("test", metrics)
            logger.info(f"----Test_metrics---- split_idx: {split_idx + 1} Fold: {best_fold}: {test_results.metrics}")
            
            wandb.log({"test_metrics": test_results.metrics})
            
            # save the test results for the current split
            test_accuracy_results.append(test_results.metrics['test_accuracy'])
            test_precision_results.append(test_results.metrics['test_precision'])
            test_recall_results.append(test_results.metrics['test_recall'])
            test_f1_results.append(test_results.metrics['test_f1'])
            wandb.finish()
        
        # calculate the average performance of the test results
        test_average_accuracy = sum(test_accuracy_results) / len(test_accuracy_results)
        test_average_precision = sum(test_precision_results) / len(test_precision_results)
        test_average_recall = sum(test_recall_results) / len(test_recall_results)
        test_average_f1 = sum(test_f1_results) / len(test_f1_results)
        
        logger.info(f"Average Test Performance  Accuracy: {test_average_accuracy}, Precision: {test_average_precision}, Recall: {test_average_recall}, F1: {test_average_f1}")
    

    if training_args.push_to_hub: # push the trained model to the Hugging Face Model Hub
        kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "classification"} # description, a dictionary of keyword arguments that will be passed to the push_to_hub method
        if data_args.dataset_name is not None:
            kwargs["dataset_tags"] = data_args.dataset_name
            if data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = data_args.dataset_config_name
                kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
            else:
                kwargs["dataset"] = data_args.dataset_name

        trainer.push_to_hub(**kwargs)
    
    return


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == "__main__":
    main()
