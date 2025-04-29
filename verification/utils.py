from torch.utils.data.dataloader import DataLoader
from transformers import DataCollatorForSeq2Seq
from dataclasses import dataclass
from transformers import (
    PreTrainedTokenizerBase,
    PreTrainedModel,
)
from transformers.file_utils import PaddingStrategy
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import torch


@dataclass
class MYDataCollatorForSeq2Seq:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        features2 = []
        memory_bank = []
        memory_bank_attention_mask = []
        for feature in features:
            memory_bank.append(feature['memory_bank'])
            memory_bank_attention_mask.append(feature['memory_bank_attention_mask'])
            feature.pop('memory_bank')
            feature.pop('memory_bank_attention_mask')
            features2.append(feature)
        
        memory_bank = torch.LongTensor(memory_bank)
        memory_bank_attention_mask = torch.LongTensor(memory_bank_attention_mask)
        
        features = self.tokenizer.pad(
            features2,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        features['memory_bank'] = memory_bank
        features['memory_bank_attention_mask'] = memory_bank_attention_mask

        return features
    