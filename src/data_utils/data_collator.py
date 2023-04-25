import itertools
import torch
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional
from ..configs.training_args import TrainingArguments
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase
from ..utils.notifier import logging, log_formatter
notifier = logging.getLogger(__name__)
notifier.addHandler(log_formatter())
                
InputDataClass = NewType("InputDataClass", Any)

"""
A DataCollator is a function that takes a list of samples from a Dataset and collate them into a batch, as a dictionary
of Tensors.
"""
DataCollator = NewType("DataCollator", Callable[[List[InputDataClass]], Dict[str, torch.Tensor]])

@dataclass
class PredDataCollator:
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """
    tokenizer: PreTrainedTokenizerBase
    args: TrainingArguments
    vocab_ignore_idx: Optional[List] = None
    prediction: bool = False

    def _pad(self, encoded_inputs: List[Dict], max_length, padding_side="right"):
        required_input = encoded_inputs["input_ids"]
        needs_to_be_padded = len(required_input) != max_length

        if needs_to_be_padded:
            difference = max_length - len(required_input)
            if padding_side == "right":
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] += [self.tokenizer.pad_token_type_id] * difference
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] += [1] * difference
                if "additional_pe" in encoded_inputs:
                    encoded_inputs["additional_pe"] += [0] * difference
                encoded_inputs["input_ids"] = required_input + [self.tokenizer.pad_token_id] * difference
                encoded_inputs["attention_mask"] = [1] * len(required_input) + [0] * difference

            elif padding_side == "left":
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = [self.tokenizer.pad_token_type_id] * difference + encoded_inputs["token_type_ids"]
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = [1] * difference + encoded_inputs["special_tokens_mask"]
                if "additional_pe" in encoded_inputs:
                    encoded_inputs["additional_pe"] = [0] * difference + encoded_inputs["additional_pe"]
                encoded_inputs["input_ids"] = [self.pad_token_id] * difference + required_input
                encoded_inputs["attention_mask"] = [0] * difference + [1] * len(required_input)

        return encoded_inputs


    def __call__(self,features: List[InputDataClass]) -> Dict[str, torch.Tensor]:
        # max_length = max([len(f.input_ids) for f in features])
        batch = {}
        for f in features:
            encoded_inputs = vars(f)
            # outputs = self._pad(
            #     encoded_inputs,
            #     512,
            # )
            # for key, value in outputs.items():
            for key, value in encoded_inputs.items():
                if key not in batch:
                    batch[key] = []
                batch[key].append(value)
        batch = {
            k: torch.tensor(list(itertools.chain(*batch[k]))) if (k=='labels') and (self.args.method=='clz') and (self.args.difficulty!="thesaurus2") else 
            torch.tensor(batch[k]) 
            for k in batch.keys() 
        }
        if self.args.method=='clz':
            batch['label_mask'] = (batch['input_ids']==self.tokenizer.mask_token_id)
            batch['vocab_mask'] = torch.zeros(len(self.tokenizer)).index_fill_(dim=0,index=torch.tensor(self.vocab_ignore_idx),value=-1e10)

        return batch
          
@dataclass
class QADataCollator:
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """
    tokenizer: PreTrainedTokenizerBase
    args: TrainingArguments
    prediction: bool = False

    def __call__(self,features: List[InputDataClass]) -> Dict[str, torch.Tensor]:
        batch = {}
        for f in features:
            for key, value in vars(f).items():
                if key not in batch:
                    batch[key] = []
                batch[key].append(value)
        tensorized_batch = {
            k:torch.tensor(v) for k, v in batch.items()
        }
        # notifier.warning(tensorized_batch)
        # notifier.warning(features[0])

        return tensorized_batch

# @dataclass
# class GenDataCollator:
#     """
#     Data collator used for language modeling.
#     - collates batches of tensors, honoring their tokenizer's pad_token
#     - preprocesses batches for masked language modeling
#     """
#     tokenizer: PreTrainedTokenizerBase
#     prediction: bool = False

#     def __call__(self,features: List[InputDataClass]) -> Dict[str, torch.Tensor]:
#         if not isinstance(features[0], (dict, BatchEncoding)):
#             enc_features = [f.input_ids for f in features]
#             dec_features = [f.decoder_input_ids for f in features]

#         batch = self.tokenizer.pad(
#             enc_features,
#             padding="longest",
#             max_length=512,
#             pad_to_multiple_of=8,
#             return_tensors="pt",
#         )

#         dec_batch =  self.tokenizer.pad(
#             dec_features,
#             padding="longest",
#             max_length=512,
#             pad_to_multiple_of=8,
#             return_tensors="pt",
#         )

#         for k in batch.copy():
#             batch[f"decoder_{k}"] = dec_batch[k]

#         # Label for generation
#         batch['labels'] = batch["decoder_input_ids"].clone().detach()
#         batch['labels'][batch['labels']==self.tokenizer.pad_token_id] = -100

#         # Causal attention mask
#         n_ctx = dec_batch["input_ids"].size(-1)
#         batch['decoder_attention_mask'] = torch.tril(torch.ones((n_ctx, n_ctx), dtype=torch.uint8)).view(1, n_ctx, n_ctx)

#         return batch