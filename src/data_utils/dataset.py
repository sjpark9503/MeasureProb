import json
import gc
import logging
import torch
from itertools import chain
from numpy.random import choice
from typing import List
from torch.utils.data.dataset import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from ..configs.parameters import DataTrainingArguments
from ..configs.training_args import TrainingArguments
from ..utils.notifier import logging, log_formatter
notifier = logging.getLogger(__name__)
notifier.addHandler(log_formatter())
              
"""
Define Dataset & Load
"""
class InputFeatures:
    """
    A single set of features of data. Property names are the same names as the corresponding inputs to a model.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``: Usually ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded)
            tokens.
        token_type_ids: (Optional) Segment token indices to indicate first and second
            portions of the inputs. Only some models use them.
        label: (Optional) Label corresponding to the input. Int for classification problems,
            float for regression problems.
    """
    def __init__(self, input_ids, **kwargs):
        self.input_ids = input_ids
        for k, v in kwargs.items():
            self.__setattr__(k,v)

class PredDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, args: TrainingArguments, accelerator=None):
        self.tokenizer = tokenizer
        self.args = args
        self.pe_type = args.pe_type
        self.features = list()
        self.accelerator = accelerator
        # notifier.info("Creating features from dataset file at %s", file_path)
        if accelerator.is_main_process:
            notifier.critical("Creating features from dataset file at %s", file_path)
        self.batch_encoding = torch.load(file_path)
        if "BioNumBERT" in self.args.model_name_or_path:
            if accelerator.is_main_process:
                notifier.critical("pre-processing input for BioNumBERT")
            self.batch_encoding['numbers'] = list()
            self.num_token_id = tokenizer("[NUM]", add_special_tokens=False)['input_ids'][0]
            for idx in range(len(self.batch_encoding['input'])):
                num_offset = sorted(self.batch_encoding['number_offset'][idx], key=lambda x: x[0])
                num_count = 0
                FLAG_DETECT_NUM = False 
                converted_sentence = ""
                numbers = list()
                for char_i, char in enumerate(self.batch_encoding['input'][idx]):
                    if num_count == len(num_offset):
                        converted_sentence += char
                    elif (char_i>=num_offset[num_count][0]) and (char_i<num_offset[num_count][1]):
                        if FLAG_DETECT_NUM:
                            number += char
                            continue 
                        else: 
                            FLAG_DETECT_NUM = True
                            number = ""
                            number += char
                            converted_sentence += "[NUM]"
                    else: 
                        if FLAG_DETECT_NUM:
                            FLAG_DETECT_NUM = False
                            num_count += 1
                            numbers.append(float(number))
                        converted_sentence += char

                self.batch_encoding['numbers'].append(numbers)
                self.batch_encoding['input'][idx] = converted_sentence
        elif "NumBERT" in self.args.model_name_or_path:
            if accelerator.is_main_process:
                notifier.critical("pre-processing input for NumBERT")
            for idx in range(len(self.batch_encoding['input'])):
                self.batch_encoding['input'][idx] = self.batch_encoding['input'][idx].replace("e-","scinotexp-").replace("e+","scinotexp+")
        self.batch_encoding['input'] = tokenizer(self.batch_encoding['input'], add_special_tokens=True, padding='longest', return_attention_mask=True, return_token_type_ids=True, return_offsets_mapping=True)

        if accelerator.is_main_process:
            self.padded_length = len(self.batch_encoding['input'][0])
            notifier.critical(f"Maximum length of sequence : {self.padded_length}")
        self.target_vocab = list(set(self.batch_encoding['label']))

        if "thesaurus" in self.args.difficulty:
            self.target_vocab = list(chain(*[self.paraphrase(x, "thesaurus", return_entire_candids=True) for x in set(self.batch_encoding['label'])]))
            if "train" in file_path:
                self.batch_encoding['label'] = [self.paraphrase(x, "thesaurus", return_entire_candids=self.args.difficulty=="thesaurus2") for x in self.batch_encoding['label']]
        
        if "train" not in file_path:
            self.actual_vocab = list(set(self.batch_encoding['label']))

        self.batch2feature()

        del self.batch_encoding
        gc.collect()
  
    def paraphrase(self, text, method="neural", return_entire_candids=False):
        if method == "thesaurus":
            thesaurus = {
                "larger":["higher","bigger","larger"],
                "smaller":["lower","less","smaller"],
                "largest":["biggest","maximum","largest"],
                "smallest":["lowest","minimum","smallest"],
                "middle":["medium","intermediate","middle"],
                "random":["random"],
                "increasing":["growing","ascending","increasing"],
                "decreasing":["descending","reducing","decreasing"],
                "same":["identical", "equal", "same"],
                "different":["distinct", "different"],
                "normal":["regular", "safe", "normal"],
                "abnormal":["irregular", "lethal", "abnormal"],
            }
            if return_entire_candids:
                return thesaurus[text]
            else:
                return choice(thesaurus[text])
        elif method == "neural":
            raise NotImplementedError()
        else:
            raise NotImplementedError(f"Method [{method}] not implemented yet.")

    def batch2feature(self):
        for idx in range(len(self.batch_encoding['label'])):
        # for idx in range(256):
            inputs = {k:self.batch_encoding['input'][k][idx] for k in self.batch_encoding['input'].keys() if "offset" not in k}
            if self.args.method == "clz":
                inputs['labels'] = self.tokenizer(self.batch_encoding['label'][idx], add_special_tokens=False)['input_ids']
            else:
                inputs['labels'] = int(self.batch_encoding['label'][idx])
            if ("number_offset" in self.batch_encoding) and (self.pe_type !=0):
                inputs['additional_pe'] = self._create_add_pe(self.batch_encoding['number_offset'][idx], self.batch_encoding['input']['offset_mapping'][idx])
            if "BioNumBERT" in self.args.model_name_or_path:
                inputs['md_num_encoding'] = list()
                count = 0   
                for x in inputs['input_ids']:
                    if x == self.num_token_id:
                        inputs['md_num_encoding'].append(self.batch_encoding['numbers'][idx][count])
                        count += 1
                    else:
                        inputs['md_num_encoding'].append(0)
            feature = InputFeatures(**inputs)
            self.features.append(feature)
            if (self.accelerator.is_main_process) and (idx==0):
                for k, v in inputs.items():
                    notifier.warning(f"{k}:")
                    notifier.warning(v)
         
    def _create_add_pe(self, number_offset, token_offset):
        token_offset.reverse()
        add_pe = list()
        if self.pe_type == 1:
            for pos in token_offset:
                if pos == (0, 0):
                    pe_idx = 0
                    add_pe.insert(0,pe_idx)
                    continue
                FLAG = False
                # check pos is contained in number
                for num_pos in number_offset:
                    if (num_pos[0]<=pos[0]) and (num_pos[1]>=pos[1]):
                        FLAG = True
                        break
                if FLAG:
                    pe_idx+=1
                else:
                    pe_idx=0
                add_pe.insert(0,pe_idx)
        else:
            raise NotImplementedError("Other type of positional encoding is not supported.")

        return add_pe

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

class ExtractiveQADataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, args: TrainingArguments, accelerator=None):
        self.tokenizer = tokenizer
        self.args = args
        self.pe_type = args.pe_type
        self.training = "train" in file_path
        self.accelerator = accelerator
        self.features = list()
        if accelerator.is_main_process:
            notifier.critical("Creating features from dataset file at %s", file_path)
        dataset = torch.load(file_path)

        self.batch2feature(dataset)

        # notifier.warning(vars(self.features[0]))
 
        del dataset
        gc.collect()
            
    def batch2feature(self, dataset):
        self.answers = dict()
        for idx, sample in enumerate(dataset):
        # for sample in dataset[:256]:
            inputs = self.tokenizer(
                sample["question"],
                sample["evidence"],
                truncation="only_second",
                max_length=512,
                return_attention_mask=True,
                return_token_type_ids=True,
                return_offsets_mapping=True,
                padding="max_length",
            )
            offsets = [x for x,y in zip(inputs['offset_mapping'], inputs.sequence_ids()) if y == 1]
            sentence_start_pos = len([y for y in inputs.sequence_ids() if y == 0])+2 
 
            in_span = False
            for _idx, offset in enumerate(offsets):
                if (offset[0]==sample['answer_ind'][0]) and not in_span:
                    in_span=True
                    start_pos = sentence_start_pos+_idx
                    if offset[1]==sample['answer_ind'][1]:
                        end_pos = sentence_start_pos+_idx
                        break
                elif (offset[1]==sample['answer_ind'][1]) and in_span:
                    in_span=False
                    end_pos = sentence_start_pos+_idx
                    break
                else:
                    continue
            try:
                inputs["start_positions"] = start_pos
                inputs["end_positions"] = end_pos
            except:
                continue
            if ("number_offset" in sample) and (self.pe_type !=0):
                add_pe = self._create_add_pe(sample['number_offset'], offsets)
                inputs['additional_pe'] = [0,]*len(inputs['input_ids'])
                inputs['additional_pe'][sentence_start_pos:sentence_start_pos+len(add_pe)]=add_pe
                assert len(inputs['additional_pe']) == len(inputs['input_ids'])
            if not self.training:
                self.answers[idx] = {
                    "context":sample["evidence"],  
                    "offset_mapping":inputs["offset_mapping"],
                    "answer":sample["evidence"][sample['answer_ind'][0]:sample['answer_ind'][1]]
                }
                inputs["id"] = idx 
            inputs.pop("offset_mapping") 
            feature = InputFeatures(**inputs)
            if (self.accelerator.is_main_process) and (idx==0):
                for k, v in inputs.items():
                    notifier.warning(f"{k}:")
                    notifier.warning(v)
            self.features.append(feature)

    def _create_add_pe(self, number_offset, token_offset):
        token_offset.reverse()
        add_pe = list()
        if self.pe_type == 1:
            pe_idx = 0
            for pos in token_offset:
                if pos == (0, 0):
                    add_pe.insert(0,pe_idx)
                    continue
                FLAG = False
                # check pos is contained in number
                for num_pos in number_offset:
                    if (num_pos[0]<=pos[0]) and (num_pos[1]>=pos[1]):
                        FLAG = True
                        break
                if FLAG:
                    pe_idx+=1
                else:
                    pe_idx=0
                add_pe.insert(0,pe_idx)
        else:
            raise NotImplementedError("Other type of positional encoding is not supported.")

        return add_pe


    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

def get_dataset(
    args: DataTrainingArguments,
    training_args: TrainingArguments,
    tokenizer: PreTrainedTokenizer,
    accelerator = None,
    splits: List[str] = ["train","valid","test"]
):
    def _dataset(file_path):
        if training_args.task == "emrQA":
            return ExtractiveQADataset(tokenizer=tokenizer, file_path=file_path, args=training_args, accelerator=accelerator)
        else:
            if "[PH]" not in file_path:
                return {"Inter":PredDataset(tokenizer=tokenizer, file_path=file_path, args=training_args, accelerator=accelerator)}
            else:
                return {PH:PredDataset(tokenizer=tokenizer, file_path=file_path.replace("[PH]",PH), args=training_args, accelerator=accelerator) for PH in ['Inter', 'Xtra']}

    target_vocab = list()
    actual_eval_vocab = list()
    train_datasets = _dataset(args.train_data_file) if "train" in splits else None
    if train_datasets and (training_args.task != "emrQA"):
        for train_dataset in train_datasets.values():
            if "target_vocab" in vars(train_dataset):
                target_vocab += train_dataset.target_vocab
    eval_datasets = _dataset(args.eval_data_file) if "valid" in splits else None
    if eval_datasets and (training_args.task != "emrQA"):
        for eval_dataset in eval_datasets.values():
            if "target_vocab" in vars(eval_dataset):
                target_vocab += eval_dataset.target_vocab
            if "actual_vocab" in vars(eval_dataset):
                actual_eval_vocab += eval_dataset.actual_vocab
    test_datasets = _dataset(args.test_data_file) if "test" in splits else None
    if test_datasets and (training_args.task != "emrQA"):
        for test_dataset in test_datasets.values():
            if "target_vocab" in vars(test_dataset):
                target_vocab += test_dataset.target_vocab
            if "actual_vocab" in vars(test_dataset):
                actual_eval_vocab += test_dataset.actual_vocab

    return train_datasets, eval_datasets, test_datasets, list(set(target_vocab)), list(set(actual_eval_vocab))