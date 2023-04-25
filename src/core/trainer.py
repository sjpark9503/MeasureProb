# Base pkgs
import os
import ast
import itertools
import numpy as np
import math
from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch.optim import Adadelta, Adagrad, Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.data.dataloader import DataLoader
from transformers import get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from accelerate import DistributedType
# User defined pkgs
import sys
sys.path.append(os.path.dirname(os.getcwd()))
from ..data_utils.dataset import get_dataset
from ..data_utils.data_collator import PredDataCollator,QADataCollator
from .model import ProbingEncModel, print_num_learnable_params
# Transformers
from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForQuestionAnswering,
    AutoConfig,
    AutoTokenizer,
)
from datasets import load_metric
# Logging tool
from ..utils.notifier import logging, log_formatter
notifier = logging.getLogger(__name__)
notifier.addHandler(log_formatter())
               
MODEL_CLASS = {
    "clz": AutoModelForMaskedLM,
    "bin": AutoModel
}
              
class Trainer():
    def __init__(self, model_args, training_args, data_args):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        self.data_args = data_args
        self.eval_sets = {0:"Inter", 1:"Xtra"}
        
        self.best_eval_metric = {
            "valid":{"Inter":None, "Xtra":None, "":None},
            "test":{"Inter":None, "Xtra":None, "":None},
        }

        self.es_config = ast.literal_eval(self.training_args.es_config)
        self.es_count = 0

        self.samples_seen = 0
        self.few_shot_valid_interval = list(map(int,self.training_args.few_shot_valid_interval.split(',')))
         
    def create_dataset(self, model_args, data_args, training_args, accelerator, splits):
        # Load Tokenizer
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
        if model_args.encoder_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(model_args.encoder_name_or_path)
        elif model_args.model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
        elif model_args.tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)
        else:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
                "and load it from here, using --tokenizer_name"
            )
        num_added_tokens =  tokenizer.add_special_tokens({'additional_special_tokens':["[NUM]","scinotexp"]})

        train_dataset, eval_dataset, test_dataset, target_vocab, actual_eval_vocab = get_dataset(
                data_args,
                tokenizer=tokenizer,
                training_args=training_args,
                accelerator=accelerator,
                splits=splits
        )
        self.master_print(accelerator=accelerator,level="warning",message=f"Target vocabulary : {target_vocab}")
        self.master_print(accelerator=accelerator,level="warning",message=f"Actual vocabulary : {actual_eval_vocab}")
        self.dataset = {"train":train_dataset, "valid":eval_dataset, "test":test_dataset}

        return self.dataset, target_vocab, tokenizer

    def prepare_model(self, tokenizer, accelerator):
        if self.model_args.config_name:
            config = AutoConfig.from_pretrained(self.model_args.config_name)
        elif self.model_args.model_name_or_path:
            config = AutoConfig.from_pretrained(self.model_args.model_name_or_path)
        else:
            config = CONFIG_MAPPING[self.model_args.model_type]()
            self.master_print(accelerator=accelerator,level="warning",message="You are instantiating a new config instance from scratch.")
        config.model_name_or_path = self.model_args.model_name_or_path
        config.difficulty = self.training_args.difficulty
  
        # Load model for real-world task
        if self.training_args.task == 'emrQA':
            # Load encoder
            if self.model_args.model_name_or_path:
                self.master_print(accelerator=accelerator,level="critical",message="Load pretrained encoder")
                model = AutoModelForQuestionAnswering.from_pretrained(self.model_args.model_name_or_path, config=config)
            else:
                self.master_print(accelerator=accelerator,level="critical",message="Randomly initialize encoder")
                model = AutoModelForQuestionAnswering.from_config(config)
            model.scale_embedding = nn.Embedding(32, config.hidden_size)
        # Load model for synthetic task
        else:
            # Load encoder
            if self.model_args.model_name_or_path:
                self.master_print(accelerator=accelerator,level="critical",message="Load pretrained encoder")
                encoder = MODEL_CLASS[self.training_args.method].from_pretrained(self.model_args.model_name_or_path)
            else:
                self.master_print(accelerator=accelerator,level="critical",message="Randomly initialize encoder")
                encoder = MODEL_CLASS[self.training_args.method].from_config(config)
            encoder.training_args = self.training_args
            encoder.pe_type = self.training_args.pe_type
            encoder.resize_token_embeddings(len(tokenizer))

            # Define probe & freeze parameters
            model = ProbingEncModel(config=config, return_dict=True, tokenizer = tokenizer)
            model.encoder = encoder
            if self.model_args.freeze:
                model.freeze_encoder()
        self.master_print(accelerator=accelerator,level="warning",message="## Encoder Configuration ##")
        self.master_print(accelerator=accelerator,level="warning",message=model.config)
        print_num_learnable_params(model=model, accelerator=accelerator)
        
        return model

    def prepare_dataloader(self, dataset, tokenizer, target_vocab):
        if self.training_args.task=='emrQA':
            # Call Data Collator
            data_collator = QADataCollator(tokenizer=tokenizer, args=self.training_args)

            # Call Data Loader
            train_dataloader = DataLoader(
                dataset["train"],
                batch_size=self.training_args.train_batch_size,
                collate_fn=data_collator,
                drop_last=self.training_args.dataloader_drop_last,
                num_workers=self.training_args.dataloader_num_workers if self.training_args.device == "GPU" else 0,
                pin_memory=self.training_args.dataloader_pin_memory,
            ) if dataset["train"] is not None else None
            valid_dataloader = [DataLoader(
                dataset["valid"],
                batch_size=self.training_args.eval_batch_size,
                collate_fn=data_collator,
                drop_last=self.training_args.dataloader_drop_last,
                num_workers=self.training_args.dataloader_num_workers if self.training_args.device == "GPU" else 0,
                pin_memory=self.training_args.dataloader_pin_memory,
            ) if dataset["valid"] is not None else None
            ,]
            test_dataloader = [DataLoader(
                dataset["test"],
                batch_size=self.training_args.eval_batch_size,
                collate_fn=data_collator,
                drop_last=self.training_args.dataloader_drop_last,
                num_workers=self.training_args.dataloader_num_workers if self.training_args.device == "GPU" else 0,
                pin_memory=self.training_args.dataloader_pin_memory,
            ) if dataset["test"] is not None else None
            ,]
        else:
            # Call Data Collator
            if self.training_args.method == 'clz':
                living_vocab_idx = list(itertools.chain(*[tokenizer(tok,add_special_tokens=False)['input_ids'] for tok in target_vocab]))
                vocab_ignore_idx = [i for i in range(len(tokenizer)) if i not in living_vocab_idx]
                if "thesaurus" in self.training_args.difficulty:
                    thesaurus = {
                        "comparison":{                        
                            "larger":["higher","bigger","larger"],
                            "smaller":["lower","less","smaller"],
                        },
                        "min_max":{
                            "largest":["biggest","maximum","largest"],
                            "smallest":["lowest","minimum","smallest"],
                            "middle":["medium","intermediate","middle"],
                        },
                        "sort":{
                            "random":["random"],
                            "increasing":["growing","ascending","increasing"],
                            "decreasing":["descending","reducing","decreasing"],
                        },
                        "convert":{
                            "same":["identical", "equal", "same"],
                            "different":["distinct", "different"],
                        },
                        "val_range":{
                            "normal":["regular", "safe", "normal"],
                            "abnormal":["irregular", "lethal", "abnormal"],
                        },
                    }[self.training_args.task]
                    target2idx = dict()
                    for k, v in zip(target_vocab, living_vocab_idx):
                        target2idx[k] = v
                    self.alias = dict()
                    for k, v in thesaurus.items():
                        for x in v:
                            self.alias[target2idx[x]] = target2idx[k]
                else:
                    self.alias = None
                data_collator = PredDataCollator(tokenizer=tokenizer, args=self.training_args, vocab_ignore_idx=vocab_ignore_idx)
            else:
                data_collator = PredDataCollator(tokenizer=tokenizer, args=self.training_args)
            eval_data_collator = None

            # Call Data Loader
            train_dataloader = DataLoader(
                dataset["train"]["Inter"],
                batch_size=self.training_args.train_batch_size,
                collate_fn=data_collator,
                drop_last=self.training_args.dataloader_drop_last,
                num_workers=self.training_args.dataloader_num_workers if self.training_args.device == "GPU" else 0,
                pin_memory=self.training_args.dataloader_pin_memory,
            ) if dataset["train"] is not None else None

            valid_dataloader = [DataLoader(
                v,
                batch_size=self.training_args.eval_batch_size,
                collate_fn=eval_data_collator if eval_data_collator is not None else data_collator,
                drop_last=self.training_args.dataloader_drop_last,
                num_workers=self.training_args.dataloader_num_workers if self.training_args.device == "GPU" else 0,
                pin_memory=self.training_args.dataloader_pin_memory,) for k, v in dataset["valid"].items()
            ] if dataset["valid"] is not None else None
            test_dataloader = [DataLoader(
                v,
                batch_size=self.training_args.eval_batch_size,
                collate_fn=eval_data_collator if eval_data_collator is not None else data_collator,
                drop_last=self.training_args.dataloader_drop_last,
                num_workers=self.training_args.dataloader_num_workers if self.training_args.device == "GPU" else 0,
                pin_memory=self.training_args.dataloader_pin_memory,) for k, v in dataset["test"].items()
            ] if dataset["test"] is not None else None
        return train_dataloader, valid_dataloader, test_dataloader

    def prepare_optimizer(self, model):
        # Define optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.training_args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer_cls = {"Adadelta":Adadelta, "Adagrad":Adagrad, "Adam":Adam, "AdamW":AdamW}[self.training_args.optimizer]
        if "Adam" in self.training_args.optimizer:
            optimizer_kwargs = {
                "betas": (self.training_args.adam_beta1, self.training_args.adam_beta2),
                "eps": self.training_args.adam_epsilon,
            }
        optimizer_kwargs["lr"] = self.training_args.learning_rate
        optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return optimizer

    def get_lr_scheduler(self, data_loader, optimizer):
        # Define scheduler
        num_update_steps_per_epoch = len(data_loader) // self.training_args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)

        if self.training_args.max_steps > 0:
            max_steps = self.training_args.max_steps
        else:
            max_steps = math.ceil(self.training_args.num_train_epochs * num_update_steps_per_epoch)

        lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=self.training_args.warmup_steps,
                num_training_steps=max_steps,
            )
        return lr_scheduler

    def prepare_train(self, dataset, tokenizer, target_vocab, accelerator, wandb_config=None):
        # Prepare logger
        if accelerator.is_main_process:
            import wandb
            self.logger = wandb.init(config=wandb_config, entity="edlab_sjpark", project="EMNLP2022", group=self.training_args.run_name, name=f"{self.training_args.difficulty}({str(self.training_args.seed)})")
        # Prepare Model
        model = self.prepare_model(tokenizer, accelerator)
        if self.training_args.do_test and not self.training_args.do_train:
            model.load_state_dict(torch.load(os.path.join(self.training_args.output_dir,"pytorch_model.bin")))
        # Prepare Optimizer 
        optimizer = self.prepare_optimizer(model)
        # Wrap model w/ Huggingface Accelerate
        model, optimizer = accelerator.prepare(model, optimizer)
        # Prepare Data Loader
        if not (dataset["train"] is None or dataset["valid"] is None):
            train_dataloader, valid_dataloaders, _ = self.prepare_dataloader(dataset, tokenizer, target_vocab)
            train_dataloader = accelerator.prepare(train_dataloader)
            valid_dataloaders = [accelerator.prepare(x) for x in valid_dataloaders]
            lr_scheduler = self.get_lr_scheduler(train_dataloader, optimizer)
        else:
            train_dataloader,valid_dataloaders, lr_scheduler = None, None, None
        # Print Early Stop Creterion
        self.master_print(accelerator=accelerator,level="warning",message=f"Early stop criterion: ")
        self.master_print(accelerator=accelerator,level="warning",message=self.es_config)
        
        return model, optimizer, train_dataloader, valid_dataloaders, lr_scheduler

    def prepare_test(self, dataset, tokenizer, target_vocab, accelerator, wandb_config=None):
        # Prepare Data Loader
        _, _, test_dataloaders = self.prepare_dataloader(dataset, tokenizer, target_vocab)
        test_dataloaders = [accelerator.prepare(x) for x in test_dataloaders]
        
        return test_dataloaders

    def train(self, model, train_dataloader, valid_dataloaders, accelerator, optimizer, lr_scheduler):
        for epoch in trange(self.training_args.num_train_epochs):
            # Initialize global step
            if epoch == 0:
                global_step = 0
            # Validation on every epoch start
            early_stop = self.evaluate(model, valid_dataloaders, accelerator, global_step=global_step, eval_mode="valid")
            model.train()
            # Early stopping & Keep best model
            if early_stop:
                break
            elif self.es_count==0:
                self.keep_or_save(model, accelerator, mode="keep")
            # Training loop
            for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), disable=not accelerator.is_main_process, desc= f"Train [{epoch}]", leave=False):
                # if "offset_mapping" in batch:
                #     offsets = batch.pop("offset_mapping")
                if (self.training_args.task=='emrQA') and ("additional_pe" in batch):
                    batch['input_ids'], batch['inputs_embeds'] = self._get_input_embeds(model=model, input_ids=batch['input_ids'], additional_pe=batch['additional_pe'])
                    batch.pop('additional_pe')
                outputs = model(**batch)
                loss = outputs.loss
                loss = loss / self.training_args.gradient_accumulation_steps
                # Log
                if accelerator.is_main_process and ((step+epoch*len(train_dataloader))%self.training_args.logging_steps==0):
                    self.logger.log({"loss":loss.item()})
                accelerator.backward(loss)
                # Optimizer step
                if step % self.training_args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    global_step += self.training_args.gradient_accumulation_steps 
            
    def evaluate(self, model, eval_dataloaders, accelerator, global_step, eval_mode):
        model.eval() 
        TEMP_FLAGS = list()
        for dataset_idx, data_loader in enumerate(eval_dataloaders):
            if len(eval_dataloaders)>1:
                db_name = self.eval_sets[dataset_idx]
            else:
                db_name = ''
            if self.training_args.task=='emrQA':
                FLAG = self.QA_evaluation_loop(model, data_loader, accelerator, global_step, eval_mode, db_name)
            else:
                FLAG = self.synthetic_evaluation_loop(model, data_loader, accelerator, global_step, eval_mode, db_name)
            TEMP_FLAGS.append(FLAG)
        EARLYSTOP_FLAG = torch.tensor(TEMP_FLAGS).any()

        return EARLYSTOP_FLAG

    def QA_evaluation_loop(self, model, data_loader, accelerator, global_step, eval_mode, db_name=""):
        eval_epoch_ouputs = {"loss":list(),"scores":list()}
        epoch_predictions = dict()
        examples = self.dataset[eval_mode].answers

        for step, batch in tqdm(enumerate(data_loader), total=len(data_loader) ,disable=not accelerator.is_main_process, desc=f"{eval_mode.upper()}, {db_name}", leave=False):
            with torch.no_grad():
                if "id" in batch:
                    ids = batch.pop("id")
                if "additional_pe" in batch:
                    batch['input_ids'], batch['inputs_embeds'] = self._get_input_embeds(model=model, input_ids=batch['input_ids'], additional_pe=batch['additional_pe'])
                    batch.pop('additional_pe')
                outputs = model(**batch)
            # Gather the sharded evaluation results
            if accelerator.distributed_type == DistributedType.TPU:
                start_logits = accelerator.gather(outputs.start_logits)
                end_logits = accelerator.gather(outputs.end_logits)
                ids = accelerator.gather(ids)
            # Or not!  
            else:
                start_logits = start_logits
                end_logits = end_logits

            predictions = self.postprocess_qa_predictions(examples, ids, start_logits, end_logits, accelerator=accelerator)
            epoch_predictions.update(predictions)
        
        metric = load_metric("squad")
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in epoch_predictions.items()]
        references = [{"id": k, "answers": {'answer_start':[], 'text':[v["answer"]]}} for k, v in examples.items()]

        metrics = metric.compute(predictions=formatted_predictions, references=references)
        final_metrics = dict()
        for k, v in metrics.items():
            final_metrics[f"{eval_mode}_{k}"] = v
        EARLYSTOP_FLAG = self.EarlyStopping(final_metrics, eval_mode, db_name, accelerator)
        # Log metrics only on rank 0 process
        if accelerator.is_main_process:
            self.logger.log(final_metrics)

        return EARLYSTOP_FLAG 
  
    def synthetic_evaluation_loop(self, model, data_loader, accelerator, global_step, eval_mode, db_name=""):
        eval_epoch_ouputs = {"loss":list(),"scores":list()}
        for step, batch in tqdm(enumerate(data_loader), total=len(data_loader) ,disable=not accelerator.is_main_process, desc=f"{eval_mode.upper()}, {db_name}", leave=False):
            with torch.no_grad():
                outputs = model(**batch, alias=self.alias)
            # Gather the sharded evaluation results
            if accelerator.distributed_type == DistributedType.TPU:
                gathered_losses = accelerator.gather(outputs.loss)
                gathered_scores = accelerator.gather(outputs.scores)
            # Or not! 
            else:
                gathered_losses = outputs.loss
                gathered_scores = outputs.scores

            eval_epoch_ouputs["loss"].append(gathered_losses)
            eval_epoch_ouputs["scores"].append(gathered_scores)
    
        # Measure the metrics
        _epoch_score = torch.cat(eval_epoch_ouputs["scores"])
        epoch_accuracy = _epoch_score.sum()/_epoch_score.size(0)
        if eval_epoch_ouputs["loss"][0].size()!=0: 
            epoch_loss = torch.cat(eval_epoch_ouputs["loss"]).mean().item()
        else:
            epoch_loss = torch.tensor(eval_epoch_ouputs["loss"]).mean().item()
        metrics = {
            f'{eval_mode}_{db_name}_loss': epoch_loss,
            f'{eval_mode}_{db_name}_acc': epoch_accuracy,
        }
        EARLYSTOP_FLAG = self.EarlyStopping(metrics, eval_mode, db_name, accelerator)
        if global_step == 0:
            metrics.update({f"zeroshot_{eval_mode}_{db_name}_acc": epoch_accuracy})
        if eval_mode=="valid":
            metrics.update({f'best_{self.es_config["target"]}': self.best_eval_metric[eval_mode][db_name]})
        # Log metrics only on rank 0 process
        if accelerator.is_main_process:
            self.logger.log(metrics)

        return EARLYSTOP_FLAG 
 
    def EarlyStopping(self, metrics, eval_mode, db_name, accelerator):
        if eval_mode == "test":
            return False
        else:
            target = self.es_config['target']
            if target in metrics:
                self.master_print(accelerator=accelerator,level="warning",message=f"Early stop target [{target}] is detected. Evaluating early criterion...")
                if self.best_eval_metric[eval_mode][db_name] is None:
                    self.best_eval_metric[eval_mode][db_name] = metrics[target]
                    return False
                else:
                    criterion = metrics[target] - self.best_eval_metric[eval_mode][db_name]
                    if (criterion>self.es_config["delta"]) and (self.es_config["mode"]=="max"):
                        self.best_eval_metric[eval_mode][db_name] = metrics[target]
                        self.es_count = 0
                        return False
                    elif (criterion<-self.es_config["delta"]) and (self.es_config["mode"]=="min"):
                        self.best_eval_metric[eval_mode][db_name] = metrics[target]
                        self.es_count = 0
                        return False
                    else:
                        self.es_count += 1
                        if self.es_count < self.es_config["patience"]:
                            return False
                        else:
                            return True
        return False
           
    def keep_or_save(self, model, accelerator, mode=None):
        output_dir = self.training_args.output_dir
        # Wait for all distributed workers & unwrap model
        unwarpped_model = accelerator.unwrap_model(model)
        if mode == "keep":
            self.best_parameter = unwarpped_model.state_dict()
        elif mode == "save":
            accelerator.wait_for_everyone()
            unwarpped_model.load_state_dict(self.best_parameter)
            os.makedirs(output_dir, exist_ok=True)
            # accelerator.save(unwarpped_model.state_dict(), os.path.join(output_dir,"pytorch_model.bin"))
            # accelerator.save(unwarpped_model.config, os.path.join(output_dir,"config.json"))
            self.master_print(accelerator=accelerator,level="warning",message=f"Successfully saved model to {output_dir}")
        else: 
            raise NotImplementedError()

        return model

    def master_print(self, message, level, accelerator=None):
        def _level_wise_print(message, level):
            if level == "warning":
                notifier.warning(message)
            elif level == "critical":
                notifier.critical(message)
            else:
                notifier.info(message)
        if accelerator is None:
            _level_wise_print(message=message,level=level)
        else:
            if accelerator.is_main_process:
                _level_wise_print(message=message,level=level)

    def _get_input_embeds(self, model, input_ids, additional_pe=None):
        word_embedding = model.get_input_embeddings()
        input_embeds = word_embedding(input_ids)
        if additional_pe is not None:
            input_embeds += model.scale_embedding(additional_pe)
        return None, input_embeds

    def postprocess_qa_predictions(self, examples, ids, all_start_logits, all_end_logits, n_best_size = 20, max_answer_length = 30, accelerator=None):
        # The dictionaries we have to fill.
        predictions = dict()

        # Let's loop over all the examples!
        for _id, start_logit, end_logit in zip(ids, all_start_logits, all_end_logits):
            _id = _id.int().item()
            valid_answers = []
            # self.master_print(accelerator=accelerator,level="warning",message=examples[_id])
            context = examples[_id]["context"]

            # This is what will allow us to map some the positions in our logits to span of texts in the original context.
            offset_mapping = examples[_id]["offset_mapping"]

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logit.numpy())[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit.numpy())[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logit[start_index] + end_logit[end_index],
                            "text": context[start_char: end_char]
                        }
                    )
            
            if len(valid_answers) > 0:
                best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
            else:
                # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
                # failure.
                best_answer = {"text": "", "score": 0.0}
            
            # Let's pick our final answer: the best one or the null answer (only for squad_v2)
            answer = best_answer["text"]
            predictions[_id] = answer

        return predictions


