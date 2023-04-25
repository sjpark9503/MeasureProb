import os

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
             
from transformers.trainer_utils import EvaluationStrategy, IntervalStrategy, SchedulerType, ShardedDDPOption
from transformers.utils import logging

logger = logging.get_logger(__name__)

def default_logdir() -> str:
    """
    Same default as PyTorch
    """
    import socket
    from datetime import datetime

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    return os.path.join("runs", current_time + "_" + socket.gethostname())

@dataclass
class TrainingArguments:
    # For Experiment
    task: Optional[str] = field(default=None, metadata={"help": "Set task"})
    difficulty: Optional[str] = field(default=None, metadata={"help": "Set experiment name"})
    method: Optional[str] = field(default=None, metadata={"help": "Set method"})
    device: Optional[str] = field(default=None, metadata={"help": "Device type"})
    es_config: Optional[str] = field(default=None, metadata={"help": "Early stop configuration"})
    shot: str = field(default=False, metadata={"help": "Turn on few-shot learning setting?"})
    deepspeed: str = field(default=False, metadata={"help": "Turn on deepspeed?"})
    pe_type: int = field(default=0, metadata={"help": "Additional positional encoding to handle scale of number?"})
    num_category: int = field(default=2, metadata={"help": "# possible category of output. this may differ for task."})
    fp16: bool = field(default=False, metadata={"help": "AMP training"})
    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})
    run_name: Optional[str] = field(
        default=None, metadata={"help": "An optional descriptor for the run. Notably used for wandb logging."}
    )
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory."
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )

    # For Training
    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    num_train_epochs: int = field(default=5, metadata={"help": "Total number of training epochs to perform."})
    train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )

    # For Evaluation
    do_eval: bool = field(default=None, metadata={"help": "Whether to run eval on the dev set."})
    do_test: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})
    eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    eval_accumulation_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Number of predictions steps to accumulate before moving the tensors to the CPU."},
    )
    eval_frequency: float = field(default=None, metadata={"help": "Run an evaluation every X steps."})
    few_shot_valid_interval: str = field(default=None, metadata={"help": "Specific # samples that you want to log. comma sepreated string."})

    # For Optimizer
    learning_rate: float = field(default=1e-4, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})
    optimizer: str = field(default="Adam", metadata={"help": "Whether or not to replace AdamW by Adafactor."})
    lr_scheduler_type: SchedulerType = field(
        default="linear",
        metadata={"help": "The scheduler type to use."},
    )
    warmup_ratio: float = field(
        default=0.0, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})
    
    # For data
    dataloader_pin_memory: bool = field(
        default=True, metadata={"help": "Whether or not to pin memory for DataLoader."}
    )
    dataloader_num_workers: int = field(
        default=8, 
        metadata={
            "help": "Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded in the main process."
        },
    )
    dataloader_drop_last: bool = field(
        default=False, metadata={"help": "Drop the last incomplete batch if it is not divisible by the batch size."}
    )

    # For logging
    logging_dir: Optional[str] = field(default_factory=default_logdir, metadata={"help": "Tensorboard log dir."})
    logging_strategy: IntervalStrategy = field(
        default="steps",
        metadata={"help": "The logging strategy to use."},
    )
    logging_first_step: bool = field(default=True, metadata={"help": "Log the first global_step"})
    logging_steps: int = field(default=50, metadata={"help": "Log every X updates steps."})