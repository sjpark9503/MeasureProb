import gc
import sys
from src.configs.parameters import parser
from transformers import set_seed
from src.utils.notifier import logging, log_formatter
from accelerate import Accelerator, DistributedType
from src.core.trainer import Trainer
notifier = logging.getLogger(__name__)
notifier.addHandler(log_formatter())
             
def main():
    accelerator = Accelerator(split_batches=True)

    # Parse arguments
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.model_name_or_path = model_args.model_name_or_path
 
    # Set seed
    set_seed(training_args.seed)

    # Call Logger
    wandb_config = dict()
    wandb_config.update(vars(training_args))
    wandb_config.update(vars(model_args))

    # Call Trainer
    trainer = Trainer(model_args, training_args, data_args)
  
    # Train model  
    # Create dataset on each device
    dataset, target_vocab, tokenizer = trainer.create_dataset(model_args, data_args, training_args,accelerator,splits=['train', 'valid']) 
    accelerator.wait_for_everyone()
    # Prepare data loader, model and optimizer (include lr scheduler)
    model, optimizer, train_dataloader, valid_dataloaders, lr_scheduler = trainer.prepare_train(dataset, tokenizer, target_vocab, accelerator, wandb_config)
    if training_args.do_train:
        trainer.train(model, train_dataloader, valid_dataloaders, accelerator, optimizer, lr_scheduler)
        trainer.keep_or_save(model, accelerator, mode="save")
        del dataset 
        gc.collect()
                              
    # Evaluate model
    if training_args.do_test:
        dataset, target_vocab, tokenizer = trainer.create_dataset(model_args, data_args,training_args,accelerator,splits=['test'])
        test_dataloaders = trainer.prepare_test(dataset, tokenizer, target_vocab, accelerator, wandb_config)
        trainer.evaluate(model, test_dataloaders, accelerator, global_step=-1, eval_mode="test")
    if accelerator.is_main_process:
        trainer.master_print(level="critical", message="Done! Save results & Terminate process")
        trainer.keep_or_save(model, accelerator, mode="save")
    # Terminate DDP manually
    accelerator.wait_for_everyone()
    sys.exit("Cleaning up...")
if __name__ == "__main__": 
    main()