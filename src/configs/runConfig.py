import os
              
TASKS = {1: 'comparison', 2: 'min_max', 3: 'sort', 4: 'convert', 5: 'num2word', 6: 'val_range', 7: 'val_range_hard'}
CAT_SIZE = {1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 'emrQA':2}
MODELS = {'bert': 'bert-base-uncased',
        'biobert': 'dmis-lab/biobert-v1.1',
        'bluebert': 'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12',
        'albert': 'albert-base-v2',
        'roberta': 'roberta-base',
        'numbert': "NumBERT",
        'bionumbert' : "BioNumBERT",
        '': ''
}
RAW_NAME = {'bert': 'BERT',
        'biobert': 'BioBERT',
        'bluebert': 'BlueBERT',
        'albert': 'ALBERT',
        'roberta': 'RoBERTa',
        'numbert': "NumBERT",
        'bionumbert' : "BioNumBERT",
        '': 'Rand'
} 

class Configuration():
    def __init__(self, config):
        config['method'], config['freeze'], config['shot'] = config['probe'].split('-') if config['task'] not in ["emrQA",] else ["","",""]
        self.config = config
 
        # Helper Variables
        self.TASK_NAME = TASKS[config['task']] if isinstance(config['task'],int) else config['task']
        self.RAW_MODEL_NAME = RAW_NAME[config['model']]

        # Set PATH vars
        self.ROOT_PATH = os.getcwd()

        if isinstance(config['task'],int):
            self.EXP_PATH = os.path.join(config['difficulty'],
                                    str(config['task']),
                                    f"{config['probe']}",
                                    f"{self.RAW_MODEL_NAME}-pe{self.config['pe_type']}-{self.config['notation']}",
                                    f"{self.config['UoM']}",
                                    f"RNG{self.config['seed']}"
                            )    
            self.DATA_PATH = os.path.join(self.ROOT_PATH,
                                        config['difficulty'],
                                        str(config['task']),
                                        f"{config['method']}-{'all' if config['method']!='clz' else self.RAW_MODEL_NAME}/[SPLIT]/{config['UoM']}-{config['notation']}-[PH].bin"
                                        ) 
            if self.config['model'] in ['numbert','bionumbert']:
                self.DATA_PATH = self.DATA_PATH.replace(self.RAW_MODEL_NAME, "BERT")
        elif isinstance(config['task'],str):
            if config['task']=="emrQA":
                self.EXP_PATH = os.path.join(f"{self.TASK_NAME}",
                                    f"{self.RAW_MODEL_NAME}-pe{self.config['pe_type']}-{self.config['notation']}",
                                    f"RNG{self.config['seed']}"
                                )
                self.DATA_PATH = os.path.join(self.ROOT_PATH,
                                        "data",
                                        "emrQA",
                                        "minimal",
                                        f"normal_[SPLIT]-{self.config['notation']}.bin"
                                    )
        else:
            raise ValueError("Invalid type for task_number")
                         
        self.TRAINING_CONFIG = {
            "block_size": 512,
            "optimizer": "Adam",
            "do_train": True,
            "do_eval": True,
            "do_test": True, 
            "overwrite_output_dir": False,
            "num_category": 2 if config['method']=='bin' else CAT_SIZE[config['task']],
            "task" : self.TASK_NAME,
            "freeze": config['freeze']=='frz',
            "few_shot_valid_interval": "0,40,200,1000,5000,10000,20000",
            "train_data_file": self.DATA_PATH.replace("[SPLIT]","train").replace("[PH]","Inter"),
            "eval_data_file": self.DATA_PATH.replace("[SPLIT]","valid"),
            "test_data_file": self.DATA_PATH.replace("[SPLIT]","test"),
            "run_name": ','.join(self.EXP_PATH.split('/')[:-1]),
            "output_dir": os.path.join(self.ROOT_PATH,"pretrained_models", config["difficulty"], self.EXP_PATH),
        }
        # Add early stop configuration.
        self.TRAINING_CONFIG['es_config'] = str({
            "target":f"valid_{'Inter_loss' if isinstance(config['task'],int) else 'f1'}",
            "patience": 2 if isinstance(config['task'],int) else 3,
            "mode": "min" if isinstance(config['task'],int) else "max",
            "delta":0,
        })
        # Add some additional variables
        for k,v in config.items():
            if (k not in self.TRAINING_CONFIG) and (k not in ['model', 'mode', 'probe', 'UoM', 'notation']):
                self.TRAINING_CONFIG[k] = v

    def get_configuration(self):
        SRC_PATH = os.path.join(self.ROOT_PATH, f'main.py')  
        if (not self.config['model']) and self.TRAINING_CONFIG['do_train']:
            self.TRAINING_CONFIG['config_name'] = 'bert-base-uncased'
            self.TRAINING_CONFIG['tokenizer_name'] = 'bert-base-uncased'
            self.TRAINING_CONFIG['model_name_or_path'] = ''
        else:
            if not self.TRAINING_CONFIG['do_train']:
                self.TRAINING_CONFIG['model_name_or_path'] = self.TRAINING_CONFIG['output_dir']
            elif self.config['model'] in ['numbert', 'bionumbert']:
                self.TRAINING_CONFIG['model_name_or_path'] = os.path.join(self.ROOT_PATH, MODELS[self.config['model']])
            else:
                self.TRAINING_CONFIG['model_name_or_path'] = MODELS[self.config['model']]

        # Debugging mode
        if self.config['mode'] == 'Debug':
            print('-'*50+'\n\t\t Debugging mode ON\n'+'-'*50)
            self.TRAINING_CONFIG['train_batch_size']=2
            self.TRAINING_CONFIG['eval_batch_size']=2
            self.TRAINING_CONFIG['overwrite_output_dir'] = True
            self.TRAINING_CONFIG['run_name'] = f"{self.TASK_NAME}/debug"
            self.TRAINING_CONFIG['output_dir'] = os.path.join(self.ROOT_PATH,f"pretrained_models/{self.TASK_NAME}/debug")
            for k in self.TRAINING_CONFIG:
                if 'file' in k:
                    self.TRAINING_CONFIG[k] = self.TRAINING_CONFIG[k].replace(f"/{self.config['task']}/","/debug/")

        TRAINING_CONFIG_LIST = list()
        for (k,v) in list(self.TRAINING_CONFIG.items()):
            if (isinstance(v, bool)):
                if v:
                    TRAINING_CONFIG_LIST.append("--{}".format(k))
            else:
                TRAINING_CONFIG_LIST.append("--{}={}".format(k,v))
        return SRC_PATH, TRAINING_CONFIG_LIST 

    def sanity_check(self):
        if not self.TRAINING_CONFIG["overwrite_output_dir"] and os.path.exists(self.TRAINING_CONFIG["output_dir"]) and self.TRAINING_CONFIG["do_train"]:
            return False, "Output directory already exists. Consider turn on overwrite_ouput_dir."

        elif (self.TRAINING_CONFIG["task"] == "val_range") and ("biomedical" in self.TRAINING_CONFIG["train_data_file"]):
            return False, "Reference Range detection does not support biomedical UoMs."

        elif (self.config['model'] == 'bionumbert') and (self.config['notation'] != 'word'):
            return False, "BioNumBERT has specific input format, so we ignore the other type of input notation."

        elif (self.config['model'] == 'numbert') and (self.config['notation'] != 'sci'):
            return False, "NumBERT is trained on scientific notation, so ignore other notations."

        elif (self.config['model'] in ['numbert','bionumbert']) and (self.config['pe_type'] == 1):
            return False, "BioNumBERT and NumBERT does not support additional postional encoding"

        return True, None
