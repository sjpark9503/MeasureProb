# Base pkgs
import os
import time
import subprocess
from src.configs.runConfig import Configuration
from src.utils.notifier import logging, log_formatter
notifier = logging.getLogger(__name__)
notifier.addHandler(log_formatter())

# Mode, [Exp] Experiment, [Debug] Debugging, [ExpDeep] Experiment w/ Deepspeed
mode = 'Exp'
assert mode in ['Exp', 'ExpDeep', 'Debug']
  
# Select Device
device = "TPU" 
device_id = "2"
if device == "GPU":
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = device_id
    
# Experiment Settings
config = {
    # RNG seeds (please do not change order)
    'seed' : [42,1234,123],
    # Key settings 
    'difficulty' : ["null_uom"],
    'task' : [1],
    'UoM' : ["general", "biomedical"], #   "general"  "biomedical"
    'probe' : ["clz-frz-full"], 
    'pe_type' : [0, 1],    
    'model' : ["bert", "biobert", "bluebert", "albert", ""], # ["bert", "biobert", "bluebert", "albert", "numbert", "bionumbert",],
    'notation' : ['word', 'sci'],
    # Hyperparams 
    'learning_rate' : 1e-4,
    'num_train_epochs' : 30,
    'train_batch_size' : '256',
    'eval_batch_size' : '128',
    'eval_frequency' : 0.5,
    # Exp envs
    'fp16' : True, 
    'device' : device,
    'mode' : mode,
}


EXP_LIST = [dict()]
for k,v in config.items():
    temp = list()
    for e in EXP_LIST:
        if isinstance(v, list):
            for _v in v:
                e[k] = _v
                temp.append(e.copy())
                _ = e.pop(k)
        else:
            e[k] = v
            temp.append(e.copy())
    EXP_LIST = temp

# Run Exps
for exp_idx, exp_setting in enumerate(EXP_LIST):
    notifier.critical(f"{exp_idx+1}/{len(EXP_LIST)} exp is running...")
    try:
        # Sanity check
        exp_config = Configuration(exp_setting)
        RUN_FLAG, error_log = exp_config.sanity_check()
        if not RUN_FLAG: 
            print(error_log)
            raise ValueError(error_log)
        # Run script
        SRC_PATH, TRAINING_CONFIG_LIST = exp_config.get_configuration()
        print(subprocess.run(['accelerate','launch',SRC_PATH]+TRAINING_CONFIG_LIST))
    except KeyboardInterrupt:
        import sys
        sys.exit()
    except:
        continue
    time.sleep(5)
 