import torch
import h5py
from tqdm import tqdm
from termcolor import colored
import numpy as np
from math import log10, floor
from numpy.random import random, uniform, choice, shuffle, seed, randint
from typing import Optional
from num2words import num2words
from _gen_utils import unit2words, Unit2CommonPrefix, unit_conversion, unit_swap
import os
import re
import csv
from itertools import chain
from dataclasses import dataclass
from transformers import AutoTokenizer, PreTrainedTokenizer

DEBUG = False

DATA_SIZE = {'train':int(3e5) if not DEBUG else int(1e2), 
        'valid':int(3e4) if not DEBUG else int(1e1), 
        'test':int(3e4) if not DEBUG else int(1e1)}

config = {
    "template": "normal",
    #"template": "paraphrase",
    "difficulty": "manual_conversion",
    # "difficulty": "hard,onlyUoM",
    # "difficulty": "easy",
    "task" : [6],
    "method": ["numgen", "clz"],
    "note": ["word", "sci"],
    "extrapolation": [False, True],
    "uom_domain": ["general", "biomedical"],
    "dist": "uniform",
    "mantissa": 3,
    "range": {'xlb':0.001, 'lb':0.01, 'ub':100, 'xub':1000},
    "split": ["train", "valid", "test"],
}

MODEL_LIST = {
    'bert-base-uncased': "BERT",
    'dmis-lab/biobert-v1.1': "BioBERT",
    'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12': "BlueBERT",
    'albert-base-v2' : "ALBERT",
    '': 'Rand'
}

with open(f'hard_refrange.csv', newline='') as f:
    reader = csv.reader(f)
    RefRange = list(reader)[1:]
    
    if config['difficulty'] in ["easy", "manual_conversion"]:
        itm2range = {True: {rec[0].lower().strip():{"ref_range":(float(rec[2]),float(rec[3])),"uom":rec[1].lower().strip(), "prefix":rec[4].lower().strip()} for rec in RefRange[int(0.8*len(RefRange)):]},
                    False: {rec[0].lower().strip():{"ref_range":(float(rec[2]),float(rec[3])),"uom":rec[1].lower().strip(), "prefix":rec[4].lower().strip()} for rec in RefRange[:int(0.8*len(RefRange))]}}
    else:
        itm2range = {True: {rec[0].lower().strip():{"ref_range":(float(rec[2]),float(rec[3])),"uom":rec[1].lower().strip(), "prefix":rec[4].lower().strip()} for rec in RefRange},
                    False: {rec[0].lower().strip():{"ref_range":(float(rec[2]),float(rec[3])),"uom":rec[1].lower().strip(), "prefix":rec[4].lower().strip()} for rec in RefRange}} 
    full_itm2range = {rec[0].lower().strip():{"ref_range":(float(rec[2]),float(rec[3])),"uom":rec[1].lower().strip(), "prefix":rec[4].lower().strip()} for rec in RefRange}

TEMPLATE = {
    "normal":{
        1:{"bin":"{} is larger than {}","clz":"{} is {} than {}"},
        2:{"bin":"The {} amount among {} is {}","clz":"The {} amount among {} is {}"},
        3:{"bin":"sort {} in {} order is {}","clz":"sort {} in {} order is {}"},
        4:{"bin":"{} and {} are {} amount","clz":"{} and {} are the {} amount"},
        5:{"bin":"convert {} to word is {}","clz":None},
        6:{"bin":"{} of {} is normal","clz":"{} of {} is {}"},
        7:{"bin":"The unit {} is appropriate for {}","clz":"The unit {} is {} for {}"}
    },
    "paraphrase":{
        1:{"clz":[
            "{0} is {1} than {2}",
            "compared to {2}, {0} is {1} value",
            "The measurement of control group ({0}) is {1} than {2}",
            "comparison: {0}, {2}, result: {1}",
            "{0} {1} {2}",
         ]},
        2:{"clz":[
            "The {0} value among {1} is {2}",
            "{2} is the {0} value of {1}",
            "Among the list of measurements {1}, the {0} value is {2}",
            "argmin,argmax: {1}, {2}, result: {0}",
            "{0} {1} , {2}",
         ]},
        3:{"clz":[
            "sort {0} in {1} order is {2}",
            "arranging {0} in {1} order is {2}",
            "{2} is obtained by sorting {0} in {1} order",
            "sort: {0}, {2}, result: {1}",
            "{0} {1} {2}",
         ]},
        4:{"clz": [
            "{0} and {1} are the {2} value",
            "If you convert {0} to {2} value, then the result is {1}",
            "If you compare {0} to {1}, the two are the {2} value",
            "measurement comparison: {0}, {1}, result: {2}",
            "{0} , {1} {2}",
         ]},
        6:{"clz":[
            "{0} of {1} is {2}",
            "{0} of {1} falls into {2} range",
            "The physician decides {0} of {1} as {2}",
            "reference range: {0}, {1}, result: {2}",
            "{0} {1} {2}",
         ]},
    },
}

def generate_number(amount, method, xtrapol, index=None, lb=config['range']['lb'], ub=config['range']['ub'], xlb=config['range']['xlb'], xub=config['range']['xub']):
    precisions = list()
    if method == "numgen":
        if not xtrapol:
            raw_numbers = uniform(low=lb, high=ub, size=amount)
        elif xlb == lb:
            raw_numbers = uniform(low=ub, high=xub, size=amount)
        elif xub == ub:
            raw_numbers = uniform(low=xlb, high=lb, size=amount)
        else:
            if task in [6,7]:
                raw_numbers = [uniform(low=xlb, high=lb) if random()>0.5 else uniform(low=ub, high=xub) for _ in range(amount)]
            else:
                raw_numbers = uniform(low=xlb, high=xub, size=amount)
        numbers = list()
        for raw_number in raw_numbers:
            if raw_number <= 0.001:
                precision = 3
            elif raw_number == 0:
                precision = 0
            else:
                precision = randint(max(int(1-log10(raw_number)),0), config['mantissa']+1)
            number = "{{:.{}f}}".format(precision).format(raw_number)
            precisions.append(precision)
            numbers.append(number)
        preset = (numbers, precisions)
    else:
        preset = global_sample_bag[index]

    return preset

def formatting_number(number, UoM, notation, precision, word_form=False, prefix=None, add_uom=True):
    # Add prefix to UoM
    if prefix:
        UoM = "/".join([p+u for p,u in zip(prefix.split("/"),UoM.split("/"))])
    # Convert number to word
    try:
        word = f"{num2words(number)} {unit2words(UoM)}"
    except:
        word = None
        #print(number, UoM)
    
    # Number formatting
    formatted_number = "{{:.{}{}}}".format(precision, "e" if "sci" in notation else "f").format(float(number))
    if 'char' in notation:
        formatted_number = ' '.join([char for char in list(formatted_number)])
    if 'spec' in notation:
        formatted_number = formatted_number.replace('e','scinotexp')
    # Add UoM for measurement
    measure = f"{formatted_number}{UoM}" if add_uom else f"{formatted_number}"

    return {"measures":measure if not word_form else word, 
            "formatted_numbers":formatted_number, 
            "UoMs":UoM, 
            "word_numbers":word}

def _get_num_offset(sample, values, notation):
    # Exclude exponent from scientific notation
    rule = re.compile('-?[0-9\s]+\.?[0-9\s]*')
    nums = [re.findall(rule, value)[0].strip() for value in values]
    # Get number offset
    num_offset = list()
    for num, val in zip(nums,values):
        pos_starts = [p.start() for p in re.finditer(re.escape(val), sample)]
        for p in pos_starts:
            is_char = "char" in notation
            pattern = "[\d.]\s" if is_char else "[\d.]"
            pattern_len = 2 if is_char else 1
            is_sod = re.match(pattern,sample[p-pattern_len:p]) is not None
            if (p==0) or not is_sod:
                num_offset.append((p,p+len(num)))

    return list(set(num_offset))

def _get_new_bounds(ref_range, xtrapol):
    xlb = config['range']['xlb'] if xtrapol else config['range']['lb']
    xub = config['range']['xub'] if xtrapol else config['range']['ub']
    if ref_range[0] < xlb:
        xlb = 0
    if ref_range[1] > xub:
        xub = 1000
    lb = ref_range[0]
    ub = ref_range[1]

    return {"xlb":xlb, "lb":lb, "ub":ub, "xub":xub}

def sanity_check(x):
    if x not in global_sanity_bag:
        global_sanity_bag[x] = None
        return True
    else:
        return False 

def sample_generator(task, method, index, xtrapol, uom_domain, template, notation, tok, difficulty="easy", **kwargs):
    # Get number and template
    amount = {1: 2, 2: randint(low=3, high=11), 3: randint(low=3, high=11), 4: 1, 5: 2, 6: 1, 7:1}[task]
    if task in [6,7]:
        entity = choice(list(itm2range[xtrapol].keys()))
        new_range = _get_new_bounds(itm2range[xtrapol][entity]["ref_range"], xtrapol)
        label = choice(["normal", "abnormal"])
        preset = generate_number(amount, method=method, index=index, xtrapol=True if label=="abnormal" else False, **new_range)
    else:
        preset = generate_number(amount, method=method, index=index, xtrapol=xtrapol)
    
    # Number generation mode
    if method == "numgen":
        if task in [6,7]:
            UoM = itm2range[xtrapol][entity]["uom"]
            org_prefix = itm2range[xtrapol][entity]["prefix"]
            preset = [preset[0], preset[1], UoM, org_prefix, entity]
        else:
            UoM = choice(list(Unit2CommonPrefix[uom_domain].keys()))
            org_prefix = "" if "noprefix" == difficulty else choice(Unit2CommonPrefix[uom_domain][UoM])
            preset = [preset[0], preset[1], UoM, org_prefix]
        global_sample_bag.append(preset)

        return preset

    else:
        try:
            if task in [6,7]:
                numbers, precisions, UoM, org_prefix, entity = preset
            else:
                numbers, precisions, UoM, org_prefix = preset
            if notation == "sci":
                precisions = [p+floor(log10(float(n))) if float(n)>0 else 0 for n, p in zip(numbers, precisions)]
            sample = {
                "measures": [],
                "formatted_numbers": [],
                "raw_numbers": [],
                "UoMs": [],
                "word_numbers": [],
            }

            if difficulty == "null_uom":
                UoM = choice(
                    {
                        "general":["ww", "xx", "yy", "zz"],
                        "biomedical":["ii", "jj", "kk", "ll", "nn", "oo", "qq", "rr", "ss", "tt", "uu", "vv", "ww", "xx", "yy", "zz"],
                    }[uom_domain]
                )
                org_prefix = ""

            if task == 4:
                number = numbers[0]
                ub, lb = (config['range']['xub'], config['range']['xlb'])
                converted_prefix, converted_number = unit_conversion(UoM, org_prefix, float(number), ub, lb, uom_domain, negative=False, include_itself=False, task=task)
                neg_prefix, neg_number = unit_conversion(UoM, org_prefix, float(number), ub, lb, uom_domain, negative=True, include_itself=False, task=task)
                numbers = [number, converted_number, neg_number]
                precisions = precisions*3
                prefixes = [org_prefix, converted_prefix, neg_prefix]
            else:
                prefixes = [org_prefix]*len(numbers)
            for number, precision, org_prefix in zip(numbers, precisions, prefixes):
                if ("hard,onlyUoM" == difficulty) and (task != 4):
                    prefix, converted_number = unit_swap(UoM, org_prefix, float(number), uom_domain, task)
                    if converted_number == 0:
                        number_before_conversion = 0
                    else:
                        number_before_conversion = float(number)*(float(number)/converted_number)
                    number_for_precision = float(number)
                elif "manual_conversion" == difficulty:
                    if task not in [4,]:
                        # Attach the different UoM 
                        prefix, converted_number = unit_swap(UoM, org_prefix, float(number), uom_domain, task)
                        if converted_number == 0:
                            number_before_conversion = 0
                        else:
                            number_before_conversion = float(number)*(float(number)/converted_number)   
                        # Re-conversion of number
                        org_prefix = prefix  
                    else:
                        number_before_conversion = float(number)  
                    if task == 4:
                        number_for_precision = float(numbers[0])
                    else:                    
                        number_for_precision = float(number)
                    ub, lb = (new_range['xub'], new_range['xlb']) if task in [6,7] else (config['range']['xub'], config['range']['xlb'])
                    prefix, number = unit_conversion(UoM, org_prefix, float(number), ub, lb, uom_domain, 
                                                      negative=False, include_itself=True, man_conversion=True, task=task)
                else:
                    if task == 4:
                        number_before_conversion = float(numbers[0])
                    else:
                        number_before_conversion = float(number)
                    number_for_precision = number_before_conversion
                    prefix = org_prefix

                if (config["difficulty"] != "easy") or (task==4):
                    if number_before_conversion == 0:
                        precision = 0
                    elif (config["difficulty"] in ["manual_conversion", "easy"]) and (notation=="sci"):
                        precision = precision
                    elif (config["difficulty"] == "hard,onlyUoM") and (notation=="sci") and (task==4):
                        precision = precision
                    else:
                        precision = max(0, precision+int(-log10(float(number)/number_for_precision)))
                generated_sample = formatting_number(number, UoM, notation, precision, 
                                                        word_form=choice([True,False]) if "word" in difficulty else False, 
                                                        prefix=prefix, 
                                                        add_uom=(config['difficulty']!="uom_less"))
                for pk in sample:
                    if pk == "raw_numbers":
                        sample[pk].append(float(number_before_conversion))
                    elif isinstance(sample[pk], dict):
                        sample[pk].append(generated_sample[pk])
                    else:  
                        sample[pk].append(generated_sample[pk])

                if task in [6,7]:
                    sample["entity"] = entity.lower()
            data, label = task_wise_generator(task, method, template, tok, notation, index=index, **sample)
            data = data.strip()
            num_offset = _get_num_offset(data, sample["formatted_numbers"], notation)
        except:
            global_ignore_index.append(index)
            return None, None, None

        return data, label, num_offset

def generate_mask(tok, word):
    return " ".join([tok.mask_token]*len(tok(word, add_special_tokens=False)['input_ids']))

def task_wise_generator(task, method, template, tok, notation, measures, formatted_numbers, raw_numbers, UoMs, word_numbers, entity=None, label=None, index=None):
    if template == "paraphrase":
        template = TEMPLATE[template][task][method][index%5]
    else:
        template = TEMPLATE[template][task][method]
    if task == 1:
        if method == "bin":
            label = raw_numbers[0]>raw_numbers[1]
            sample = template.format(measures[0], measures[1])
        else:
            label = "larger" if raw_numbers[0]>raw_numbers[1] else "smaller"
            sample = template.format(measures[0], generate_mask(tok, label), measures[1])

    elif task == 2:
        sorted_measures = [x for _, x in sorted(zip(raw_numbers, measures), key=lambda pair: pair[0])]
        # Select label
        p = random()
        if p<1/3:
            pos = sorted_measures[0]
            label = "smallest"
        elif p>=2/3:
            pos = sorted_measures[-1]
            label = "largest"
        else:
            pos = choice(sorted_measures[1:-1])
            label = "middle"
        # Fill slots
        if method=='bin':
            neg_conds = ['largest', 'smallest', 'middle']
            neg_conds.remove(label)
            cond = label if random()>0.5 else choice(neg_conds)
            label = (cond==label)
            sample = template.format(cond, ",".join(measures), pos)
        else:
            sample = template.format(generate_mask(tok, label), ",".join(measures), pos)
    elif task == 3:
        asc_order = [x for _, x in sorted(zip(raw_numbers, measures), key=lambda pair: pair[0])]
        dec_order = [x for _, x in sorted(zip(raw_numbers, measures), key=lambda pair: pair[0], reverse=True)]
        # Select label
        p = random()
        if p<1/3:
            sorted_measures = asc_order
            label = "increasing"
        elif p>=2/3:
            sorted_measures = dec_order
            label = "decreasing"
        else:
            sorted_measures = measures.copy()
            label = "random"
            count = 0 
            while (np.array(asc_order) == np.array(sorted_measures)).all() or (np.array(dec_order) == np.array(sorted_measures)).all():
                shuffle(sorted_measures)
                if count>3:
                    if p<1/2:
                        sorted_measures = asc_order
                        label = "increasing"
                    elif p>=1/2:
                        sorted_measures = dec_order
                        label = "decreasing"
                    break
                count += 1
        # Fill slots
        if method=='bin':
            neg_conds = ["increasing", "decreasing", "random"]
            neg_conds.remove(label)
            cond = label if random()>0.5 else choice(neg_conds)
            label = (cond==label)
            sample = template.format(",".join(measures), cond, ",".join(sorted_measures))
        elif method=='clz':
            sample = template.format(",".join(measures), generate_mask(tok, label), ",".join(sorted_measures))
    elif task == 4:
        if method == 'clz':
            if random()>0.5:
                label = "different"
                sample = template.format(measures[0],measures[2], generate_mask(tok, label))
            else:
                label = "same"
                sample = template.format(measures[0],measures[1], generate_mask(tok, label))
    elif task == 5:
        if method =='bin':
            label = random()>0.5
            word_form = word_numbers[0] if label else word_numbers[1] 
            sample.format(measures[0], word_form)
        else:
            raise NotImplementedError("cloze task for num2word is not supported yet.")
    elif task == 6:
        if method == "bin":
            label = True if label=="normal" else False
            sample = template.format(measures[0], entity)
        else:
            nr = _get_new_bounds(full_itm2range[entity]["ref_range"], xtrapol=False)
            if (raw_numbers[0]>=nr["lb"]) and (raw_numbers[0]<=nr["ub"]):
                label = "normal"
            else:
                label = "abnormal"
            sample = template.format(measures[0], entity, generate_mask(tok, label))
    elif task == 7:
        raise NotImplementedError("Skip")
    else:
        raise NotImplementedError("Task not supported.")

    if sanity_check(sample):
        return sample, label
    else:
        global_ignore_index.append(index)
        raise ValueError("Invalid sample")

for uom_domain in config["uom_domain"]:
    for task in config["task"]:
        if (task==6) and (uom_domain=='biomedical'):
                continue
        for method in config["method"]:
            if method == 'numgen':
                global_sample_bag = []
            if (task==5) and (method=='clz'):
                continue
            for model in MODEL_LIST if method == "clz" else ["all"]:
                global_sanity_bag = {}
                global_ignore_index = list()

                if model=="all":
                    tokenizer = None
                elif not model:
                    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model)

                db = {k:{'input':list(), 'label':list(), 'number_offset': list()} for k in config["note"]} 

                for note in config["note"] if method != 'numgen' else ['word']:
                    global_idx = 0

                    if (task == 7) and (note != 'word'):
                        continue
                    for xtrapol in config["extrapolation"]:
                        for split in config["split"]:
                            # Skip train split when generating data for extrapolation
                            if xtrapol and (split=='train'):
                                continue

                            # Set generator
                            _seed = {"train":42, "valid":1234, "test":12}[split]
                            seed(_seed)

                            if method == "numgen":
                                print(f"Generating: {split}/{'xtra' if xtrapol else 'inter'}")
                                
                            local_idx = 0
                            db_size = DATA_SIZE[split]
                            with tqdm(total=db_size, desc=f"{note},{split},{'xtra' if xtrapol else 'inter'}", leave=False, disable=(method != 'numgen') and (split != "train")) as pbar:
                                while local_idx<db_size:
                                    result = sample_generator(task, method, global_idx, xtrapol, uom_domain, config["template"], note, tok=tokenizer, difficulty=config["difficulty"])                                
                                    if method == "numgen":
                                        pbar.update(1)
                                        local_idx += 1
                                        continue
                                    else:
                                        _input, _label, _num_offset = result
                                        db[note]['input'].append(_input)
                                        db[note]['label'].append(_label)
                                        db[note]['number_offset'].append(_num_offset)
                                        pbar.update(1)
                                        global_idx += 1
                                        local_idx += 1

                if method == "numgen":
                    continue

                global_ignore_index = list(set(global_ignore_index))
                start = 0
                for xtrapol in config["extrapolation"]:
                    for split in config["split"]:
                        if xtrapol and (split=='train'):
                                continue
                        for note in config["note"]:
                            end = start+DATA_SIZE[split]
                            ignore_indices = [x-start for x in global_ignore_index if (x>=start) and (x<end)]
                            cur_db_split = dict()
                            for k, v in db[note].items():
                                cur_db_split[k] = [x for i, x in enumerate(v[start:end]) if i not in ignore_indices]

                            if config["template"] == "paraphrase":
                                PARENT_DIR = os.path.join(config["template"],config["difficulty"])
                            elif config["difficulty"]!="easy":
                                PARENT_DIR = config["difficulty"]
                            else:
                                PARENT_DIR = config["template"]
                            DIR_PATH = os.path.join(PARENT_DIR,
                                                    str(task),
                                                    f"{method}-{MODEL_LIST[model]}",
                                                    split)
                            os.makedirs(DIR_PATH,exist_ok=True)
                            PATH = os.path.join(DIR_PATH,
                                                f"{uom_domain}-{note}-{'Xtra' if xtrapol else 'Inter'}.bin")
                            # Print some generated samples
                            print(colored(PATH, "red"))
                            if method == 'bin':
                                print(sum(cur_db_split['label'])/len(cur_db_split['label']))
                            elif method == 'clz':
                                stats = dict()
                                for idx in range(len(cur_db_split['input'])):
                                    if cur_db_split['label'][idx] not in stats:
                                        stats[cur_db_split['label'][idx]]=0
                                    stats[cur_db_split['label'][idx]]+=1
                                print({k:v/len(cur_db_split['label']) for k,v in stats.items()})
                            print(cur_db_split['input'][-5:])
                            print(cur_db_split['label'][-5:])
                            if 'number_offset' in cur_db_split:
                                print(cur_db_split['number_offset'][-5:])
                            print(colored(f"--------------- \t \t DB SIZE : {len(cur_db_split['input'])} \t \t -----------------", "yellow"))
                            print(colored(f"--------------- \t \t Contains Wrong Sample? : {str(np.any([x is None for x in cur_db_split['label']]))} \t \t -----------------", "blue"))
                            
                            torch.save(cur_db_split, PATH)
                        start += DATA_SIZE[split]
