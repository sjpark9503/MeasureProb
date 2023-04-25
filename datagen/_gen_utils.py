from numpy.random import random, uniform, lognormal, choice, shuffle, seed, randint
import re

Unit2Word = {
    "g": "grams",
    "l": "litres",
    "s": "seconds",
    "m": "metres",
    "mmhg":"millimeter of mercury",
    "beat":"beat",
    "hr":"hours",
    "min":"minutes",
    "u":"units",
    "mol": "mole",
    "eq": "equivalent",
    "eq/l": "normal",
    "%": "percent",
    "tab": "tablet",
    "#": "number",
    "ratio":"ratio",
    "osm": "osmole",
    "a": "ampere",
    "k": "kelvin",
    "#/ul": "per cubic milliliter",
    "gpl":"microgram of igg antibody",
    "mpl":"microgram of igm antibody",
    "k/ul": "thousands per cubic milliliter",
    "sec": "seconds",
    "u/g/hb": "units per gram of hemoglobin",
    "#/lpf":"per low power field",
    "#/hpf":"per high power field",
    "units": "units",
    "bpm": "beats per minute",
    "iu": "international units",
    "_f": "fahrenheit",
    "_c": "celsius",

}

Unit2CommonPrefix = {
    "biomedical":{
        "m":["","c","m","u","n"],
        "a":["","m","u","n"],
        "k":["","m","u"],
        "mol":["","m","u","n"],
        'eq/l': ['/', 'm/',  'u/', 'm/m', 'm/u', 'u/m'], 
        'mol/l': ['m/', 'u/', 'n/', 'm/m', 'u/m', 'n/m', 'm/u', 'u/u', 'n/u',], 
        'g/l': ['/', 'm/', 'u/', 'm/d', '/d', 'u/d', 'n/d', '/m', 'm/m', 'u/m', 'n/m', 'p/m', '/u', 'm/u', 'u/u', 'n/u', 'p/u'], 
        'iu/l': ['/', '/m', 'm/m', 'u/m','m/','u/', '/u', 'm/u'], 
        'u/l': ['/','/m','/u', ], 
        'l/min': ['m/', '/', 'u/', 'd/'], 
        '#/l': ['/u', '/m', '/d',], 
        'k/l': ['/u', '/m', '/d',], 
        "g":["","f","m","u","n", "p"],
        "s":["","m","u","n"],
        'l': ['', 'f', 'c', 'm', 'u'], 
        'm/hr': ['m/','/','u/','c/'], 
        'l/hr': ['m/','/','u/','d/']
    },
    "general":{
        "g":["","f","m","u","n", "p"],
        "l":["","d","m","u","n"],
        "m":["","c","m","u","n"],
        "s":["","m","u","n"],
    },
}

Unit2Prefix4RefRange = {
        "m":["","c","m","u","n"],
        "a":["","m","u","n"],
        "k":["","m","u"],
        "mol":["","m","u","n"],
        'eq/l': ['/', 'm/',  'u/', 'm/m', 'm/u', 'u/m'], 
        'mol/l': ['m/', 'u/', 'n/', 'm/m', 'u/m', 'n/m', 'm/u', 'u/u', 'n/u',], 
        'g/l': ['/', 'm/', 'u/', 'm/d', '/d', 'u/d', 'n/d', '/m', 'm/m', 'u/m', 'n/m', 'p/m', '/u', 'm/u', 'u/u', 'n/u', 'p/u'], 
        'iu/l': ['/', '/m', 'm/m', 'u/m','m/','u/', '/u', 'm/u'], 
        'u/l': ['/','/m','/u', ], 
        'osm/g': ['m/k', 'm/', '/k'], 
        'l/min': ['m/', '/', 'u/', 'd/'], 
        '#/l': ['/u', '/m', '/d',], 
        'k/l': ['/u', '/m', '/d',], 
        "g":["","f","m","u","n", "p"],
        "s":["","m","u","n"],
        'l': ['', 'f', 'c', 'm', 'u'], 
        'm/hr': ['m/','/','u/','c/'], 
        'l/hr': ['m/','/','u/','d/']
}

SIprefix2Word = {
     '': "",
     'k': "kilo",
     'd': "deci",
     'c': "centi",
     'm': "milli",
     'u': "micro",
     'mc': "micro",
     'n': "nano",
     'p': "pico",
     'f': "femto"
 }

SIprefix2Value = {
     '': 1e0,
     'k': 1e3,
     'd': 1e-1,
     'c': 1e-2,
     'm': 1e-3,
     'u': 1e-6,
     'mc': 1e-6,
     'n': 1e-9,
     'p': 1e-12,
     'f': 1e-15,
     'hr': 3600,
     'min' : 60,
     'sec' : 1,
     'beat' : 1,
}

Value2SIprefix = {v:k for k,v in SIprefix2Value.items()}

def unit_conversion(uom, prefix, number, ub, lb, scenario, negative=False, include_itself=False, man_conversion=False, task=None):
    if task in [6,7]:
        pfx_list = Unit2Prefix4RefRange[uom]
    else:
        pfx_list = Unit2CommonPrefix[scenario][uom]
    pfx_to_scale = dict()
    # Convert SI prefix to scales
    for pfx in pfx_list:
        nume , *denom = pfx.split('/')
        if denom:
            scale = SIprefix2Value[nume]/SIprefix2Value[denom[0]]
        else:
            scale = SIprefix2Value[nume]
        pfx_to_scale[pfx] = scale
    cur_scale = pfx_to_scale[prefix]
    if not include_itself:
        pfx_to_scale.pop(prefix)
        
    if man_conversion: 
        pfx_to_scale = {k:cur_scale/v if cur_scale!=0 else 1 for k,v in pfx_to_scale.items()}
    else:
        pfx_to_scale = {k:cur_scale/v for k,v in pfx_to_scale.items()}
    
    if pfx_to_scale:
        converted_pfx = choice(list(pfx_to_scale.keys())) if not man_conversion else list(pfx_to_scale.keys())[0]
        pos_num = number*pfx_to_scale[converted_pfx]
        neg_num_candidates = [number*x for x in pfx_to_scale.values() if x != pfx_to_scale[converted_pfx]]
        if neg_num_candidates and negative:
            converted_number = choice(neg_num_candidates)
        elif not negative:
            converted_number = pos_num
        else:
            converted_pfx, converted_number = None, None
    else:
        converted_pfx, converted_number = None, None

    return converted_pfx, converted_number

def unit_swap(uom, prefix, number, scenario, task):
    if task in [6,7]:
        pfx_list = Unit2Prefix4RefRange[uom]
    else:
        pfx_list = Unit2CommonPrefix[scenario][uom]
    pfx_to_scale = dict()
    # Convert SI prefix to scales
    for pfx in pfx_list:
        nume , *denom = pfx.split('/')
        if denom:
            scale = SIprefix2Value[nume]/SIprefix2Value[denom[0]]
        else:
            scale = SIprefix2Value[nume]
        pfx_to_scale[pfx] = scale
    cur_scale = pfx_to_scale[prefix]
    pfx_to_scale = {k:cur_scale/v for k,v in pfx_to_scale.items()}
    
    converted_pfx = choice(list(pfx_to_scale.keys()))
    converted_number = number*pfx_to_scale[converted_pfx]

    return converted_pfx, converted_number

def parse_uom(uom):
    unit = list()
    prefix = list()
    # 1. Detect UoMs.
    for i in range(1,len(uom)+1):
        if uom[-i:] in Unit2Word:
            unit.append(uom[-i:])
            prefix.append(uom[:-i] if uom[:-i] in SIprefix2Word else '')
            break
    # None is appended if unit & prefix is not detected
    if not unit:
        # 2. Detect per(/) in uom. / usually denotes the concentration.
        uom_chunks = uom.split('/')

        # 3. Sequentially merging right most characters of each chunk until it appears in unit vocab
        for chunk in uom_chunks: 
            num_units = len(unit)
            for i in range(1,len(chunk)+1):
                if chunk[-i:] in Unit2Word:
                    unit.append(chunk[-i:])
                    prefix.append(chunk[:-i] if chunk[:-i] in SIprefix2Word else None)
                    break
            # None is appended if unit & prefix is not detected
            if len(unit) == num_units:
                raise ValueError(f"Invalid word {chunk}")
            # unit.append(None)
            # prefix.append(None)

    return unit, prefix

def add_prefix(uom):
    unit, _ = parse_uom(uom)
    new_uom = list()
    for _unit in unit:
        prefix = choice(list(SIprefix2Value.keys()))
        new_uom.append(f"{prefix}{_unit}")

    return '/'.join(new_uom)

def unit2words(uom):
    unit, prefix = parse_uom(uom)
    word_form = list()
    for _unit, _prefix in zip(unit, prefix):
        word_form.append(f'{SIprefix2Word[_prefix] if _prefix is not None else "[UNK_PFX]"}{Unit2Word[_unit] if _unit is not None else "[UNK_UNIT]"}')

    return ' per '.join(word_form)
