from __future__ import absolute_import, division, print_function
import sys
import time
from datetime import datetime, date, timedelta
import json
import os

rootpath = os.path.dirname(os.path.abspath(__file__)).\
    replace("\\pgportfolio\\tools", "").replace("/pgportfolio/tools","")

try:
    unicode        # Python 2
except NameError:
    unicode = str  # Python 3


def preprocess_config(config, live=False):
    fill_default(config)
    if (live):
        config['input']['live'] = True
    modify_live_epoch(config)
#    load_consumption_vector (config)
    if sys.version_info[0] == 2:
        return byteify(config)
    else:
        return config

def polonify_pairnames (config, consumption):
    if config['input']['market'] == 'poloniex':
        return consumption
    c = {}
    for p in consumption:
        if p == 'BTCUSDT':
            q = 'USDT_BTC'
        else:
            q = 'BTC_' + p[:-3]
        c[q] = consumption[p]
    return c

def load_consumption_vector (config, index):
#    if index:
#        with open(rootpath+"/train_package/" + str(index) + "/consumptions.json") as file:
#            consumption = json.load(file)
#    else:
    with open(rootpath+"/pgportfolio/" + "consumptions.json." + config['input']['market']) as file:
        consumption = json.load(file)
    config['trading']['consumption_vector'] = polonify_pairnames (config, consumption)

def modify_live_epoch(config):
    if config['input']['live']:
        period = config['input']['global_period']
        window_size = config['input']['window_size']
#        now = int(time.time())
#        now = now - (now % period)
#        config['input']['start_date'] = time.strftime('%Y/%m/%d', now - max(86400, 3 * window_size * period))
#        last_week = date.today() - timedelta(7); # fugly hack
        today = date.today()
#        last_week = time.gmtime(time.time() - 14 * 86400)
        last_week = time.gmtime(time.time() - 30 * 86400) # a different fugly hack
        today = time.gmtime()
        config['input']['start_date'] = time.strftime('%Y/%m/%d', last_week)
        config['input']['end_date'] = time.strftime('%Y/%m/%d', today)

def fill_default(config):
    set_missing(config, "random_seed", 0)
    set_missing(config, "agent_type", "NNAgent")
    fill_layers_default(config["layers"])
    fill_input_default(config["input"])
    fill_train_config(config["training"])


def fill_train_config(train_config):
    set_missing(train_config, "fast_train", True)
    set_missing(train_config, "decay_rate", 1.0)
    set_missing(train_config, "decay_steps", 50000)
    set_missing(train_config, "batching_epochs", 1)
    set_missing(train_config, "consumption_scaling", "sqrtsqrt") # "linear" / "sqrt" / "sqrtsqrt" / "s3" / "const" / "biased"

def fill_input_default(input_config):
    set_missing(input_config, "save_memory_mode", False)
    set_missing(input_config, "portion_reversed", False)
    set_missing(input_config, "market", "poloniex")
    set_missing(input_config, "norm_method", "absolute")
    set_missing(input_config, "is_permed", False)
    set_missing(input_config, "fake_ratio", 1)
    set_missing(input_config, "live", False)
    set_missing(input_config, "net_dir", "")
    set_missing(input_config, "augment_train_set", False)

def fill_layers_default(layers):
    for layer in layers:
        if layer["type"] == "ConvLayer":
            set_missing(layer, "padding", "valid")
            set_missing(layer, "strides", [1, 1])
            set_missing(layer, "activation_function", "relu")
            set_missing(layer, "regularizer", None)
            set_missing(layer, "weight_decay", 0.0)
        elif layer["type"] == "EIIE_Dense":
            set_missing(layer, "activation_function", "relu")
            set_missing(layer, "regularizer", None)
            set_missing(layer, "weight_decay", 0.0)
        elif layer["type"] == "DenseLayer":
            set_missing(layer, "activation_function", "relu")
            set_missing(layer, "regularizer", None)
            set_missing(layer, "weight_decay", 0.0)
        elif layer["type"] == "EIIE_LSTM" or layer["type"] == "EIIE_RNN":
            set_missing(layer, "dropouts", None)
        elif layer["type"] == "EIIE_Output" or\
                layer["type"] == "Output_WithW" or\
                layer["type"] == "EIIE_Output_WithW" or\
                layer["type"] == "EIIE_Output_WithWC":
            set_missing(layer, "regularizer", None)
            set_missing(layer, "weight_decay", 0.0)
        elif layer["type"] == "BatchNormalization":
            pass
        elif layer["type"] == "ReLU":
            pass
        elif layer["type"] == "EIIE_Output_WithW_WithBN":
            set_missing(layer, "regularizer", "L2")
            set_missing(layer, "weight_decay", 5e-8)
        elif layer["type"] == "DropOut":
            pass
        elif layer["type"] == "TCNLayer":
            pass
        else:
            raise ValueError("layer name {} not supported".format(layer["type"]))


def set_missing(config, name, value):
    if name not in config:
        config[name] = value


def byteify(input):
    if isinstance(input, dict):
        return {byteify(key): byteify(value)
                for key, value in input.iteritems()}
    elif isinstance(input, list):
        return [byteify(element) for element in input]
    elif isinstance(input, unicode):
        return str(input)
    else:
        return input


def parse_time(time_string):
    return time.mktime(datetime.strptime(time_string, "%Y/%m/%d").timetuple())


def load_config(index=None, live=False):
    """
    @:param index: if None, load the default in pgportfolio;
     if a integer, load the config under train_package
    """
    if index:
        with open(rootpath+"/train_package/" + str(index) + "/net_config.json") as file:
            config = json.load(file)
    else:
        with open(rootpath+"/pgportfolio/" + "net_config.json") as file:
            config = json.load(file)
    if not 'consumption_vector' in config['trading']:
        load_consumption_vector (config, index) # now done in generate
    return preprocess_config(config, live)


def check_input_same(config1, config2):
    input1 = config1["input"]
    input2 = config2["input"]
    if input1["start_date"] != input2["start_date"]:
        return False
    elif input1["end_date"] != input2["end_date"]:
        return False
    elif input1["test_portion"] != input2["test_portion"]:
        return False
    else:
        return True

