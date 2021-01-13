import importlib
import os

import humps
import yaml

GENERATOR_CONFIG_REGISTRY = {}
EVAL_CONFIG_REGISTRY = {}

def get_eval_config(name: str = None):
    return EVAL_CONFIG_REGISTRY[name]

def get_generator_config(name: str):
    return GENERATOR_CONFIG_REGISTRY[name]


config_folder = os.path.join(os.path.dirname(__file__), "generator_configs")
for file in os.listdir(config_folder):
    if file.endswith('.yaml') and not file.startswith('_'):
        basename = os.path.basename(file)
        filename = basename.replace(".yaml", "")

        with open(os.path.join(config_folder, file)) as f:
            GENERATOR_CONFIG_REGISTRY[filename] = yaml.safe_load(f)

        #print("-    Successfully Loaded Generator Config {} from {}".format(
        #    filename, basename
        #))

eval_config_folder = os.path.join(os.path.dirname(__file__), "eval_configs")
for file in os.listdir(eval_config_folder):
    if file.endswith('.yaml') and not file.startswith('_'):
        basename = os.path.basename(file)
        filename = basename.replace(".yaml", "")

        with open(os.path.join(eval_config_folder, file)) as f:
            EVAL_CONFIG_REGISTRY[filename] = yaml.safe_load(f)

        print("-    Successfully Loaded Evaluation Config {} from {}".format(
            filename, basename
        ))
