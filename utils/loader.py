#!/usr/bin/env python
import glob
import importlib.machinery
import os
import types

import gym
import humps
from ray.rllib import MultiAgentEnv
from ray.rllib.models import ModelCatalog

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.attention_net import GTrXLNet
from ray.tune import registry

"""
Helper functions
"""


def load_class_from_file(_file_path):
    basename = os.path.basename(_file_path)
    filename = basename.replace(".py", "")
    class_name = humps.pascalize(filename)

    # TODO : Add validation here for env_name as being snake_case

    # Load the module
    loader = importlib.machinery.SourceFileLoader(filename, _file_path)
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)
    try:
        _class = getattr(mod, class_name)
    except KeyError:
        # TODO : Add a better error message
        raise Exception(
            "Looking for a class named {} in the file {}."
            "Did you name the class correctly ?".format(
                filename, class_name
            ))
    return filename, class_name, _class


"""
A loder utility, which takes an experiment directory
path, and loads necessary things into the ModelRegistry.

This imposes an opinionated directory structure on the
users, which looks something like :

- envs/
    - my_env_1.py
    - my_env_2.py
    ....
    - my_env_N.py
- models/
    - my_model_1.py
    - my_model_2.py
    .....
    - my_model_N.py
"""


def load_envs(local_dir="."):
    """
    This function takes a path to a local directory
    and looks for an `envs` folder, and imports
    all the available files in there.
    """
    for _file_path in glob.glob(os.path.join(
            local_dir, "envs", "*.py")):
        """
        Determine the filename, env_name and class_name

        # Convention :
            - filename : snake_case
            - classname : PascalCase

            the class implementation, should be an inheritance
            of gym.Env
        """
        env_name, class_name, _class = load_class_from_file(_file_path)
        env = _class
        # Validate the class
        if not issubclass(env, gym.Env) and not issubclass(env, MultiAgentEnv):
            raise Exception(
                "We expected the class named {} to be "
                "a subclass of either gym.Env or ray.rllib.MultiAgentEnv. "
                "Please read more here : https://ray.readthedocs.io/en/latest/rllib-env.html"
                .format(
                    class_name
                ))
        # Finally Register Env in Tune
        registry.register_env(env_name, lambda config: env(config))
        print("-    Successfully Loaded Environment class {} from {}".format(
            class_name, os.path.basename(_file_path)
        ))


def load_models(local_dir="."):
    """
    This function takes a path to a local directory
    and looks for a `models` folder, and imports
    all the available files in there.
    """
    for _file_path in glob.glob(os.path.join(
            local_dir, "models", "*.py")):
        """
        Determine the filename, env_name and class_name

        # Convention :
            - filename : snake_case
            - classname : PascalCase

            the class implementation, should be an inheritance
            of TFModelV2 (ModelV2 : Added PyTorch Model support too,
                          Model: Added Custom loss Model support)
        """
        model_name, class_name, _class = load_class_from_file(_file_path)
        CustomModel = _class
        # Validate the class
        if not issubclass(CustomModel, ModelV2) and not\
                issubclass(CustomModel, TFModelV2):
            raise Exception(
                "We expected the class named {} to be "
                "a subclass of TFModelV2. "
                "Please read more here : <insert-link>"
                .format(
                    class_name
                ))
        # Finally Register Model in ModelCatalog
        ModelCatalog.register_custom_model(model_name, CustomModel)
        print("-    Successfully Loaded Model class {} from {}".format(
            class_name, os.path.basename(_file_path)
        ))
    ModelCatalog.register_custom_model("GTrXLNet", GTrXLNet)