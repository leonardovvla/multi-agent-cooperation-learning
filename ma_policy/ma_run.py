# import spinup
# from spinup.user_config import DEFAULT_BACKEND
# from spinup.utils.run_utils import ExperimentGrid
# from spinup.utils.serialization_utils import convert_json
# import argparse
# import gym
# import json
# import os, subprocess, sys
# import os.path as osp
# import string
# import tensorflow as tf
# import torch
# from copy import deepcopy
# from textwrap import dedent

import sys
sys.path.append('../mae_envs/envs/')

from spinup import ppo_tf1 as ppo
import tensorflow as tf
import gym
import MKSB

env_fn = lambda : MKSB.make_env('MKSB')

ac_kwargs = dict(hidden_sizes=[64,64], activation=tf.nn.relu, clip_ratio=0.2)

logger_kwargs = dict(output_dir='./exp', exp_name='MKSB_ppo')

ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=100, logger_kwargs=logger_kwargs)