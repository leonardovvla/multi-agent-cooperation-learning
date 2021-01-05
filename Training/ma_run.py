
"""
Author: Leonardo Albuquerque - ETHz, 2020

File run by the user in order to execute the PPO training process on the MACL environment

"""

# ====================================================================== #
# ================================ IMPORTS ============================= #
# ====================================================================== #

from ppo_pkg.ppo import ppo
import tensorflow as tf
import gym
import time

import sys
sys.path.append('../mae_envs/envs/')
import MACL
sys.path.append('../')
from ma_policy.MA_policy import MAPolicy
from Testing.test_policy_MACL import test_ppo

# ====================================================================== #
# ============================== Functions ============================= #
# ====================================================================== #

env_fn = lambda : MACL.make_env() 						#choose desired environment

dir_str = '../Testing/exp/'								#set output directory

now_str = time.asctime(time.localtime())								#get current time stamp and set logger dict
logger_kwargs = dict(output_dir=dir_str+now_str, exp_name='MACL_ppo')	#for saving information during training

ppo(env_fn=env_fn, 										#run ppo training loop (check ppo.py for documentation)
	pi_lr=3e-4,
	vf_lr=3e-4, 
	steps_per_epoch=10000, 
	epochs=1000, 
	train_pi_iters=50, 
	train_v_iters=50, 
	logger_kwargs=logger_kwargs)

test_ppo(dir_str+now_str)								#test learned policy

