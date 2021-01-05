
"""
Author: Leonardo Albuquerque - ETHz, 2020

File run by the user in order to visualize and evaluate a trained policy

"""

from spinup.utils.test_policy_ppo import load_policy_and_env, run_policy

import sys
sys.path.append('../mae_envs/envs/')
import MACL

_, get_action, lstm = load_policy_and_env('./NoLSTM') #After ../exp/ insert the name of the output directory to be analyzed 
env = MACL.make_env()
run_policy(env, get_action, lstm=lstm, max_ep_len=150)