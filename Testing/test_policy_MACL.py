
"""
Author: Leonardo Albuquerque - ETHz, 2020

File run automatically in order to visualize and evaluate a trained policy by the end of a training session

"""

from spinup.utils.test_policy_ppo import load_policy_and_env, run_policy

import sys
sys.path.append('../')
import MACL

def test_ppo(exp_dir, itr='last'):

	_, get_action, lstm = load_policy_and_env(exp_dir,itr=itr)
	env = MACL.make_env()
	run_policy(env, get_action, lstm=lstm)