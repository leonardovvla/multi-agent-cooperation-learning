"""
Based on OpenAI's PPO algorithm from spinningup
Author: Leonardo Albuquerque - ETHz, 2020

This file implements the Proximal Policy Optimization algorithm through TensorFlow

"""

# ==================================================================================================== #
# ============================================ IMPORTS =============================================== #
# ==================================================================================================== #

import numpy as np
import tensorflow as tf
import gym
import time
from copy import deepcopy
import ppo_pkg.core as core
from ppo_pkg.specs import pi_specs, v_specs
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from gym.spaces import Box, Discrete, Dict, MultiDiscrete, Tuple

import sys
sys.path.append('../ma_policy/')
from ma_policy.MA_policy import MAPolicy

# ==================================================================================================== #
# ====================================== PPO EXPERIENCE BUFFER ======================================= #
# ==================================================================================================== #


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):

        self.n_agents = 1
        self.size = size                                            #Buffer size
        self.obs_buf = {}                                           #Observations
        self.act_buf = {}                                           #Actions
        self.adv_buf = []                                           #Advantage estimations
        self.rew_buf = []                                           #Rewards
        self.ret_buf = []                                           #Returns (Rewards-to-go)
        self.val_buf = []                                           #Values (From the value network)
        self.logp_buf = []                                          #Log probabilities of actions
        self.gamma, self.lam = gamma, lam                           #Gamma and lambda are used for GAE advantage estimation
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size   #Control variables for buffer management

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store

        if self.ptr==0:
            self.obs_buf = obs
            self.act_buf = act
            self.rew_buf = rew
            self.val_buf = val
            self.logp_buf = logp

            self.n_agents = rew.size
        else:
            for k,v in obs.items():
                self.obs_buf[k] = np.vstack((self.obs_buf[k],obs[k]))
            for k,v in act.items():
                self.act_buf[k] = np.vstack((self.act_buf[k],act[k]))
            self.rew_buf = np.vstack((self.rew_buf, rew))
            self.val_buf = np.vstack((self.val_buf, val))
            self.logp_buf = np.vstack((self.logp_buf, logp))

        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)

        if self.n_agents>1:
            if last_val==0:
                last_val = [0, 0]
                rews = np.vstack((self.rew_buf[path_slice], last_val))
                vals = np.vstack((self.val_buf[path_slice], last_val))

        else:
            rews = np.append(self.rew_buf[path_slice], last_val)
            vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self, aux_vars_only = False):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """

        # buffer has to be full before you can get the data from it
        assert self.ptr == self.max_size    
        self.ptr, self.path_start_idx = 0, 0

        buf_shape = (self.size,self.n_agents)

        adv_list = np.zeros(buf_shape, dtype=np.float32)
        ret_list = np.zeros(buf_shape, dtype=np.float32)
        for i in range(self.size):
            for j in range(self.n_agents):
                adv_list[i][j] = self.adv_buf[i][j]
                ret_list[i][j] = self.ret_buf[i][j]

        # the next lines implement the advantage normalization trick
        adv_mean = np.zeros(self.n_agents, dtype=np.float32)
        adv_std = np.zeros(self.n_agents, dtype=np.float32)
        for i in range(self.n_agents):
            mean, std = mpi_statistics_scalar(np.reshape(adv_list[:,i],(self.size,1)))
            adv_mean[i] = mean
            adv_std[i] = std

        self.adv_buf = (adv_list - adv_mean) / adv_std
        self.ret_buf = ret_list

        act_data = [v for k,v in self.act_buf.items()]
        obs_data = [v for k,v in self.obs_buf.items()]

        aux_shape = (self.size*self.n_agents,1)

        if aux_vars_only:          
            return [np.reshape(self.adv_buf, aux_shape), np.reshape(self.ret_buf, aux_shape), np.reshape(self.logp_buf, aux_shape)]
        else:
            return [act_data[0], obs_data[0], np.reshape(self.adv_buf, aux_shape), np.reshape(self.ret_buf, aux_shape), np.reshape(self.logp_buf, aux_shape)]

# ==================================================================================================== #
# ====================================== PPO TRAINING FUNCTION ======================================= #
# ==================================================================================================== #

def ppo(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=33, 
        steps_per_epoch=4000, epochs=50, gamma=0.998, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=60, train_v_iters=60, lam=0.95, max_ep_len=1000,
        target_kl=0.01, logger_kwargs=dict(), save_freq=10):
    """
    Proximal Policy Optimization (by clipping), 

    with early stopping based on approximate KL

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in placeholder symbols 
            for state, ``x_ph``, and action, ``a_ph``, and returns the main 
            outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       (batch, act_dim)  | Samples actions from policy given 
                                           | states.
            ``logp``     (batch,)          | Gives log probability, according to
                                           | the policy, of taking actions ``a_ph``
                                           | in states ``x_ph``.
            ``logp_pi``  (batch,)          | Gives log probability, according to
                                           | the policy, of the action sampled by
                                           | ``pi``.
            ``v``        (batch,)          | Gives the value estimate for states
                                           | in ``x_ph``. (Critical: make sure 
                                           | to flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
            function you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while 
            still profiting (improving the objective function)? The new policy 
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`. 

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take 
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    ## Logger setup
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    ## Random seed setting
    seed += 10000 * proc_id()
    tf.set_random_seed(seed)
    np.random.seed(seed)

    ## Environment instantiation
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    #Policies vector (only one for this project)
    policies = []

    #TensorFlow session
    sess = tf.Session()

    #Build policy anc value networks
    MAP = MAPolicy(scope='policy_0', ob_space=env.observation_space, ac_space=env.action_space, network_spec=pi_specs, normalize=True, v_network_spec=v_specs)
    policies = [MAP]

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Create aux placeholders for the computation graph
    adv_ph, ret_ph, logp_old_ph = core.placeholders(1, 1, 1)

    # Get main placeholders for the computation graph
    map_phs_dict = MAP.phs  
    map_phs = [v for k,v in map_phs_dict.items()] 

    for k,v in map_phs_dict.items():
        if v.name == None:
            v.name = k             

    # Append aux and main placeholders
    # Need placeholders in *this* order later (to zip with data from buffer)
    new_phs = [adv_ph, ret_ph, logp_old_ph]
    all_phs = np.append(map_phs, new_phs)

    # Intantiate Experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam) 

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in ['policy_net', 'vpred_net'])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # PPO objectives
    ratio = tf.exp(MAP.taken_action_logp - logp_old_ph)                         # pi(a|s) / pi_old(a|s)
    min_adv = tf.where(adv_ph>0, (1+clip_ratio)*adv_ph, (1-clip_ratio)*adv_ph)  # PPO-clip limits
    pi_loss = -tf.reduce_mean(tf.minimum(ratio * adv_ph, min_adv))              # Policy loss function
    v_loss = tf.reduce_mean((ret_ph - MAP.scaled_value_tensor)**2)              # Value loss function

    # Info (useful to watch during learning)
    approx_kl = tf.reduce_mean(logp_old_ph - MAP.taken_action_logp)             # a sample estimate for KL-divergence, easy to compute
    approx_ent = tf.reduce_mean(-MAP.taken_action_logp)                         # a sample estimate for entropy, also easy to compute
    clipped = tf.logical_or(ratio > (1+clip_ratio), ratio < (1-clip_ratio))     # a logical value which states whether there was clipping
    clipfrac = tf.reduce_mean(tf.cast(clipped, tf.float32))                     # a measure of clipping for posterior analysis

    # Optimizers
    train_pi = MpiAdamOptimizer(learning_rate=pi_lr).minimize(pi_loss)          #Policy network optimizer
    train_v = MpiAdamOptimizer(learning_rate=vf_lr).minimize(v_loss)            #Value network optimizer

    #initialize TensorFlow variabels
    sess.run(tf.global_variables_initializer())

    # Sync params across processes
    sess.run(sync_all_params())

    # Set up logger variables to be saved (it is necessary to save everything that is  
    # input/output to the networks so that the policy can be played afterwards during testing)
    out_act_dict = MAP.sampled_action
    out_state_dict = MAP.state_out    
    logger_outputs = {**out_act_dict, **out_state_dict}

    for k,v in logger_outputs.items():
        if 'lstm' in k:
            logger_outputs[k + '_out'] = logger_outputs.pop(k)

    logger_inputs = map_phs_dict

    logger.setup_tf_saver(sess, inputs=logger_inputs, outputs=logger_outputs)

    # ======================================================================== #
    # ===================== Auxiliary Training Functions ===================== #
    # ======================================================================== #

    # Compute metrics for analysis during and after training
    def compute_metrics(extra_dict={}):

        loss_outs = {'pi_loss': pi_loss, 
                     'v_loss': v_loss,
                     'approx_ent': approx_ent,
                     'approx_kl': approx_kl,
                     'approx_cf': clipfrac,
                     'taken_action_logp': MAP.taken_action_logp,
                     'ratio': ratio,
                     'min_adv': min_adv}

        out_loss = policies[0].sess_run(buf.obs_buf, 
                                        sess_act=sess, 
                                        extra_feed_dict=extra_dict, 
                                        other_outputs=loss_outs, 
                                        replace=True)

        return out_loss['pi_loss'], out_loss['v_loss'], out_loss['approx_ent'], out_loss['approx_kl'], out_loss['approx_cf']

    # ======================================================================= #

    # Run session on policy and value optimizers for training their respective networks
    def train(net, extra_dict={}):

        if net == 'pi':
            train_outs = {'train_pi': train_pi, 
                          'approx_kl': approx_kl}
        elif net == 'v':
            train_outs = {'train_v': train_v}
        else:
            print("Error: Network not defined")
            return

        out_train = policies[0].sess_run(buf.obs_buf, 
                                         sess_act=sess, 
                                         extra_feed_dict=extra_dict, 
                                         other_outputs=train_outs, 
                                         replace=True)
        if net == 'pi':
            return out_train['approx_kl']

    # ======================================================================= #

    # Perform training procedure
    def update():

        print("======= update!")

        #get aux data from the buffer and match it with its respective placeholders
        buf_data = buf.get(aux_vars_only=True)
        aux_inputs = {k:v for k,v in zip(new_phs, buf_data)}

        #for the training, the actions taken during the experience loop are also inputs to the network
        extra_dict = {k:v for k,v in buf.act_buf.items() if k is not 'vpred'}

        for k,v in extra_dict.items():
            if k == 'action_movement':
                extra_dict[k] = np.expand_dims(v, 1)

        #actions and aux variables from the buffer are joined and passed to compute_metrics (observations are joined within the functions)
        extra_dict.update(aux_inputs)
        pi_l_old, v_l_old, ent, kl, cf = compute_metrics(extra_dict)

        # Policy training loop
        for i in range(train_pi_iters):
            if i%10==0:
                print("training pi iter ", i)
            kl = train('pi',extra_dict)
            kl = mpi_avg(kl)
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break

        logger.store(StopIter=i)
        print("")

        # Value training loop
        for j in range(train_v_iters):
            if j%10==0:
                print("training v iter ", j)
            train('v', extra_dict)

        # Log changes from update with a new run on compute_metrics
        pi_l_new, v_l_new, ent, kl, cf = compute_metrics(extra_dict)

        #Store information
        logger.store(LossPi=pi_l_old, LossV=v_l_old, 
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(pi_l_new - pi_l_old),
                     DeltaLossV=(v_l_new - v_l_old))

        #Reset experience varibales
        o, ep_ret, ep_len = env.reset(), 0, 0

        #Reset policy
        for policy in policies:
            policy.reset()

        print("======= update finished!")

    # ======================================================================= #

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

    # ======================================================================= #
    # ========================== Experience Loop ============================ #
    # ======================================================================= #

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        print("epoch: ", epoch)
        for t in range(local_steps_per_epoch):


            # Pass observations through networs and get action + predicted value
            if len(policies) == 1: #this project's case
                a, info = policies[0].sess_run(o, sess_act=sess)
                v_t = info['vpred']
                logp_t = info['ac_logp']
            else:
                o = splitobs(o, keepdims=False)
                ob_policy_idx = np.split(np.arange(len(o)), len(policies))
                actions = []
                for i, policy in enumerate(policies):
                    inp = itemgetter(*ob_policy_idx[i])(o)
                    inp = listdict2dictnp([inp] if ob_policy_idx[i].shape[0] == 1 else inp)
                    ac, info = policy.act(inp)
                    actions.append(ac)
                action = listdict2dictnp(actions, keepdims=True)

            # Take a step in the environment
            o2, r, d, env_info = env.step(a)
            ep_ret += r
            ep_len += 1

            # If env.render is uncommented, the experience loop is displayed (visualized) 
            # in real time (much slower, but excelent debugging)

            # env.render()

            # save experience in buffer and log
            buf.store(o, a, r, v_t, logp_t)
            logger.store(VVals=v_t)

            # Update obs (critical!)
            o = o2

            # Treat the end of a trajectory
            terminal = d or (ep_len == max_ep_len)
            if terminal or (t==local_steps_per_epoch-1) or env_info.get('discard_episode', False):
                if not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len)
                # if trajectory didn't reach terminal state, bootstrap value target
                if d:
                    last_val = 0
                else: 
                    _, info = policies[0].sess_run(o, sess_act=sess)
                    last_val = info['vpred']

                #Compute advantage estimates and rewards-to-go
                buf.finish_path(last_val)

                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)

                o, ep_ret, ep_len = env.reset(), 0, 0

                for policy in policies:
                    policy.reset()

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            print("Saved epoch: ", epoch)
            logger.save_state({'env': env}, None)

        # Perform PPO update!
        update()

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

# ==================================================================================================== #
# ==================================================================================================== #
# ==================================================================================================== #

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    ppo(lambda : gym.make(args.env), actor_critic=core.mlp_actor_critic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs)
