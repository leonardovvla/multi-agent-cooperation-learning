"""
Based on OpenAI's base environment from multi-agent-emergence-environments
Author: Leonardo Albuquerque - ETHz, 2020

This file describes the Multi-Agent Cooperation Learning Environment 

"""

# ==================================================================================================== #
# ============================================== IMPORTS ============================================= #
# ==================================================================================================== #

import numpy as np
import gym
import logging
from mujoco_worldgen import Floor, WorldBuilder, WorldParams, Env
from mae_envs.wrappers.multi_agent import (SplitMultiAgentActions, SplitObservations,
                                           SelectKeysWrapper)
from mae_envs.wrappers.util import (DiscretizeActionWrapper, DiscardMujocoExceptionEpisodes, SpoofEntityWrapper,
                                    AddConstantObservationsWrapper,ConcatenateObsWrapper, MaskActionWrapper)
from mae_envs.wrappers.line_of_sight import AgentAgentObsMask2D, AgentGeomObsMask2D
from mae_envs.modules.world import FloorAttributes, WorldConstants
from mae_envs.modules.agents import Agents, AgentManipulation
from mae_envs.modules.walls import RandomWalls, WallScenarios
from mae_envs.modules.objects import Boxes, Ramps
from mae_envs.modules.util import (thresh_uniform_placement, uniform_placement, center_placement,
                                   uniform_placement_middle, thresh_close_to_other_object_placement, corner_placement)
from mae_envs.wrappers.manipulation import GrabObjWrapper, GrabClosestWrapper

# ==================================================================================================== #
# ========================================== REWARD WRAPPER ========================================== #
# ==================================================================================================== #

class MACLRewardWrapper(gym.Wrapper):
    '''
        Establishes MACL dynamics. 
    '''
    def __init__(self, env, reward_scale=1.0, rew_type='SingleWalk', fs=6.):
        super().__init__(env)
        \
        self.n_agents = self.unwrapped.n_agents #Number of agents
        self.reward_scale = reward_scale        #Reward scale       
        self.rew_type = rew_type                #Reward type                      
        self.fs = fs                            #Floor size       

    def step(self, action):

        #step the environment by taking the chosen action
        obs, rew, done, info = self.env.step(action)    
        #set reward to -1 for all agents
        this_rew = -np.ones((self.n_agents,))           
        #set goal region (after the threshold line, max reward is given)
        thresh = (7.0/11.0)*self.fs                     

        # ============================== Single Walk (toy experiment) =============================== #
        #  The objective is simply for the agent to cross the room and stay close to the wall on the  #
        #    right. The reward progresses from -5 to -1 from the left wall until the threshold and    #
        #                 becomes 10 (max reward) if the agent is after the threshold.                #
        #                               |----------------------------|                                #
        #                               |                      .     |                                #
        #                               |                      .     |                                #
        #                               |   A ---------------> .     |                                #
        #                               |                      .     |                                #
        #                               |                      .     |                                #
        #                               |                      .     |                                #
        #                               |                      .     |                                #
        #                               |                      .     |                                #
        #                               |                      .     |                                #
        #                               |____________________________|                                #                           
        # =========================================================================================== #

        if(self.rew_type == 'SingleWalk'):              
            if(obs['observation_self'][0][0] > thresh): 
                this_rew[0] = 10;
            else:
                this_rew[0] = (4.0/thresh)*obs['observation_self'][0][0] - 5

        # ====================================== Single Push ======================================== #
        #   The objective is for the agent to push the box across the room and keep it close to the   #
        #   wall on the right. The reward increases quadratically with the distance from the box to   #
        #          the threshold and becomes 10 (max reward) if the box is after the threshold.       #
        #                               |----------------------------|                                #
        #                               |                      .     |                                #
        #                               |                      .     |                                #
        #                               |   A                  .     |                                #
        #                               |    \                 .     |                                #
        #                               |     `-----> B -----> .     |                                #
        #                               |                      .     |                                #
        #                               |                      .     |                                #
        #                               |                      .     |                                #
        #                               |                      .     |                                #
        #                               |____________________________|                                #                           
        # =========================================================================================== #

        elif(self.rew_type == 'SinglePush'):
            if(obs['box_obs'][0][0][0] > thresh):
                this_rew[0] = 10
            else:
                sq_dist = pow((thresh - obs['box_obs'][0][0][0]),2)/2
                this_rew[0] = -sq_dist

        # ====================================== Double Push ======================================== #
        #   The objective is for the two agents to cooperate in pushing a heavy box across the room   #
        #    and keeping it close to the wall on the right. The reward increases quadratically with   #
        #    the distance from the box to the threshold and becomes 10 if it is after the threshold.  #
        #                               |----------------------------|                                #
        #                               |                      .     |                                #
        #                               |                      .     |                                #
        #                               |   A1                 .     |                                #
        #                               |    \                 .     |                                #
        #                               |     `,----> B -----> .     |                                #
        #                               |     /                .     |                                #
        #                               |   A2                 .     |                                #
        #                               |                      .     |                                #
        #                               |                      .     |                                #
        #                               |____________________________|                                #                           
        # =========================================================================================== #

        elif(self.rew_type == 'DoublePush'):
            if(obs['box_obs'][0][0][0] > thresh):
                for i in range(len(this_rew)):
                    this_rew[i] = 10
            else:
                sq_dist = pow((thresh - obs['box_obs'][0][0][0]),2)/2
                for i in range(len(this_rew)):
                    this_rew[i] = -sq_dist

        # ====================================== Cooperate ========================================== #
        #   The objective is for the two agents to cooperate in understanding a little team puzzle:   #
        #    as before, the reward will increases quadratically as the box approaches the threshold   #
        #    and becomes maximum after it crosses it. However, this will only happen if one of the    # 
        #   agents goes to the bottom corner of the environment and stays there while the other one   #
        #    moves the box. Regardless of where the box is, if none of the agents if at the bottom    #
        #   corner, rewards go back to -1. Therefore, the agents need to split: while one takes care  #
        #  of pushing the box, the other has to go to the bottom of the room for rewards to come.     # 
        #                               |----------------------------|                                #
        #                               |                      .     |                                #
        #                               |                      .     |                                #
        #                               |   A1                 .     |                                #
        #                               |    \                 .     |                                #
        #                               |     `-----> B -----> .     |                                #
        #                               |                      .     |                                #
        #                               |                      .     |                                #
        #                               |......                .     |                                #
        #                               |     : <----- A2      .     |                                #
        #                               |_____:______________________|                                #                           
        # =========================================================================================== #

        elif(self.rew_type == 'DoubleSplit'):

            corner_agent1 = True if (obs['observation_self'][0][0]<1 and obs['observation_self'][0][1]<1) else False
            corner_agent2 = True if (obs['observation_self'][1][0]<1 and obs['observation_self'][1][1]<1) else False

            if corner_agent1 or corner_agent2:
                if(obs['box_obs'][0][0][0] > thresh):
                    for i in range(len(this_rew)):
                        this_rew[i] = 10
                else:
                    sq_dist = pow((thresh - obs['box_obs'][0][0][0]),2)/2
                    for i in range(len(this_rew)):
                        this_rew[i] = -sq_dist

        this_rew *= self.reward_scale
        rew += this_rew
        return obs, rew, done, info

# ==================================================================================================== #
# ============================ MULTI-AGENT COOPERATION LEARNING ENVIRONMENT ========================== #
# ==================================================================================================== #

class MACL(Env):
    '''
        Multi-agent MACL Environment.
        Args:
            horizon (int): Number of steps agent gets to act
            n_substeps (int): Number of internal mujoco steps per outer environment step;
                essentially this is action repeat.
            n_agents (int): number of agents in the environment
            floor_size (float): size of the floor
            grid_size (int): size of the grid that we'll use to place objects on the floor
            action_lims (float tuple): lower and upper limit of mujoco actions
            deterministic_mode (bool): if True, seeds are incremented rather than randomly sampled.
    '''
    def __init__(self, horizon=250, n_substeps=5, n_agents=1,
                 floor_size=6., grid_size=30,
                 action_lims=(-1.0, 1.0), deterministic_mode=False,
                 **kwargs):
        super().__init__(get_sim=self._get_sim,
                         get_obs=self._get_obs,
                         action_space=tuple(action_lims),
                         horizon=horizon,
                         deterministic_mode=deterministic_mode)
        self.n_agents = n_agents
        self.metadata = {}
        self.metadata['n_actors'] = n_agents
        self.horizon = horizon
        self.n_substeps = n_substeps
        self.floor_size = floor_size
        self.grid_size = grid_size
        self.kwargs = kwargs
        self.placement_grid = np.zeros((grid_size, grid_size))
        self.modules = []

    def add_module(self, module):
        self.modules.append(module)

    def _get_obs(self, sim):
        '''
            Loops through modules, calls their observation_step functions, and
                adds the result to the observation dictionary.
        '''
        obs = {}
        for module in self.modules:
            obs.update(module.observation_step(self, self.sim))
        return obs

    def _get_sim(self, seed):
        '''
            Calls build_world_step and then modify_sim_step for each module. If
            a build_world_step failed, then restarts.
        '''
        world_params = WorldParams(size=(self.floor_size, self.floor_size, 2.5),
                                   num_substeps=self.n_substeps)
        # world_params2 = WorldParams(size=(self.floor_size-2, self.floor_size-1, 5.0),
        #                            num_substeps=self.n_substeps)
        successful_placement = False
        failures = 0
        while not successful_placement:
            if (failures + 1) % 10 == 0:
                logging.warning(f"Failed {failures} times in creating environment")
            builder = WorldBuilder(world_params, seed)
            floor = Floor()

            builder.append(floor)

            self.placement_grid = np.zeros((self.grid_size, self.grid_size))

            successful_placement = np.all([module.build_world_step(self, floor, self.floor_size)
                                           for module in self.modules])
            failures += 1

        sim = builder.get_sim()

        for module in self.modules:
            module.modify_sim_step(self, sim)

        return sim


def make_env(floor_size=3, n_substeps=5, horizon=250, deterministic_mode=False, n_agents=2,
             n_boxes=1, n_ramps=0, n_elongated_boxes=0, box_size=0.5, box_only_z_rot=False,
             grab_box=True, grab_selective=False, lock_grab_radius=0.25, grab_exclusive=False,
             grab_out_of_vision=False, box_floor_friction=0.2, gravity=[0, 0, -50], box_together_radius=0.5,
             action_lims=(-0.9, 0.9), polar_obs=True, boxid_obs=True, boxsize_obs=True, additional_obs={},
             rew_type='DoubleSplit'):

    '''
        Initializes the environment.

        floor_size (float): area of the floor
        grid_size (int): size of the grid that we'll use to place objects on the floor
        n_substeps (int): Number of internal mujoco steps per outer environment step
        horizon (int): Number of steps agent gets to act
        deterministic_mode (bool): if True, seeds are incremented rather than randomly sampled.   
        n_<.> (int): number of <.> in the environment
        box_size (float): size of Boxes
        box_only_z_rot (bool): if True, allows boxes to rotate only around the z axis
        grab_box (bool): if True agents can pull on boxes
        grab_selective (bool): if True each agent can only grab one box at a time   
        lock_grab_radius (float): distance form box for a valid grab
        grab_exclusive (bool): if True, boxes can only be grabbed by one agent at a time 
        grab_out_of_vision (bool): if True, agents can pull on the boxes without having the box in their vision cone  
        box_floor_friction (float): friction between the box and the floor
        gravity (vec): gravity vector
        box_together_radius (float): radius around the agent in which boxes are spawned
        action_lims (float tuple): lower and upper limit of mujoco actions
        polar_obs (bool): if True, observation in polar coordinates
        boxid_obs (bool): if True ID of different boxes is passed as an observation to the agents
        boxsize_obs (bool): if True, size of boxes is passed as an observation to the agents
        additional_obs (dict): additional observations
        rew_type (str): name of reward function used (Cooperation, DoublePush, SinglePush or SingleWalk)
    '''

    #Create MCL environment
    env = MACL(floor_size=floor_size, n_agents=n_agents, n_substeps=n_substeps, horizon=horizon, 
               deterministic_mode=deterministic_mode)

    #Add walls (empty: only one room)
    env.add_module(WallScenarios(grid_size=30, scenario='empty', door_size=2,   
                 low_outside_walls=True))

    #If dp is set, GrabObjWrapper (the wrapper which allows agents to grab boxes has to be changed)
    dp = False                      
    if rew_type == 'DoublePush':    
        dp = True

    #Notice that the floor is divided in cells and that further on, positions are calulated in cell size
    cell_size = env.floor_size / env.grid_size  
                                                
    #Agents are placed uniformly at random before the reward threshold (since their objective is either to cross it or to push the box across it).
    agent_placement_fn = thresh_uniform_placement 
                           
    #Add agents, one green and one red, according to their placement function
    env.add_module(Agents(n_agents,                                               
                          color=[np.array((50., 250., 250., 255.)) / 255] + [np.array((250., 50., 200., 255.)) / 255],
                          placement_fn=agent_placement_fn))  

    #Boxes are placed uniformly at random before the reward threshold (since the objective is for the agents to push it across the threshold).
    box_placement_fn = thresh_uniform_placement
    # box_placement_fn = corner_placement

    #Add boxes (box_no_rot = True does not allow boxes to rotate at all)
    if n_boxes > 0:                             
        env.add_module(Boxes(n_boxes=n_boxes, 
                             placement_fn=box_placement_fn,
                             friction=box_floor_friction, polar_obs=polar_obs,
                             n_elongated_boxes=n_elongated_boxes,
                             boxid_obs=boxid_obs, boxsize_obs=boxsize_obs,
                             box_size=box_size,
                             box_only_z_rot=box_only_z_rot,
                             box_no_rot=False,
                             mark_box_corners=False,
                             box_mass=15.0))

    #Add Ramps (No ramps in MACL)
    if n_ramps > 0:                              
        env.add_module(Ramps(n_ramps=n_ramps))

    #If there is a box in the environment and agents can grab it, add manipulation capability to the agents
    if np.max(n_boxes) > 0 and grab_box:
        env.add_module(AgentManipulation())

    #Add friction to the floor
    if box_floor_friction is not None:
        env.add_module(FloorAttributes(friction=box_floor_friction))

    #Add gravity to the world
    env.add_module(WorldConstants(gravity=gravity))

    #Reset the environment
    env.reset()

    #These are the observation keys pertaining to the agents

    keys_self = ['agent_qpos_qvel']         #self position and orientation, linear and angular velocity
    keys_mask_self = ['mask_aa_obs']        #binary vector which shows for each agent which other agent it sees (present in its vision cone)
    keys_external = ['agent_qpos_qvel']     #other agents' position and orientation, linear and angular velocity form the perspective of each agent
    keys_mask_external = []                 #other masks to be added hereafter

    # ================== Wrappers =================== #

    #Adds new constant observations to the environment.
    env = AddConstantObservationsWrapper(env, new_obs=additional_obs)
    keys_external += list(additional_obs)
    keys_mask_external += [ob for ob in additional_obs if 'mask' in ob]

    #Attribute action space to each agent
    env = SplitMultiAgentActions(env)      

    #Set vision cone for each agent                               
    env = AgentAgentObsMask2D(env)   

    #Discretize Action Space into discrete bins: 3 rows (X translation, Y translation, Z rotation) of 5 "speeds" (-1 to 1) for each agent
    env = DiscretizeActionWrapper(env, 'action_movement', nbuckets=5)
    
    #Adds a mask observation that states which objects are visible to which agents.
    if np.max(n_boxes) > 0:
        env = AgentGeomObsMask2D(env, pos_obs_key='box_pos', mask_obs_key='mask_ab_obs',
                                 geom_idxs_obs_key='box_geom_idxs')
        keys_external += ['mask_ab_obs', 'box_obs']
        keys_mask_external.append('mask_ab_obs')

    #Adds wrapper for allowing boxes to be grabbed by agents
    grab_radius_multiplier = lock_grab_radius / box_size
    if grab_box and np.max(n_boxes) > 0:
        env = GrabObjWrapper(env, [f'moveable_box{i}' for i in range(n_boxes)],
                             radius_multiplier=grab_radius_multiplier,
                             grab_exclusive=grab_exclusive,                
                             obj_in_game_metadata_keys=['curr_n_boxes'],
                             dp=dp)                        

    #Separate dimensions and attribute observation space to each agent
    env = SplitObservations(env, keys_self + keys_mask_self)     

    #Create spoof masks for the value network (which receives full information of the environment)
    if n_agents == 1:
        env = SpoofEntityWrapper(env, 2, ['agent_qpos_qvel'], ['mask_aa_obs'])
    env = SpoofEntityWrapper(env, n_boxes,
                             ['box_obs'],
                             ['mask_ab_obs'])
    keys_mask_external += ['mask_ab_obs_spoof']

    #Guarantees that pull action will only take effect if in vision cone of the agent (if grab_out_of_vision==False)
    if not grab_out_of_vision and grab_box:
        env = MaskActionWrapper(env, 'action_pull', ['mask_ab_obs'])

    #Guarantees an agent will only be able to grab one box at a time
    if not grab_selective and grab_box:
        env = GrabClosestWrapper(env)

    #Formats observations
    env = ConcatenateObsWrapper(env, {'agent_qpos_qvel': ['agent_qpos_qvel'],
                                      'box_obs': ['box_obs']})

    #Filters observations to only the few interesting ones
    env = SelectKeysWrapper(env, keys_self=keys_self,
                            keys_external=keys_external,
                            keys_mask=keys_mask_self + keys_mask_external,
                            flatten=False)

    #Adds reward signals to the environment according to the reward function chosen
    env = MACLRewardWrapper(env, rew_type=rew_type, fs=floor_size)

    #Discards episodes in case errors are detected
    env = DiscardMujocoExceptionEpisodes(env)

    return env
