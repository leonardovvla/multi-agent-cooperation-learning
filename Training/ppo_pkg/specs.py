
"""
Author: Leonardo Albuquerque - ETHz, 2020

This file describes the architectures of the policy and value networks 

"""

# =========================================================================================================================================================================================================================================================================================================== #
# ======================================================= Multi-Agent Cooperation Learning ======================================================================================================================================================================================================================== #
# =========================================================================================================================================================================================================================================================================================================== #

pi_specs = [{'layer_type': 'concat', 'nodes_in': ['main', 'agent_qpos_qvel'], 'nodes_out': ['agent_qpos_qvel']}, 
            {'layer_type': 'concat', 'nodes_in': ['main', 'box_obs'], 'nodes_out': ['box_obs']}, 
            {'activation': 'relu', 'layer_name': 'dense6', 'layer_type': 'dense', 'nodes_in': ['agent_qpos_qvel', 'box_obs', 'main'], 'nodes_out': ['agent_qpos_qvel', 'box_obs', 'main'], 'units': 64}, 
            {'layer_type': 'entity_concat', 'mask_out': 'objects_mask', 'masks_in': ['mask_aa_obs', 'mask_ab_obs_spoof', None], 'nodes_in': ['agent_qpos_qvel', 'box_obs', 'main'], 'nodes_out': ['objects']}, 
            {'heads': 4, 'internal_layer_name': 'residual_sa_block8', 'layer_name': 'self-attention8', 'layer_norm': True, 'layer_type': 'residual_sa_block', 'mask': 'objects_mask', 'n_embd': 64, 'n_mlp': 1, 'nodes_in': ['objects'], 'nodes_out': ['objects'], 'post_sa_layer_norm': True}, 
            {'layer_type': 'entity_pooling', 'mask': 'objects_mask', 'nodes_in': ['objects'], 'nodes_out': ['objects_pooled']}, 
            # {'layer_type': 'entity_concat', 'nodes_in': ['agent_qpos_qvel', 'box_obs', 'main'], 'nodes_out': ['objects']}, 
            # {'heads': 4, 'internal_layer_name': 'residual_sa_block8', 'layer_name': 'self-attention8', 'layer_norm': True, 'layer_type': 'residual_sa_block', 'n_embd': 64, 'n_mlp': 1, 'nodes_in': ['objects'], 'nodes_out': ['objects'], 'post_sa_layer_norm': True}, 
            # {'layer_type': 'entity_pooling', 'nodes_in': ['objects'], 'nodes_out': ['objects_pooled']}, 
            {'layer_type': 'concat', 'nodes_in': ['main', 'objects_pooled'], 'nodes_out': ['main']}, 
            {'layer_name': 'layernorm11', 'layer_type': 'layernorm'}, 
            # {'activation': 'relu', 'layer_name': 'dense12', 'layer_type': 'dense', 'units': 64}, 
            # {'layer_name': 'layernorm13', 'layer_type': 'layernorm'}, 
            {'activation': 'relu', 'layer_name': 'dense13', 'layer_type': 'dense', 'units': 32}, 
            # {'layer_name': 'lstm14', 'layer_type': 'lstm', 'units': 32}, 
            {'layer_name': 'layernorm15', 'layer_type': 'layernorm'}]

v_specs = [{'layer_type': 'concat', 'nodes_in': ['main', 'agent_qpos_qvel'], 'nodes_out': ['agent_qpos_qvel']}, 
           {'layer_type': 'concat', 'nodes_in': ['main', 'box_obs'], 'nodes_out': ['box_obs']}, 
           {'activation': 'relu', 'layer_name': 'dense6', 'layer_type': 'dense', 'nodes_in': ['agent_qpos_qvel', 'box_obs', 'main'], 'nodes_out': ['agent_qpos_qvel', 'box_obs', 'main'], 'units': 64}, 
           {'layer_type': 'entity_concat', 'mask_out': 'objects_mask', 'masks_in': [None, 'mask_ab_obs_spoof', None], 'nodes_in': ['agent_qpos_qvel', 'box_obs', 'main'], 'nodes_out': ['objects']}, 
           {'heads': 4, 'internal_layer_name': 'residual_sa_block8', 'layer_name': 'self-attention8', 'layer_norm': True, 'layer_type': 'residual_sa_block', 'mask': 'objects_mask', 'n_embd': 64, 'n_mlp': 1, 'nodes_in': ['objects'], 'nodes_out': ['objects'], 'post_sa_layer_norm': True}, 
           {'layer_type': 'entity_pooling', 'mask': 'objects_mask', 'nodes_in': ['objects'], 'nodes_out': ['objects_pooled']}, 
           # {'layer_type': 'entity_concat', 'nodes_in': ['agent_qpos_qvel', 'box_obs', 'main'], 'nodes_out': ['objects']}, 
           # {'heads': 4, 'internal_layer_name': 'residual_sa_block8', 'layer_name': 'self-attention8', 'layer_norm': True, 'layer_type': 'residual_sa_block', 'n_embd': 64, 'n_mlp': 1, 'nodes_in': ['objects'], 'nodes_out': ['objects'], 'post_sa_layer_norm': True}, 
           # {'layer_type': 'entity_pooling', 'nodes_in': ['objects'], 'nodes_out': ['objects_pooled']}, 
           {'layer_type': 'concat', 'nodes_in': ['main', 'objects_pooled'], 'nodes_out': ['main']}, 
           {'layer_name': 'layernorm11', 'layer_type': 'layernorm'}, 
           # {'activation': 'relu', 'layer_name': 'dense12', 'layer_type': 'dense', 'units': 64}, 
           # {'layer_name': 'layernorm13', 'layer_type': 'layernorm'}, 
           {'activation': 'relu', 'layer_name': 'dense13', 'layer_type': 'dense', 'units': 32}, 
           # {'layer_name': 'lstm14', 'layer_type': 'lstm', 'units': 32}, 
           {'layer_name': 'layernorm15', 'layer_type': 'layernorm'}]

# =========================================================================================================================================================================================================================================================================================================== #
# ========================================================= Hide & Seek OpenAI =========================================================================================================================================================================================================================== #
# =========================================================================================================================================================================================================================================================================================================== #


# pi_specs = [{'activation': 'relu', 'filters': 9, 'kernel_size': 3, 'layer_type': 'circ_conv1d', 'nodes_in': ['lidar'], 'nodes_out': ['lidar']}, 
#             {'layer_type': 'flatten_outer', 'nodes_in': ['lidar'], 'nodes_out': ['lidar']}, 
#             {'layer_type': 'concat', 'nodes_in': ['main', 'lidar'], 'nodes_out': ['main']}, 
#             {'layer_type': 'concat', 'nodes_in': ['main', 'agent_qpos_qvel'], 'nodes_out': ['agent_qpos_qvel']}, 
#             {'layer_type': 'concat', 'nodes_in': ['main', 'box_obs'], 'nodes_out': ['box_obs']}, 
#             {'layer_type': 'concat', 'nodes_in': ['main', 'ramp_obs'], 'nodes_out': ['ramp_obs']}, 
#             {'layer_type': 'concat', 'nodes_in': ['main', 'construction_site_obs'], 'nodes_out': ['construction_site_obs']}, 
#             {'activation': 'relu', 'layer_name': 'dense6', 'layer_type': 'dense', 'nodes_in': ['agent_qpos_qvel', 'box_obs', 'ramp_obs', 'main', 'construction_site_obs'], 'nodes_out': ['agent_qpos_qvel', 'box_obs', 'ramp_obs', 'main', 'construction_site_obs'], 'units': 128}, 
#             {'layer_type': 'entity_concat', 'mask_out': 'objects_mask', 'masks_in': ['mask_aa_obs', 'mask_ab_obs', 'mask_ar_obs', None, 'mask_acs_obs_spoof'], 'nodes_in': ['agent_qpos_qvel', 'box_obs', 'ramp_obs', 'main', 'construction_site_obs'], 'nodes_out': ['objects']}, 
#             {'heads': 4, 'internal_layer_name': 'residual_sa_block8', 'layer_name': 'self-attention8', 'layer_norm': True, 'layer_type': 'residual_sa_block', 'mask': 'objects_mask', 'n_embd': 128, 'n_mlp': 1, 'nodes_in': ['objects'], 'nodes_out': ['objects'], 'post_sa_layer_norm': True}, 
#             {'layer_type': 'entity_pooling', 'mask': 'objects_mask', 'nodes_in': ['objects'], 'nodes_out': ['objects_pooled']}, 
#             {'layer_type': 'concat', 'nodes_in': ['main', 'objects_pooled'], 'nodes_out': ['main']}, 
#             {'layer_name': 'layernorm11', 'layer_type': 'layernorm'}, 
#             {'activation': 'relu', 'layer_name': 'dense12', 'layer_type': 'dense', 'units': 256}, 
#             {'layer_name': 'layernorm13', 'layer_type': 'layernorm'}, 
#             {'layer_name': 'lstm14', 'layer_type': 'lstm', 'units': 256}, 
#             {'layer_name': 'layernorm15', 'layer_type': 'layernorm'}]

# v_specs = [{'activation': 'relu', 'filters': 9, 'kernel_size': 3, 'layer_type': 'circ_conv1d', 'nodes_in': ['lidar'], 'nodes_out': ['lidar']}, 
#            {'layer_type': 'flatten_outer', 'nodes_in': ['lidar'], 'nodes_out': ['lidar']}, 
#            {'layer_type': 'concat', 'nodes_in': ['main', 'lidar'], 'nodes_out': ['main']}, 
#            {'layer_type': 'concat', 'nodes_in': ['main', 'agent_qpos_qvel'], 'nodes_out': ['agent_qpos_qvel']}, 
#            {'layer_type': 'concat', 'nodes_in': ['main', 'box_obs'], 'nodes_out': ['box_obs']}, 
#            {'layer_type': 'concat', 'nodes_in': ['main', 'ramp_obs'], 'nodes_out': ['ramp_obs']}, 
#            {'layer_type': 'concat', 'nodes_in': ['main', 'construction_site_obs'], 'nodes_out': ['construction_site_obs']}, 
#            {'activation': 'relu', 'layer_name': 'dense6', 'layer_type': 'dense', 'nodes_in': ['agent_qpos_qvel', 'box_obs', 'ramp_obs', 'main', 'construction_site_obs'], 'nodes_out': ['agent_qpos_qvel', 'box_obs', 'ramp_obs', 'main', 'construction_site_obs'], 'units': 128}, 
#            {'layer_type': 'entity_concat', 'mask_out': 'objects_mask', 'masks_in': [None, 'mask_ab_obs_spoof', None, None, 'mask_acs_obs_spoof'], 'nodes_in': ['agent_qpos_qvel', 'box_obs', 'ramp_obs', 'main', 'construction_site_obs'], 'nodes_out': ['objects']}, 
#            {'heads': 4, 'internal_layer_name': 'residual_sa_block8', 'layer_name': 'self-attention8', 'layer_norm': True, 'layer_type': 'residual_sa_block', 'mask': 'objects_mask', 'n_embd': 128, 'n_mlp': 1, 'nodes_in': ['objects'], 'nodes_out': ['objects'], 'post_sa_layer_norm': True}, 
#            {'layer_type': 'entity_pooling', 'mask': 'objects_mask', 'nodes_in': ['objects'], 'nodes_out': ['objects_pooled']}, 
#            {'layer_type': 'concat', 'nodes_in': ['main', 'objects_pooled'], 'nodes_out': ['main']}, 
#            {'layer_name': 'layernorm11', 'layer_type': 'layernorm'}, 
#            {'activation': 'relu', 'layer_name': 'dense12', 'layer_type': 'dense', 'units': 256}, 
#            {'layer_name': 'layernorm13', 'layer_type': 'layernorm'}, 
#            {'layer_name': 'lstm14', 'layer_type': 'lstm', 'units': 256}, 
#            {'layer_name': 'layernorm15', 'layer_type': 'layernorm'}]


# =========================================================================================================================================================================================================================================================================================================== #
# ======================================================= Blueprint Construction OpenAI ======================================================================================================================================================================================================================== #
# =========================================================================================================================================================================================================================================================================================================== #


# pi_specs = [{'layer_type': 'concat', 'nodes_in': ['main', 'agent_qpos_qvel'], 'nodes_out': ['agent_qpos_qvel']}, 
#             {'layer_type': 'concat', 'nodes_in': ['main', 'box_obs'], 'nodes_out': ['box_obs']}, 
#             {'layer_type': 'concat', 'nodes_in': ['main', 'construction_site_obs'], 'nodes_out': ['construction_site_obs']}, 
#             {'activation': 'relu', 'layer_name': 'dense6', 'layer_type': 'dense', 'nodes_in': ['agent_qpos_qvel', 'box_obs', 'main', 'construction_site_obs'], 'nodes_out': ['agent_qpos_qvel', 'box_obs', 'main', 'construction_site_obs'], 'units': 128}, 
#             {'layer_type': 'entity_concat', 'mask_out': 'objects_mask', 'masks_in': ['mask_aa_obs', 'mask_ab_obs', None, 'mask_acs_obs_spoof'], 'nodes_in': ['agent_qpos_qvel', 'box_obs', 'main', 'construction_site_obs'], 'nodes_out': ['objects']}, 
#             {'heads': 4, 'internal_layer_name': 'residual_sa_block8', 'layer_name': 'self-attention8', 'layer_norm': True, 'layer_type': 'residual_sa_block', 'mask': 'objects_mask', 'n_embd': 128, 'n_mlp': 1, 'nodes_in': ['objects'], 'nodes_out': ['objects'], 'post_sa_layer_norm': True}, 
#             {'layer_type': 'entity_pooling', 'mask': 'objects_mask', 'nodes_in': ['objects'], 'nodes_out': ['objects_pooled']}, 
#             {'layer_type': 'concat', 'nodes_in': ['main', 'objects_pooled'], 'nodes_out': ['main']}, 
#             {'layer_name': 'layernorm11', 'layer_type': 'layernorm'}, 
#             {'activation': 'relu', 'layer_name': 'dense12', 'layer_type': 'dense', 'units': 256}, 
#             {'layer_name': 'layernorm13', 'layer_type': 'layernorm'}, 
#             {'layer_name': 'lstm14', 'layer_type': 'lstm', 'units': 256}, 
#             {'layer_name': 'layernorm15', 'layer_type': 'layernorm'}]

# v_specs = [{'layer_type': 'concat', 'nodes_in': ['main', 'agent_qpos_qvel'], 'nodes_out': ['agent_qpos_qvel']}, 
#            {'layer_type': 'concat', 'nodes_in': ['main', 'box_obs'], 'nodes_out': ['box_obs']}, 
#            {'layer_type': 'concat', 'nodes_in': ['main', 'construction_site_obs'], 'nodes_out': ['construction_site_obs']}, 
#            {'activation': 'relu', 'layer_name': 'dense6', 'layer_type': 'dense', 'nodes_in': ['agent_qpos_qvel', 'box_obs', 'main', 'construction_site_obs'], 'nodes_out': ['agent_qpos_qvel', 'box_obs', 'main', 'construction_site_obs'], 'units': 128}, 
#            {'layer_type': 'entity_concat', 'mask_out': 'objects_mask', 'masks_in': [None, 'mask_ab_obs_spoof', None, 'mask_acs_obs_spoof'], 'nodes_in': ['agent_qpos_qvel', 'box_obs', 'main', 'construction_site_obs'], 'nodes_out': ['objects']}, 
#            {'heads': 4, 'internal_layer_name': 'residual_sa_block8', 'layer_name': 'self-attention8', 'layer_norm': True, 'layer_type': 'residual_sa_block', 'mask': 'objects_mask', 'n_embd': 128, 'n_mlp': 1, 'nodes_in': ['objects'], 'nodes_out': ['objects'], 'post_sa_layer_norm': True}, 
#            {'layer_type': 'entity_pooling', 'mask': 'objects_mask', 'nodes_in': ['objects'], 'nodes_out': ['objects_pooled']}, 
#            {'layer_type': 'concat', 'nodes_in': ['main', 'objects_pooled'], 'nodes_out': ['main']}, 
#            {'layer_name': 'layernorm11', 'layer_type': 'layernorm'}, 
#            {'activation': 'relu', 'layer_name': 'dense12', 'layer_type': 'dense', 'units': 256}, 
#            {'layer_name': 'layernorm13', 'layer_type': 'layernorm'}, 
#            {'layer_name': 'lstm14', 'layer_type': 'lstm', 'units': 256}, 
#            {'layer_name': 'layernorm15', 'layer_type': 'layernorm'}]
