# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
from typing import List

import torch
from isaacgym import gymapi, gymtorch, gymutil
from torch import Tensor

from holiday_sim_isaac_gym_rl.utils.torch_jit_utils import *
from holiday_sim_isaac_gym_rl.utils.torch_jit_utils import to_torch, torch_rand_float
from holiday_sim_isaac_gym_rl.tasks.the_hand_franka.the_hand_franka_base import TheHandFrankaBase
from holiday_sim_isaac_gym_rl.tasks.the_hand_franka.the_hand_franka_utils import tolerance_curriculum, tolerance_successes_objective

class TheHandFrankaInsert(TheHandFrankaBase):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
    
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

        self.target_goal_pos      = torch.tensor([0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        self.target_goal_pos_org  = torch.tensor([0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        self.target_goal_quat     = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)
        self.target_goal_quat_org = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)
   
    def set_target(self):
        current_phase = self.phase.unsqueeze(1)        
        
        goal_offset = torch.tensor([0.0, 0.0, 0.01], device=self.device).repeat((self.num_envs, 1))
        goal_offset[:,2] -= 0.002 * self.phase              
        # goal_offset = torch.where(current_phase > 1, goal_offset * 3.0, goal_offset)            

        self.target_goal_pos      = self.obj_end_pos['holi_bottom_08']     + quat_rotate(self.obj_end_rot['holi_bottom_08'],     goal_offset)                           
        self.target_goal_pos_org  = self.obj_end_pos_org['holi_bottom_08'] + quat_rotate(self.obj_end_rot_org['holi_bottom_08'], goal_offset)       

        self.target_goal_quat     = self.obj_end_rot['holi_bottom_08']    
        self.target_goal_quat_org = self.obj_end_rot_org['holi_bottom_08']
                    
        self.goal_states[:, 0:3] = self.target_goal_pos
        self.goal_states[:, 3:7] = self.target_goal_quat
        
    def get_target(self):        
        return self.target_goal_pos, self.target_goal_quat

    def _reset_target(self, env_ids: Tensor) -> None:
        pass        

    def _extra_curriculum(self):
        self.success_tolerance, self.last_curriculum_update = tolerance_curriculum(
            self.last_curriculum_update,
            self.frame_since_restart,
            self.tolerance_curriculum_interval,
            self.prev_episode_successes,
            self.success_tolerance,
            self.initial_tolerance,
            self.target_tolerance,
            self.tolerance_curriculum_increment,
        )

    def _true_objective(self) -> Tensor:
        true_objective = tolerance_successes_objective(
            self.success_tolerance, self.initial_tolerance, self.target_tolerance, self.successes
        )
        return true_objective
