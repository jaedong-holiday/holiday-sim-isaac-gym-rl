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

from .the_hand_franka.the_hand_franka_insert import TheHandFrankaInsert

from holiday_sim_isaac_gym.src.modules.simulations.insert_holly_v1_sim.insert_holly_v1_sim import InsertHollyV1Sim

def resolve_the_hand_franka(cfg, *args, **kwargs):
    subtask_name: str = cfg["env"]["subtask"]
    subtask_map = dict(
        insert=TheHandFrankaInsert,        
    )

    if subtask_name not in subtask_map:
        print("!!!!!")
        raise ValueError(f"Unknown subtask={subtask_name} in {subtask_map}")

    return subtask_map[subtask_name](cfg, *args, **kwargs)

def resolve_the_hand_franka_two_arms(cfg, *args, **kwargs):
    subtask_name: str = cfg["env"]["subtask"]
    subtask_map = dict(
        reorientation=TheHandFrankaTwoArmsReorientation,        
    )

    if subtask_name not in subtask_map:
        raise ValueError(f"Unknown subtask={subtask_name} in {subtask_map}")

    return subtask_map[subtask_name](cfg, *args, **kwargs)

def resolve_insert_holly_v1(cfg, *args, **kwargs):
    subtask_name: str = "insert"
    subtask_map = dict(
        insert=InsertHollyV1Sim,        
    )

    if subtask_name not in subtask_map:
        raise ValueError(f"Unknown subtask={subtask_name} in {subtask_map}")

    print("cfg ===============================")
    print(cfg)
    print("args ==============================")
    print(args)
    print("kwargs ============================")
    print(kwargs)

    return subtask_map[subtask_name](cfg, *args, **kwargs)


# Mappings from strings to environments
isaacgym_task_map = {                
    "TheHandFranka": resolve_the_hand_franka,    
    "TheHandFrankaTwoArms": resolve_the_hand_franka_two_arms,
    "InsertHolly": resolve_insert_holly_v1, 
}
