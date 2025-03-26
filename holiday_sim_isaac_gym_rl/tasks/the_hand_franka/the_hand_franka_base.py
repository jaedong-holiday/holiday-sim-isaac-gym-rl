import math
import os
import tempfile
from copy import copy
from os.path import join
from typing import List, Tuple
from collections import deque
import matplotlib.pyplot as plt

from isaacgym import gymapi, gymtorch, gymutil
from torch import Tensor

from holiday_sim_isaac_gym_rl.utils import torch_jit_utils as torch_utils
from holiday_sim_isaac_gym_rl.tasks.the_hand_franka.the_hand_franka_utils import DofParameters, populate_dof_properties
from holiday_sim_isaac_gym_rl.tasks.base.vec_task import VecTask
from holiday_sim_isaac_gym_rl.utils.torch_jit_utils import *
import holiday_sim_isaac_gym_rl.tasks.the_hand_franka.the_hand_franka_control as fc

from holiday_sim_isaac_gym.api import asset_dir

class TheHandFrankaBase(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        self.frame_since_restart: int = 0 
        self.clamp_abs_observations: float = self.cfg["env"]["clampAbsObservations"]
        
        self.is_test = self.cfg["env"]["isTest"]        
        self.action_type  = self.cfg["env"]["actionType"]   
        self.action_scale = self.cfg["env"]["actionScale"]                             
        self.control_type = self.cfg["env"]["controlType"]           

        self.num_frankas = 1
        
        self.hand_type = self.cfg["env"]["handType"]
        if self.hand_type == "Gripper":
            self.franka_asset_file = self.cfg["env"]["asset"]["frankaGripper"] # franka + gripper                    
        elif self.hand_type == "TheHand":
            self.franka_asset_file = self.cfg["env"]["asset"]["frankaTheHandLeft"] # franka + the hand                    
        elif self.hand_type == "ShadowHand":
            self.franka_asset_file = self.cfg["env"]["asset"]["frankaShadowHand"] # franka + shadow hand                    
        else:        
            self.franka_asset_file = self.cfg["env"]["asset"]["frankaGripper"] # else?            

        self.asset_files = {}        
        self.asset_files['holi_top']       = self.cfg["env"]["asset"]["holiTop"]
        self.asset_files['holi_bottom_08'] = self.cfg["env"]["asset"]["holiBottom08"]
        self.asset_files['table_big']      = self.cfg["env"]["asset"]["tableBig"]
        
        self.num_objects = 3
        
        self.compliance_adjust = self.cfg["env"]["complianceAdjust"]          

        self.num_franka_actions = 7 if self.compliance_adjust else 6        # task space delta pos
        self.num_arm_dofs = 7                                               # 7 joints for franka arm        
        if self.hand_type == "Gripper":
            self.num_hand_dofs = 2                                          # 2 joints for gripper   
        elif self.hand_type == "TheHand":
            self.num_hand_dofs = 20                                         # 20 joints for the hand
        elif self.hand_type == "ShadowHand":
            self.num_hand_dofs = 24                                         # 22? joints for shadow hand
        self.num_dofs = self.num_arm_dofs + self.num_hand_dofs
        
        self.dof_params: DofParameters = DofParameters.from_cfg(self.cfg)

        self.reach_goal_bonus   = self.cfg["env"]["reachGoalBonus"]
        self.keypoint_rew_scale = self.cfg["env"]["keypointRewScale"]
        self.contact_rew_scale  = self.cfg["env"]["contactRewScale"]
        self.action_rew_scale   = self.cfg["env"]["actionRewScale"]        

        self.bonus_tolerance   = self.cfg["env"]["bonusTolerance"]
        self.initial_tolerance = self.cfg["env"]["initialTolerance"]
        self.success_tolerance = self.initial_tolerance
        self.target_tolerance  = self.cfg["env"]["targetSuccessTolerance"]
        self.tolerance_curriculum_increment = 1.0 if self.is_test else self.cfg["env"]["toleranceCurriculumIncrement"]        
        self.tolerance_curriculum_interval  = self.cfg["env"]["toleranceCurriculumInterval"]
  
        self.reset_dof_pos_noise_arm = self.cfg["env"]["resetDofPosRandomIntervalArm"]
        self.reset_dof_vel_noise_arm = self.cfg["env"]["resetDofVelRandomIntervalArm"]
        
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.success_steps: int = self.cfg["env"]["successSteps"]
        
        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize_object_dimensions = self.cfg["env"]["randomizeObjectDimensions"]
        self.object_base_size = self.cfg["env"]["objectBaseSize"]
        self.object_type = self.cfg["env"]["objectType"]
        # assert self.object_type in ["block"]

        self.obs_type = self.cfg["env"]["observationType"] # can be only "full_state"
        # if not (self.obs_type in ["full_state"]):
        #     raise Exception("Unknown type of observations!")

        self._set_full_state_size()     

        self.num_obs_dict = {"full_state": self.full_state_size,}
        self.cfg["env"]["numStates"]       = self.full_state_size
        self.cfg["env"]["numActions"]      = self.num_franka_actions
        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        
        self.cfg["device_type"] = sim_device.split(":")[0]
        self.cfg["device_id"] = int(sim_device.split(":")[1])
        self.cfg["headless"] = headless   

        super().__init__(
            config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id,
            headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render,
        )        

        if self.viewer is not None:
            cam_pos    = gymapi.Vec3(5.0, 0.0, 1.0)            
            cam_target = gymapi.Vec3(0.0, 0.0, 0.5)            
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)            
            light_intensity = gymapi.Vec3(0.5, 0.5, 0.5)            
            light_ambient   = gymapi.Vec3(0.5, 0.5, 0.5)            
            light_direction = gymapi.Vec3(1.0, 1.0, 1.0)
            self.gym.set_light_parameters(self.sim, 0, light_intensity, light_ambient, light_direction)
        
        self.phase = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.phase_size = 13                
        self.phase_insert = 5
        self.plot_sensor = False
        #=========================================================================================
        self.acquire_base_tensor()
        self.parse_controller_spec()
        self.custom_init()
        if self.plot_sensor:
            self.plot_init()
        #=========================================================================================

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)        
        self.gym.refresh_rigid_body_state_tensor(self.sim)                
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)                   
        self.gym.refresh_force_sensor_tensor(self.sim)

        ctrl_dim = self.num_dofs if self.cfg_ctrl["ctrl_type"] == "joint_space_impedance" else 7 # task space target (pos + quat) : 7        
        self.cur_eef_targets            = torch.zeros((self.num_envs, ctrl_dim), dtype=torch.float, device=self.device)
        self.cur_eef_targets_w_actions  = torch.zeros((self.num_envs, ctrl_dim), dtype=torch.float, device=self.device)                
        self.prev_eef_targets           = torch.zeros((self.num_envs, ctrl_dim), dtype=torch.float, device=self.device)                
        self.prev_eef_targets_w_actions = torch.zeros((self.num_envs, ctrl_dim), dtype=torch.float, device=self.device)                
        self.cur_obj_targets            = torch.zeros((self.num_envs, ctrl_dim), dtype=torch.float, device=self.device)
        self.cur_obj_targets_w_actions  = torch.zeros((self.num_envs, ctrl_dim), dtype=torch.float, device=self.device)
        self.prev_obj_targets           = torch.zeros((self.num_envs, ctrl_dim), dtype=torch.float, device=self.device)                          
        self.cur_hand_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
                 
        self.target_contact_force = to_torch([0, 0, 0, 0, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.torques      = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.prev_actions = torch.zeros((self.num_envs, self.num_franka_actions), dtype=torch.float, device=self.device)        
        
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.prev_episode_successes = torch.zeros_like(self.successes)
        self.episode_progress_buf = torch.zeros_like(self.successes)

        self.frame_since_restarts = torch.zeros((self.num_envs,1), dtype=torch.float, device=self.device) 
        self.initial_phase_bool   = torch.zeros((self.num_envs,1), dtype=torch.float, device=self.device)
        self.contact_termination  = torch.zeros((self.num_envs,1), dtype=torch.float, device=self.device)

        # true objective value for the whole episode, plus saving values for the previous episode
        self.true_objective = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.prev_episode_true_objective = torch.zeros_like(self.true_objective)

        self.total_successes = 0
        self.total_resets = 0

        # how many steps we were within the goal tolerance
        self.near_goal_steps = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        
        reward_keys = [                        
            "default_rew",
            "bonus_rew",
            "raw_keypoint_rew",
            "contact_rew",
            "action_rew",
        ]

        self.rewards_episode = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device) for key in reward_keys
        }

        self.last_curriculum_update = 0
        self.episode_root_state_tensors = [[] for _ in range(self.num_envs)]
        self.episode_dof_states = [[] for _ in range(self.num_envs)]

        self.eval_stats: bool = self.cfg["env"]["evalStats"]
        if self.eval_stats:
            self.last_success_step = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.success_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.total_num_resets = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.successes_count = torch.zeros(
                self.max_consecutive_successes + 1, dtype=torch.float, device=self.device
            )
            from tensorboardX import SummaryWriter

            self.eval_summary_dir = "./eval_summaries"
            # remove the old directory if it exists
            if os.path.exists(self.eval_summary_dir):
                import shutil

                shutil.rmtree(self.eval_summary_dir)
            self.eval_summaries = SummaryWriter(self.eval_summary_dir, flush_secs=3)
        
    def plot_init(self):
        # _label = ["fx", "fy", "fz", "px", "py", "pz", "tx", "ty", "tz", "rx", "ry", "rz"]
        _label = ["fx", "fy", "fz", "ax", "ay", "az"]
        self.lines0 = []
        self.lines1 = []
        plt.ion()  # Turn on interactive mode
        self.fig, self.axes = plt.subplots(2, 3, figsize=(16,8))  
        self.axes = self.axes.flatten()              
        for i, ax in enumerate(self.axes):
            line0, = ax.plot([], [], label=_label[i], color='blue')
            line1, = ax.plot([], [], label=_label[i], color='red')
            self.lines0.append(line0)        
            self.lines1.append(line1)        
            ax.set_xlim(0, 100)  # Set x-axis limits (adjust based on deque length)
            ax.set_ylim(-2.0, 2.0)   # Set y-axis limits (adjust based on expected data range)
            ax.set_xlabel("Steps")
            ax.set_ylabel("Value")
            ax.legend()

    def _set_full_state_size(self):
        num_arm_dof = self.num_arm_dofs
        num_hand_dof = self.num_hand_dofs
        num_dof = num_arm_dof + num_hand_dof

        num_actions = self.num_franka_actions

        phase_size = 1
        task_pos_quat_size = 7 # task space (pos + quat)      
        sensor_size = 6
        progress_obs_size = 1 + 1        
        reward_obs_size = 1   
        gain_size = 1    
        
        self.full_state_size = (
            
            phase_size
            
            + num_actions           # actions
            
            + num_dof               # franka : unscaled dof pos            

            + task_pos_quat_size    # franka : eef pos and quat

            + task_pos_quat_size    # franka : obj pos and quat            

            + task_pos_quat_size    # franka : eef to obj pos and quat            

            + task_pos_quat_size    # franka : target pos and quat            

            + task_pos_quat_size    # franka : obj target pos and quat            
           
            + sensor_size           # franka : force/torque sensor
            
            + gain_size             # franka : gain ratio
            
            + progress_obs_size
            + reward_obs_size    
        )

    def _normalize_quat(self, q):        
        # Clone the input tensor to avoid in-place modifications
        q = q.clone()

        # Ensure consistent quaternion representation
        # Negate rows where the fourth element (q[:, 3]) is negative
        q = torch.where(q[:, 3:4] < 0, -q, q)

        # Compute norms for all quaternions in the batch
        norms = torch.norm(q, dim=1, keepdim=True)

        # Handle near-zero norms using torch.where
        normalized_q = torch.where(
            norms < 1e-6,
            torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=q.dtype, device=q.device).expand_as(q),
            q / norms
        )
       
        return normalized_q

    def parse_controller_spec(self):
        self.cfg_ctrl = self.cfg["ctrl"]
        self.cfg_ctrl["num_envs"] = self.cfg["env"]["numEnvs"]
        self.cfg_ctrl['ik_method'] = 'dls'
        self.cfg_ctrl['gain_space'] = 'task'        
        self.cfg_ctrl['jacobian_type'] = 'geometric'
        
        if self.cfg_ctrl["ctrl_type"] == "task_space_impedance":
            self.cfg_ctrl['do_force_ctrl']    = False
            self.cfg_ctrl['do_motion_ctrl']   = True            
            self.cfg_ctrl['do_inertial_comp'] = False         

            self.cfg_ctrl['task_space_impedance']['franka']['task_prop_gains']        = torch.tensor(self.cfg_ctrl['task_space_impedance']['franka']['task_prop_gains'], device=self.device).repeat((self.cfg_ctrl['num_envs'], 1))
            self.cfg_ctrl['task_space_impedance']['franka']['task_deriv_gains']       = torch.tensor(self.cfg_ctrl['task_space_impedance']['franka']['task_deriv_gains'], device=self.device).repeat((self.cfg_ctrl['num_envs'], 1))

            self.cfg_ctrl['task_space_impedance']['gripper']['joint_prop_gains']      = torch.tensor(self.cfg_ctrl['task_space_impedance']['gripper']['joint_prop_gains'], device=self.device).repeat((self.cfg_ctrl['num_envs'], 1))
            self.cfg_ctrl['task_space_impedance']['gripper']['joint_deriv_gains']     = torch.tensor(self.cfg_ctrl['task_space_impedance']['gripper']['joint_deriv_gains'], device=self.device).repeat((self.cfg_ctrl['num_envs'], 1))

            self.cfg_ctrl['task_space_impedance']['the_hand']['joint_prop_gains']     = torch.tensor(self.cfg_ctrl['task_space_impedance']['the_hand']['joint_prop_gains'], device=self.device).repeat((self.cfg_ctrl['num_envs'], 1))
            self.cfg_ctrl['task_space_impedance']['the_hand']['joint_deriv_gains']    = torch.tensor(self.cfg_ctrl['task_space_impedance']['the_hand']['joint_deriv_gains'], device=self.device).repeat((self.cfg_ctrl['num_envs'], 1))

            self.cfg_ctrl['task_space_impedance']['shadow_hand']['joint_prop_gains']  = torch.tensor(self.cfg_ctrl['task_space_impedance']['shadow_hand']['joint_prop_gains'], device=self.device).repeat((self.cfg_ctrl['num_envs'], 1))
            self.cfg_ctrl['task_space_impedance']['shadow_hand']['joint_deriv_gains'] = torch.tensor(self.cfg_ctrl['task_space_impedance']['shadow_hand']['joint_deriv_gains'], device=self.device).repeat((self.cfg_ctrl['num_envs'], 1))
            
            self.cfg_ctrl['task_space_impedance']['motion_ctrl_axes'] = torch.tensor(self.cfg_ctrl['task_space_impedance']['motion_ctrl_axes'], device=self.device).repeat((self.cfg_ctrl['num_envs'], 1))
            
            self.task_prop_gains      = self.cfg_ctrl['task_space_impedance']['franka']['task_prop_gains']  
            self.task_prop_gains_base = self.cfg_ctrl['task_space_impedance']['franka']['task_prop_gains']  
            self.task_deriv_gains     = self.cfg_ctrl['task_space_impedance']['franka']['task_deriv_gains'] 
            self.gains_ratio          = torch.tensor(1.0, device=self.device).repeat((self.num_envs))
        
        self.f_gain = torch.tensor(self.cfg_ctrl['hybrid_force_motion']['wrench_prop_gains'][2], device=self.device).repeat((self.cfg_ctrl['num_envs'], 1))

    def acquire_base_tensor(self):
        # get gym GPU state tensors        
        _root_state   = self.gym.acquire_actor_root_state_tensor(self.sim)
        _body_state   = self.gym.acquire_rigid_body_state_tensor(self.sim)
        _dof_state    = self.gym.acquire_dof_state_tensor(self.sim)        
        _force_sensor = self.gym.acquire_force_sensor_tensor(self.sim)                
        _jacobian     = self.gym.acquire_jacobian_tensor(self.sim,    'franka')  # shape = (num envs, num_bodies, 6, num_dofs)
        _mass_matrix  = self.gym.acquire_mass_matrix_tensor(self.sim, 'franka')  # shape = (num_envs, num_bodies, num_bodies)  : collapse fixed joint = False 
            
        self.root_state   = gymtorch.wrap_tensor(_root_state)                    # [franka,        holi_top, holi_bottom_08, table] :  4 x 13
        self.body_state   = gymtorch.wrap_tensor(_body_state)                    # [franka body #, holi_top, holi_bottom_08, table] : (14 + 1 + 1 + 1) x 13
        self.dof_state    = gymtorch.wrap_tensor(_dof_state)                     # [dof] x [pos, vel]                               :  9 x 2        
        self.force_sensor = gymtorch.wrap_tensor(_force_sensor)                  # [franka_link]   x [F/T]                          :  1 x 6
        self._jacobian    = gymtorch.wrap_tensor(_jacobian)                      # [franka] x [franka body #] x [task dim] x [dof]  :  1 x 13 x 6 x 9
        self.mass_matrix  = gymtorch.wrap_tensor(_mass_matrix)                   # [franka] x [franka body #] x [franka body #]     :  1 x 9 x 9          

        self.root_pos    = self.root_state.view(self.num_envs, self.num_actors_per_env, 13)[..., 0:3]
        self.root_quat   = self.root_state.view(self.num_envs, self.num_actors_per_env, 13)[..., 3:7]
        self.root_linvel = self.root_state.view(self.num_envs, self.num_actors_per_env, 13)[..., 7:10]
        self.root_angvel = self.root_state.view(self.num_envs, self.num_actors_per_env, 13)[..., 10:13]
        
        self.body_pos    = self.body_state.view(self.num_envs, self.num_bodies_per_env, 13)[..., 0:3]
        self.body_quat   = self.body_state.view(self.num_envs, self.num_bodies_per_env, 13)[..., 3:7]
        self.body_linvel = self.body_state.view(self.num_envs, self.num_bodies_per_env, 13)[..., 7:10]
        self.body_angvel = self.body_state.view(self.num_envs, self.num_bodies_per_env, 13)[..., 10:13]
        
        self.dof_states   = self.dof_state.view(self.num_envs, -1, 2)[:, 0:self.num_dofs]
        self.dof_pos      = self.dof_states.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self.dof_vel      = self.dof_states.view(self.num_envs, self.num_dofs, 2)[..., 1]            
        self.hand_dof_pos = self.dof_pos[:, self.num_arm_dofs:self.num_arm_dofs+self.num_hand_dofs]
        self.hand_dof_vel = self.dof_vel[:, self.num_arm_dofs:self.num_arm_dofs+self.num_hand_dofs]

        self.robot_pos, self.robot_quat = {}, {}        
        self.robot_linvel, self.robot_angvel = {}, {}        
        self.jacobian, self.jacobian_tf = {}, {}        
        for link in self.robot_links:
            self.robot_pos[link]    = self.body_pos[:, self.rigid_body_id_env[link], 0:3]
            self.robot_quat[link]   = self.body_quat[:, self.rigid_body_id_env[link], 0:4]
            self.robot_linvel[link] = self.body_linvel[:, self.rigid_body_id_env[link], 0:3]
            self.robot_angvel[link] = self.body_angvel[:, self.rigid_body_id_env[link], 0:3]                                
            self.jacobian[link]     = self._jacobian[:, self.rigid_body_id_env[link]-1, 0:6, 0:7] 

        self.ft_sensor = self.force_sensor[:, :]                                
        # self.default_dof_pos = torch.tensor([0, -0.785398, 0, -2.356194, 0, 1.570796, 0.785398], dtype=torch.float, device=self.device)                        
        self.default_dof_pos = torch.tensor([0.0,  0.2196,  0.0, -1.9528, -0.0,  2.1647,  0.785398], dtype=torch.float, device=self.device)                                
        self.default_dof_vel = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float, device=self.device)

        if self.hand_type == "Gripper":
            self.robot_pos['panda_firgertip_midpoint']    = self.robot_pos['panda_fingertip_centered'].detach().clone()
            self.robot_quat['panda_firgertip_midpoint']   = self.robot_quat['panda_fingertip_centered'].detach().clone()
            self.robot_linvel['panda_firgertip_midpoint'] = self.robot_linvel['panda_fingertip_centered'].detach().clone()
            self.robot_angvel['panda_firgertip_midpoint'] = self.robot_angvel['panda_fingertip_centered'].detach().clone()
            self.jacobian['panda_firgertip_midpoint']     = 0.5 * (self.jacobian['panda_leftfinger_tip'] + self.jacobian['panda_rightfinger_tip']).detach().clone()
          
            gripper_pos = torch.tensor([0.035, 0.035], dtype=torch.float, device=self.device)
            gripper_vel = torch.tensor([0.0,  0.0],  dtype=torch.float, device=self.device)

            self.default_dof_pos = torch.cat((self.default_dof_pos, gripper_pos), dim=-1)
            self.default_dof_vel = torch.cat((self.default_dof_vel, gripper_vel), dim=-1)
        
        elif (self.hand_type == "TheHand") or (self.hand_type == "ShadowHand") :                
            hand_pos = torch.tensor([0.0]*self.num_hand_dofs, dtype=torch.float, device=self.device)
            hand_vel = torch.tensor([0.0]*self.num_hand_dofs, dtype=torch.float, device=self.device)

            self.default_dof_pos = torch.cat((self.default_dof_pos, hand_pos), dim=-1)
            self.default_dof_vel = torch.cat((self.default_dof_vel, hand_vel), dim=-1)

        self.object_pos,  self.object_linvel = {}, {}
        self.object_quat, self.object_angvel = {}, {}               
        self.object_reset_pos, self.object_reset_quat = {}, {}         
        for name in ['holi_top', 'holi_bottom_08', 'table_big']:
            self.object_pos[name]    = self.root_pos[:,    self.actor_id_env[name], 0:3]
            self.object_quat[name]   = self.root_quat[:,   self.actor_id_env[name], 0:4]
            self.object_linvel[name] = self.root_linvel[:, self.actor_id_env[name], 0:3]
            self.object_angvel[name] = self.root_angvel[:, self.actor_id_env[name], 0:3]      
        
            self.object_reset_pos[name]  = self.object_pos[name].clone()
            self.object_reset_quat[name] = self.object_quat[name].clone()     

        self.set_actor_root_state_object_indices: List[Tensor] = []

    def custom_init(self):
        self.target_goal_noise = torch.zeros((self.num_envs,3), device='cuda') 

        if self.hand_type == "Gripper":
            self.end_link = 'panda_firgertip_midpoint'            
        elif self.hand_type == "TheHand":
            self.end_link = 'PALM1'            
        elif self.hand_type == "ShadowHand":
            self.end_link = 'mfdistal'            

        self.eef_end_pos    = self.robot_pos[self.end_link]
        self.eef_end_rot    = self.robot_quat[self.end_link]                    
        self.eef_end_linvel = self.robot_linvel[self.end_link]
        self.eef_end_angvel = self.robot_angvel[self.end_link]

        self.transform_eef_to_obj = {}
        self.transform_eef_to_obj['pos']  = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.transform_eef_to_obj['quat'] = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        
        self.transform_obj_to_eef = {}
        self.transform_obj_to_eef['pos']  = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.transform_obj_to_eef['quat'] = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.obj_end_offset = {}        
        self.obj_end_offset['holi_top']       = torch.tensor([0.0, 0.0, 0.0],  dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.obj_end_offset['holi_bottom_08'] = torch.tensor([0.0, 0.0, 0.04], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.obj_end_rot,    self.obj_end_rot_org = {}, {}
        self.obj_end_pos,    self.obj_end_pos_org = {}, {}
        self.obj_end_linvel, self.obj_end_angvel  = {}, {}
        self.obj_grasp_point_pos,  self.obj_grasp_point_rot         = {}, {}
        self.obj_estimation_noise, self.obj_estimation_biased_noise = {}, {}        
        for name in ['holi_top', 'holi_bottom_08']:
            self.obj_end_rot[name]     = self.object_quat[name]
            self.obj_end_rot_org[name] = self.object_quat[name]
            self.obj_end_pos[name]     = self.object_pos[name] + quat_rotate(self.obj_end_rot[name], self.obj_end_offset[name]) 
            self.obj_end_pos_org[name] = self.object_pos[name] + quat_rotate(self.obj_end_rot[name], self.obj_end_offset[name]) 
            self.obj_end_linvel[name]  = self.object_linvel[name]
            self.obj_end_angvel[name]  = self.object_angvel[name]

            self.obj_grasp_point_pos[name] = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
            self.obj_grasp_point_rot[name] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

            self.obj_estimation_noise[name] = {}
            self.obj_estimation_noise[name]['pos'] = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
            self.obj_estimation_noise[name]['quat'] = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))          

            self.obj_estimation_biased_noise[name] = {}
            self.obj_estimation_biased_noise[name]['pos'] = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
            self.obj_estimation_biased_noise[name]['quat'] = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))          
        
        self.curriculum_level        = 200 * (0.01 - self.success_tolerance)   # level : 0 -> 1
        self.curriculum_noise        = {'pos' : 0.0, 'quat': 0.0}
        self.curriculum_biased_noise = {'pos' : 0.0, 'quat': 0.0}

    def _object_keypoint_offsets(self):
        raise NotImplementedError()

    def _load_additional_assets(self, object_asset_root, arm_y_offset: float) -> Tuple[int, int]:
        """
        returns: tuple (num_rigid_bodies, num_shapes)
        """
        return 0, 0

    def _create_additional_objects(self, env_ptr, env_idx, object_asset_idx):
        pass

    def _after_envs_created(self):
        pass

    def _extra_reset_rules(self, resets):
        return resets

    def _reset_target(self, env_ids: Tensor) -> None:
        raise NotImplementedError()

    def _extra_object_indices(self, env_ids: Tensor) -> List[Tensor]:
        return []

    def _extra_curriculum(self):
        pass
    
    def get_env_state(self):
        """
        Return serializable environment state to be saved to checkpoint.
        Can be used for stateful training sessions, i.e. with adaptive curriculums.
        """
        return dict(
            success_tolerance=self.success_tolerance,
        )

    def set_env_state(self, env_state):
        if env_state is None:
            return

        for key in self.get_env_state().keys():
            value = env_state.get(key, None)
            if value is None:
                continue

            self.__dict__[key] = value
            print(f"Loaded env state value {key}:{value}")

        print(f"Success tolerance value after loading from checkpoint: {self.success_tolerance}")

    # noinspection PyMethodOverriding
    def create_sim(self):
        self.dt = self.sim_params.dt # 1.0 / simulation hz
        self.sim_dt = 1.0/self.simulation_hz
        self.con_dt = 1.0/self.control_hz
        self.sim_per_render = int(self.simulation_hz/60.0)
        print("dt : ", self.dt)
        
        self.up_axis_idx = 2  # index of up axis: Y=1, Z=2 (same as in allegro_hand.py)

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)    

    def _create_envs(self, spacing, num_per_row):        
        self.num_bodies = {}
        self.num_shapes = {}

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../assets")        
        franka_asset = self._set_franka_asset_options(asset_root)        
        franka_poses = self._set_franka(franka_asset)        

        max_agg_bodies = self.num_bodies['franka']
        max_agg_shapes = self.num_shapes['franka']

        object_asset = {}     
        object_poses = {}
        for name in ['holi_top', 'holi_bottom_08', 'table_big']:
            object_asset[name] = self._set_object_asset_options(asset_root, name)
            object_poses[name] = self._set_object(object_asset[name], name)            
            
            max_agg_bodies += self.num_bodies[name]
            max_agg_shapes += self.num_shapes[name]

        self.actor_handles, self.actor_id_sim, self.actor_id_env = {}, {}, {}
        for name in ['franka', 'holi_top', 'holi_bottom_08', 'table_big']:
            self.actor_handles[name] = []
            self.actor_id_sim[name]  = []
            self.actor_id_env[name]  = []        
        
        self.franka_indices = torch.empty([self.num_envs, self.num_frankas], dtype=torch.long, device=self.device)
        self.object_indices = torch.empty([self.num_envs, self.num_objects], dtype=torch.long, device=self.device)
        
        self.num_actors_per_env = 0
        self.num_bodies_per_env = 0
        self.num_dofs_per_env   = 0

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3( spacing,  spacing, spacing)

        self.envs = []
        actor_count = 0

        # create env instance
        assert self.num_envs >= 1        
        for i in range(self.num_envs):            
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(env_ptr)
            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)
            
            for j, name in enumerate(['franka']):            
                franka_actor = self.gym.create_actor(env_ptr, franka_asset, franka_poses, 'franka', i, -1, 0)            
                self.franka_indices[i, j] = self.gym.get_actor_index(env_ptr, franka_actor, gymapi.DOMAIN_SIM)            
                self.gym.set_actor_dof_properties(env_ptr, franka_actor, self.dof_props)
                self.gym.enable_actor_dof_force_sensors(env_ptr, franka_actor)            
                self.actor_handles[name].append(franka_actor)
                self.actor_id_sim[name].append(actor_count)                
                actor_count += 1

            # add peg and hole             
            for j, name in enumerate(['holi_top', 'holi_bottom_08', 'table_big']):
                object_actor = self.gym.create_actor(env_ptr, object_asset[name], object_poses[name], name, i, -1, 0)
                self.object_indices[i, j] = self.gym.get_actor_index(env_ptr, object_actor, gymapi.DOMAIN_SIM)    
                self.actor_handles[name].append(object_actor)        
                self.actor_id_sim[name].append(actor_count)                
                actor_count += 1

            self.gym.end_aggregate(env_ptr)

        self.num_actors_per_env = int(actor_count / self.num_envs)           # per env
        self.num_bodies_per_env = self.gym.get_env_rigid_body_count(env_ptr) # per env
        self.num_dofs_per_env   = self.gym.get_env_dof_count(env_ptr)        # per env

        # we are not using new mass values after DR when calculating random forces applied to an object,
        # which should be ok as long as the randomization range is not too big
        # noinspection PyUnboundLocalVariable
        self.franka_indices = to_torch(self.franka_indices, dtype=torch.int32, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.int32, device=self.device)

        if self.hand_type == "Gripper":
            self.robot_links = ['panda_hand', 'panda_leftfinger', 'panda_rightfinger', 'panda_leftfinger_tip', 'panda_rightfinger_tip', 'panda_fingertip_centered']            
        elif self.hand_type == "TheHand":
            self.robot_links = ['HAND1', 'THUMB4', 'INDEX4', 'MIDDLE4', 'RING4', 'PINKY4', 'PALM1']            
        elif self.hand_type == "ShadowHand":
            self.robot_links = ['wrist', 'lfdistal', 'rfdistal', 'mfdistal', 'ffdistal', 'thdistal'] 

        self.rigid_body_id_env = {}        
        for link in self.robot_links:
            self.rigid_body_id_env[link] = self.gym.find_actor_rigid_body_index(env_ptr, franka_actor, link, gymapi.DOMAIN_ENV)                
        
        print("robot link id : ", self.rigid_body_id_env)

        for name in ['franka', 'holi_top', 'holi_bottom_08', 'table_big']:
            self.actor_id_sim[name] = torch.tensor(self.actor_id_sim[name], dtype=torch.int32, device=self.device) 
            self.actor_id_env[name] = self.gym.find_actor_index(env_ptr, name, gymapi.DOMAIN_ENV)               
        
        self.goal_states = to_torch([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], device=self.device, dtype=torch.float).repeat((self.num_envs, 1))

    def _set_franka_sensor(self, franka_asset):
        sensor_pose = gymapi.Transform()
        sensor_options = gymapi.ForceSensorProperties()
        sensor_options.enable_forward_dynamics_forces = False    # for example gravity
        sensor_options.enable_constraint_solver_forces = True    # for example contacts
        sensor_options.use_world_frame = True                    # report forces in world frame (easier to get vertical components)
        
        if self.hand_type == "Gripper":
            self.sensor_handles = self.gym.find_asset_rigid_body_index(franka_asset, 'panda_hand')       
        elif self.hand_type == "TheHand":                
            self.sensor_handles = self.gym.find_asset_rigid_body_index(franka_asset, 'HAND1')      
        elif self.hand_type == "ShadowHand":                
            self.sensor_handles = self.gym.find_asset_rigid_body_index(franka_asset, 'wrist')      
        
        self.gym.create_asset_force_sensor(franka_asset, self.sensor_handles, sensor_pose, sensor_options)
        
        self.sensor_data_size = int(0.1 * self.simulation_hz)  # 0.1(s) in 480 simulation hz                
        self.sensor_queue = {}
        for s_type in ['fx', 'fy', 'fz', 'tx', 'ty', 'tz']:                
            self.sensor_queue[s_type] = deque([torch.tensor([0.0]*self.num_envs, device=self.device)] * self.sensor_data_size, maxlen=self.sensor_data_size)                
        self.sensor_avg_data_plot = torch.zeros((100, 12), device='cuda')        

    def _set_franka(self, franka_asset):        
        num_dofs = self.gym.get_asset_dof_count(franka_asset)        
        print("franka num dofs : ", num_dofs)
        assert (
            self.num_dofs == num_dofs
        ), f"Number of DOFs in asset {franka_asset} is {num_dofs}, but {self.num_dofs} was expected"
    
        self.num_bodies['franka'] = self.gym.get_asset_rigid_body_count(franka_asset)    
        self.num_shapes['franka'] = self.gym.get_asset_rigid_shape_count(franka_asset)
        
        rigid_body_names = [self.gym.get_asset_rigid_body_name(franka_asset, i) for i in range(self.num_bodies['franka'])]        
        print(f"franka rigid bodies: {self.num_bodies, rigid_body_names}\n")
    
        self.dof_props = self.gym.get_asset_dof_properties(franka_asset)
        self.actuator_props = self.gym.get_asset_actuator_properties(franka_asset)
        
        # properties values are in urdf files except the stiffness        
        self.franka_dofs = 7
        self.franka_effort = [87, 87, 87, 87, 12, 12, 12]
        if self.hand_type == "Gripper":
            self.franka_dofs   += 2
            self.franka_effort += [20] * 2         
        elif self.hand_type == "TheHand":
            self.franka_dofs   += 20
            self.franka_effort += [20] * 20
        elif self.hand_type == "ShadowHand":
            self.franka_dofs   += 24
            self.franka_effort += [20] * 24
        self.franka_effort_max = torch.tensor(self.franka_effort, device=self.device)
        self.franka_effort_min = -self.franka_effort_max
                
        if self.control_type == "Pos":      # Mode : pos            
            self.franka_stiffness = [40.0] * self.franka_dofs
            self.franka_damping   =  [5.0] * self.franka_dofs
            self.franka_friction  =  [1.0] * self.franka_dofs
            self.franka_armature  =  [0.0] * self.franka_dofs
        elif self.control_type == "Effort": # Mode : effort            
            self.franka_stiffness =  [0.0] * self.franka_dofs
            self.franka_damping   =  [0.0] * self.franka_dofs
            self.franka_friction  =  [0.0] * self.franka_dofs            
            self.franka_armature  =  [0.0] * self.franka_dofs

        for i in range(self.num_dofs):
            if self.control_type == "Pos":
                self.dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            elif self.control_type == "Effort":
                self.dof_props['driveMode'][i] = gymapi.DOF_MODE_EFFORT
            self.dof_props['effort'][i]    = self.franka_effort[i]
            self.dof_props['damping'][i]   = self.franka_damping[i]                        
            self.dof_props['friction'][i]  = self.franka_friction[i]
            self.dof_props['armature'][i]  = self.franka_armature[i]
            self.dof_props['stiffness'][i] = self.franka_stiffness[i]    

        # joint limit        
        self.joint_dof_limits_lower = []
        self.joint_dof_limits_upper = []
        for i in range(self.num_dofs):
            self.joint_dof_limits_lower.append(self.dof_props["lower"][i])
            self.joint_dof_limits_upper.append(self.dof_props["upper"][i])
    
        self.joint_dof_limits_lower = to_torch(self.joint_dof_limits_lower, device=self.device)
        self.joint_dof_limits_upper = to_torch(self.joint_dof_limits_upper, device=self.device)

        franka_poses = gymapi.Transform()            
        franka_poses.p = gymapi.Vec3(*get_axis_params(0.0, self.up_axis_idx))
        franka_poses.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # sensor
        self._set_franka_sensor(franka_asset)

        return franka_poses

    def _set_franka_asset_options(self, asset_root):
        franka_asset_options = gymapi.AssetOptions()
        franka_asset_options.fix_base_link = True
        franka_asset_options.disable_gravity = True
        franka_asset_options.collapse_fixed_joints = False        
        franka_asset_options.flip_visual_attachments = True        
        franka_asset_options.enable_gyroscopic_forces = True
        franka_asset_options.thickness = 0.0   # default = 0.02
        franka_asset_options.density = 1000.0  # default = 1000.0
        franka_asset_options.armature = 0.01   # default = 0.0                
        
        stable_damping = True
        if stable_damping:         
            franka_asset_options.linear_damping  = 1.0               # default = 0.0;    increased to improve stability
            franka_asset_options.angular_damping = 5.0               # default = 0.5;    increased to improve stability
            franka_asset_options.max_linear_velocity  = 1.0          # default = 1000.0; reduced to prevent CUDA errors            
            franka_asset_options.max_angular_velocity = 2 * math.pi  # default = 64.0;   reduced to prevent CUDA errors
        else:
            franka_asset_options.linear_damping  = 0.0               # default = 0.0
            franka_asset_options.angular_damping = 0.5               # default = 0.5
            franka_asset_options.max_linear_velocity  = 1.0          # default = 1000.0            
            franka_asset_options.max_angular_velocity = 2 * math.pi  # default = 64.0
        
        convex_decomposition = False
        if convex_decomposition:
            franka_asset_options.vhacd_enabled = True
            franka_asset_options.vhacd_params = gymapi.VhacdParams()
            franka_asset_options.vhacd_params.resolution = 1000
            franka_asset_options.vhacd_params.max_num_vertices_per_ch = 32
            franka_asset_options.vhacd_params.min_volume_per_ch = 0.001 
        else:
            franka_asset_options.vhacd_enabled = False

        if self.physics_engine == gymapi.SIM_PHYSX:
            franka_asset_options.use_physx_armature = True

        # franka_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        franka_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT 

        franka_asset = self.gym.load_asset(self.sim, asset_root, self.franka_asset_file, franka_asset_options)
        print(f"Loaded franka asset {self.franka_asset_file} from {asset_root} \n")

        return franka_asset

    def _set_object_asset_options(self, asset_root, obj_type):
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.fix_base_link = False
        object_asset_options.disable_gravity = False
        object_asset_options.flip_visual_attachments = False        
        object_asset_options.enable_gyroscopic_forces = True     

        object_asset_options.density   = 1000.0            # default = 1000.0
        object_asset_options.armature  = 0.0               # default = 0.0
        object_asset_options.thickness = 0.0               # default = 0.02
        object_asset_options.linear_damping = 0.0          # default = 0.0
        object_asset_options.angular_damping = 0.0         # default = 0.5
        object_asset_options.max_linear_velocity = 1000.0  # default = 1000.0        
        object_asset_options.max_angular_velocity = 64.0   # default = 64.0

        if self.physics_engine == gymapi.SIM_PHYSX:
            object_asset_options.use_physx_armature = True
        
        object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE           
        
        object_asset = self.gym.load_asset(self.sim, asset_root, self.asset_files[obj_type], object_asset_options)
        print(f"Loaded {obj_type} asset {self.asset_files[obj_type]} from {asset_root} \n")

        return object_asset           

    def _set_object(self, object_asset, name):        
        self.num_bodies[name]  = self.gym.get_asset_rigid_body_count(object_asset)    
        self.num_shapes[name]  = self.gym.get_asset_rigid_shape_count(object_asset)
        rigid_body_names = [self.gym.get_asset_rigid_body_name(object_asset, i) for i in range(self.num_bodies[name])]        
        print(f"Set object {name}: {self.num_bodies[name], rigid_body_names}\n")

        object_pose = gymapi.Transform()        
        object_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)    
        object_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            
        return object_pose

    def _compute_phase(self):                        
        current_phase = self.phase.unsqueeze(1)
        
        end_obj_pos = torch.where(current_phase > self.phase_insert, self.obj_end_pos_org['holi_top'], self.obj_end_pos['holi_top'])
        goal_pos    = torch.where(current_phase > self.phase_insert, self.target_goal_pos_org,         self.goal_pos)            
        
        goal_dist   = torch.norm(end_obj_pos - goal_pos, dim=-1)      
        self.phase = torch.where(goal_dist < self.bonus_tolerance, self.phase+1, self.phase)
        self.phase = torch.clamp(self.phase, min=0, max=self.phase_size)                

    def _early_termination(self, resets):
        pass

    def _compute_resets(self):
        resets = torch.where(self.episode_progress_buf >= self.max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)
        # resets = torch.where(self.phase == (self.phase_size-1), torch.ones_like(resets), resets)
        # resets = self._early_termination(resets)
        
        self.successes = torch.where(self.phase == self.phase_size, torch.ones_like(self.successes), torch.zeros_like(self.successes))
        # self.successes = torch.where(self.episode_progress_buf >= self.max_episode_length - 1, torch.zeros_like(self.successes), self.successes)
        self.episode_progress_buf = torch.where(self.episode_progress_buf >= self.max_episode_length - 1, torch.zeros_like(self.episode_progress_buf), self.episode_progress_buf)
        # self.episode_progress_buf = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.zeros_like(self.episode_progress_buf), self.episode_progress_buf)

        return resets

    def _true_objective(self):
        raise NotImplementedError()

    def _default_reward(self):        
        current_phase = self.phase
        phase_size = torch.tensor([self.phase_size], dtype=torch.float32, device=self.device).repeat(self.num_envs)
        default_rew = (phase_size - current_phase) * -0.02

        return default_rew

    def _goal_bonus(self):
        current_phase = self.phase.unsqueeze(1)        
        
        end_obj_pos = torch.where(current_phase > self.phase_insert, self.obj_end_pos_org['holi_top'], self.obj_end_pos['holi_top'])
        goal_pos    = torch.where(current_phase > self.phase_insert, self.target_goal_pos_org,         self.goal_pos)
        goal_dist   = torch.norm(end_obj_pos - goal_pos, dim=-1)
            
        near_goal_ : Tensor = (goal_dist <= self.bonus_tolerance)            
        near_goal = near_goal_                        
        self.near_goal_steps += near_goal
        is_success = self.near_goal_steps >= self.success_steps
        near_goal_float = near_goal.float()
   
        curriculum_level = 200*(0.01 - self.success_tolerance) # 0 -> 1
        curriculum_bonus = 1.0 + 0.2*curriculum_level          # curriculum bonus 
        
        phase_bonus = 1.0        
        bonus_rew  = near_goal_float * self.reach_goal_bonus * phase_bonus * curriculum_bonus        

        current_phase = self.phase
        bonus_rew = torch.where(current_phase < self.phase_size, bonus_rew, 0)
        
        return bonus_rew, is_success
    
    def _contact_reward(self):
        
        ### force sensor
        force_components = torch.stack([
            self.sensor_avg_data['fx'],
            self.sensor_avg_data['fy'],
            self.sensor_avg_data['fz']
        ], dim=-1).squeeze(1)
        
        force_sensor = torch.norm(force_components, dim=-1)
        
        ### torque sensor
        torque_components = torch.stack([
            self.sensor_avg_data['tx'],
            self.sensor_avg_data['ty'],
            self.sensor_avg_data['tz']
        ], dim=-1).squeeze(1)
        
        torque_sensor = torch.norm(torque_components, dim=-1)
        
        # contact_size = force_sensor + torque_sensor         
        contact_size = force_sensor
        
        contact_rew = torch.where(contact_size > 0.25, (0.25 - contact_size) * 0.5 , 0) 
        contact_rew = torch.where(self.phase < self.phase_insert + 3, contact_rew, 0)
       
        # current_phase = self.phase
        # contact_rew = torch.where(current_phase < self.phase_size, contact_rew, 0)

        return contact_rew

    def _action_reward(self):
        # actions_diff = torch.norm((self.actions - self.prev_actions), dim=-1)
        actions_norm = torch.norm((self.actions), dim=-1)
        
        # actions_rew = (0 - actions_diff) * 1.0
        actions_rew = (0 - actions_norm) * 0.1
        actions_rew *= self.action_rew_scale        
        
        return actions_rew

    def _keypoint_reward(self):   
        current_phase = self.phase.unsqueeze(1)
        
        end_obj_pos   = torch.where(current_phase > self.phase_insert, self.obj_end_pos_org['holi_top'],  self.obj_end_pos['holi_top']) 
        end_obj_rot   = torch.where(current_phase > self.phase_insert, self.obj_end_rot_org['holi_top'],  self.obj_end_rot['holi_top']) 
        goal_pos      = torch.where(current_phase > self.phase_insert, self.target_goal_pos_org,  self.goal_pos) 
        goal_rot      = torch.where(current_phase > self.phase_insert, self.target_goal_quat_org, self.target_goal_quat) 
        
        goal_pos_dist = torch.norm((end_obj_pos - goal_pos), dim=-1)   
        goal_rot_dist = quat_diff_rad(end_obj_rot, goal_rot)        

        pos_rew = (0 - (goal_pos_dist)) * 10.0        
        rot_rew = (0 - (goal_rot_dist)) * 4.0        
        
        keypoint_rew  = (pos_rew + rot_rew)        
        keypoint_rew *= self.keypoint_rew_scale          
        
        return keypoint_rew

    def compute_franka_reward(self) -> Tuple[Tensor, Tensor]:        
        
        bonus_rew, is_success = self._goal_bonus()   
        action_rew   = self._action_reward()        
        default_rew  = self._default_reward() 
        contact_rew  = self._contact_reward()
        keypoint_rew = self._keypoint_reward()        
        
        bonus_rew    *= 1.0       
        default_rew  *= 0.0        
        keypoint_rew *= 1.0        
        contact_rew  *= 1.0
        action_rew   *= 0.0
       
        reward = (
            + bonus_rew
            + action_rew
            + default_rew                        
            + contact_rew
            + keypoint_rew            
        )

        self.rew_buf[:] = reward                
        self.rewards_episode["bonus_rew"] += bonus_rew
        self.rewards_episode["action_rew"] += action_rew           
        self.rewards_episode["default_rew"] += default_rew
        self.rewards_episode["contact_rew"] += contact_rew               
        self.rewards_episode["raw_keypoint_rew"] += keypoint_rew

        self.true_objective = self._true_objective()
        self.extras["true_objective"] = self.true_objective        
        self.extras["true_objective_mean"] = self.true_objective.mean() # scalars for logging
        self.extras["true_objective_min"] = self.true_objective.min()
        self.extras["true_objective_max"] = self.true_objective.max()
        self.extras["successes"] = self.prev_episode_successes.mean()

        rewards = [
            (bonus_rew, "bonus_rew"),
            (action_rew, "action_rew"),
            (default_rew, "default_rew"),                        
            (contact_rew, "contact_rew"),
            (keypoint_rew, "raw_keypoint_rew"),            
        ]        

        episode_cumulative = dict()
        for rew_value, rew_name in rewards:
            self.rewards_episode[rew_name] += rew_value
            episode_cumulative[rew_name] = rew_value           
        self.extras["rewards_episode"] = self.rewards_episode
        self.extras["episode_cumulative"] = episode_cumulative              

        return self.rew_buf, is_success

    def _eval_stats(self, is_success: Tensor) -> None:        
        if self.eval_stats:
            frame: int = self.frame_since_restart
            n_frames = torch.empty_like(self.last_success_step).fill_(frame)
            self.success_time = torch.where(is_success, n_frames - self.last_success_step, self.success_time)
            self.last_success_step = torch.where(is_success, n_frames, self.last_success_step)
            mask_ = self.success_time > 0
            if any(mask_):
                avg_time_mean = ((self.success_time * mask_).sum(dim=0) / mask_.sum(dim=0)).item()
            else:
                avg_time_mean = math.nan

            self.total_resets = self.total_resets + self.reset_buf.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()
            self.total_num_resets += self.reset_buf

            reset_ids = self.reset_buf.nonzero().squeeze()
            last_successes = self.successes[reset_ids].long()
            self.successes_count[last_successes] += 1

            if frame % 100 == 0:
                # The direct average shows the overall result more quickly, but slightly undershoots long term
                # policy performance.
                print(f"Max num successes: {self.successes.max().item()}")
                print(f"Min num successes: {self.successes.min().item()}")
                print(f"Average consecutive successes: {self.prev_episode_successes.mean().item():.2f}")
                # print(f"Total num resets: {self.total_num_resets.sum().item()} --> {self.total_num_resets}")
                # print(f"Reset percentage: {(self.total_num_resets > 0).sum() / self.num_envs:.2%}")
                # print(f"Last ep successes: {self.prev_episode_successes.mean().item():.2f}")
                # print(f"Last ep true objective: {self.prev_episode_true_objective.mean().item():.2f}")

                self.eval_summaries.add_scalar("last_ep_successes", self.prev_episode_successes.mean().item(), frame)
                self.eval_summaries.add_scalar(
                    "last_ep_true_objective", self.prev_episode_true_objective.mean().item(), frame
                )
                self.eval_summaries.add_scalar(
                    "reset_stats/reset_percentage", (self.total_num_resets > 0).sum() / self.num_envs, frame
                )
                self.eval_summaries.add_scalar("reset_stats/min_num_resets", self.total_num_resets.min().item(), frame)

                self.eval_summaries.add_scalar("policy_speed/avg_success_time_frames", avg_time_mean, frame)
                # frame_time = self.control_freq_inv * self.dt
                frame_time = self.dt
                self.eval_summaries.add_scalar(
                    "policy_speed/avg_success_time_seconds", avg_time_mean * frame_time, frame
                )
                self.eval_summaries.add_scalar(
                    "policy_speed/avg_success_per_minute", 60.0 / (avg_time_mean * frame_time), frame
                )
                print(f"Policy speed (successes per minute): {60.0 / (avg_time_mean * frame_time):.2f}")

                # create a matplotlib bar chart of the self.successes_count
                import matplotlib.pyplot as plt

                plt.bar(list(range(self.max_consecutive_successes + 1)), self.successes_count.cpu().numpy())
                plt.title("Successes histogram")
                plt.xlabel("Successes")
                plt.ylabel("Frequency")
                plt.savefig(f"{self.eval_summary_dir}/successes_histogram.png")
                plt.clf()

    def pose_estimate(self): 
        if self.frame_since_restart % 2 == 0:    
            self.curriculum_level = 200*(0.01 - self.success_tolerance)   # level : 0 -> 1

            self.curriculum_noise['pos'] = 0.000 + 0.001*self.curriculum_level
            # self.curriculum_noise['pos'] = 0.0        

            self.curriculum_noise['quat'] = 0.000 + 0.04*self.curriculum_level
            # self.curriculum_noise['quat'] = 0.0

            is_hard = False
            cur_noise = {}
            for name in ['holi_top', 'holi_bottom_08']:
                cur_noise[name] = {}
                cur_noise[name]['pos'] = 2.0 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)
                if is_hard:
                    beta_dist = torch.distributions.Beta(0.5, 0.5)        
                    beta_samples = beta_dist.sample((self.num_envs, 3)).to(self.device)
                    cur_noise[name]['pos'] = 2.0 * (beta_samples - 0.5)

                cur_noise[name]['pos'] *= self.curriculum_noise['pos']

                # orientation noise
                obj_euler_xyz_noise = self.curriculum_noise['quat'] * 2.0 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)            
                cur_noise[name]['quat'] = torch_utils.quat_from_euler_xyz(
                    obj_euler_xyz_noise[:,0],
                    obj_euler_xyz_noise[:,1],
                    obj_euler_xyz_noise[:,2],
                )

                # add biased orientation noise
                cur_noise[name]['quat'] = quat_mul(cur_noise[name]['quat'], self.obj_estimation_biased_noise[name]['quat'])

                # estimation error 
                # pos : (biased noise + current noise)
                # ori : (current noise * biased noise)
                self.obj_estimation_noise[name]['pos']  = self.obj_estimation_biased_noise[name]['pos'] + cur_noise[name]['pos']             
                self.obj_estimation_noise[name]['quat'] = cur_noise[name]['quat']            

            half_offset = {}
            half_offset['holi_top']       = torch.tensor([0.0, 0.0, 0.0],  dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
            half_offset['holi_bottom_08'] = torch.tensor([0.0, 0.0, 0.02], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))        
            for name in ['holi_top', 'holi_bottom_08']:  
                self.obj_end_rot[name]     = self._normalize_quat(quat_mul(self.obj_estimation_noise[name]['quat'], self.object_quat[name]))
                self.obj_end_rot_org[name] = self._normalize_quat(self.object_quat[name])
                self.obj_end_pos[name]     = self.object_pos[name] + quat_rotate(self.obj_end_rot_org[name], half_offset[name]+self.obj_estimation_noise[name]['pos']) + quat_rotate(self.obj_end_rot[name], half_offset[name]) 
                self.obj_end_pos_org[name] = self.object_pos[name] + quat_rotate(self.obj_end_rot_org[name], self.obj_end_offset[name]) 
                self.obj_end_linvel[name]  = self.object_linvel[name]
                self.obj_end_angvel[name]  = self.object_angvel[name]
            
            self.transform_eef_to_obj['pos']  = self.obj_end_pos['holi_top'] - self.eef_end_pos
            self.transform_eef_to_obj['quat'] = self._normalize_quat(quat_mul(self.obj_end_rot['holi_top'], quat_conjugate(self.eef_end_rot)))            

            self.transform_obj_to_eef['pos']  = self.eef_end_pos - self.obj_end_pos['holi_top']            
            self.transform_obj_to_eef['quat'] = self._normalize_quat(quat_mul(self.eef_end_rot, quat_conjugate(self.obj_end_rot['holi_top'])))            
    
    def compute_observations(self) -> Tuple[Tensor, int]:       
        
        self.end_obj_rel_goal_pos  = quat_rotate_inverse(self.obj_end_rot['holi_top'], self.obj_end_pos['holi_top'] - self.goal_pos)
        self.end_obj_rel_goal_quat = self._normalize_quat(quat_mul(self.goal_rot, quat_conjugate(self.obj_end_rot['holi_top'])))                    
       
        self.pose_estimate()

        if self.obs_type == "full_state":
            full_state_size, reward_obs_ofs = self.compute_full_state(self.obs_buf)
            assert (
                full_state_size == self.full_state_size
            ), f"Expected full state size {self.full_state_size}, actual: {full_state_size}"

            return self.obs_buf, reward_obs_ofs
        else:
            raise ValueError("Unkown observations type!")

    def compute_full_state(self, buf: Tensor) -> Tuple[int, int]:
        num_franka_dofs = self.num_dofs       # arm + hand dofs        
        num_actions = self.num_franka_actions          
        ofs: int = 0

        # current phase
        buf[:, ofs : ofs + 1] = self.phase.unsqueeze(-1)/40.0
        ofs += 1 

        # actions
        buf[:, ofs : ofs + num_actions] = self.actions
        ofs += num_actions

        # # actions prev
        # buf[:, ofs : ofs + num_actions] = self.prev_actions
        # ofs += num_actions

        # dof positions
        buf[:, ofs : ofs + num_franka_dofs] = unscale(
            self.dof_pos[:, :num_franka_dofs],
            self.joint_dof_limits_lower[:num_franka_dofs],
            self.joint_dof_limits_upper[:num_franka_dofs],
        )
        ofs += num_franka_dofs

        # torques for franka0
        # buf[:, ofs : ofs + num_poly_franka_dofs] = unscale(
        #     self.torques[:, :num_poly_franka_dofs],
        #     self.franka_effort_min[:num_poly_franka_dofs],
        #     self.franka_effort_max[:num_poly_franka_dofs],
        # )               
        # ofs += num_poly_franka_dofs      
                
        # torques for franka1
        # buf[:, ofs : ofs + num_hole_franka_dofs] = unscale(
        #     self.torques[:, num_poly_franka_dofs:num_poly_franka_dofs+num_hole_franka_dofs],
        #     self.franka_effort_min[:num_hole_franka_dofs],
        #     self.franka_effort_max[:num_hole_franka_dofs],
        # )               
        # ofs += num_hole_franka_dofs        

        # # dof velocities
        # buf[:, ofs : ofs + num_poly_franka_dofs] = self.franka0_dof_vel[:, :num_poly_franka_dofs]
        # ofs += num_poly_franka_dofs

        # # dof velocities
        # buf[:, ofs : ofs + num_hole_franka_dofs] = self.franka1_dof_vel[:, :num_hole_franka_dofs]
        # ofs += num_hole_franka_dofs

        # buf[:, ofs : ofs + 10] = self.franka0_poly_state[:, 3:13]        
        # ofs += 10

        # buf[:, ofs : ofs + 10] = self.franka1_hole_state[:, 3:13]        
        # ofs += 10       

        # franka : ee pos, ee quat
        buf[:, ofs : ofs + 3] = self.robot_pos[  self.end_link]
        ofs += 3
        buf[:, ofs : ofs + 4] = self.robot_quat[ self.end_link]
        ofs += 4

        # franka : obj pos, obj quat
        buf[:, ofs : ofs + 3] = self.obj_end_pos['holi_top']
        ofs += 3
        buf[:, ofs : ofs + 4] = self.obj_end_rot['holi_top']
        ofs += 4

        # franka : relative pose (ee -> obj)
        buf[:, ofs : ofs + 3] = self.transform_eef_to_obj['pos']
        ofs += 3
        buf[:, ofs : ofs + 4] = self.transform_eef_to_obj['quat']
        ofs += 4

        # franka : task space target (xyz + quaternion)
        buf[:, ofs : ofs + 7] = self.cur_eef_targets
        ofs += 7
      
        # franka : task space obj target (xyz + quaternion)
        buf[:, ofs : ofs + 7] = self.cur_obj_targets
        ofs += 7
      
        # franka0 : task space target - ee 
        # buf[:, ofs : ofs + 3] = self.franka0_cur_targets[:, 0:3] - self.franka0_poly_end_pos
        # ofs += 3

        # franka1 : task space target - ee 
        # buf[:, ofs : ofs + 3] = self.franka1_cur_targets[:, 0:3] - self.franka1_hole_end_pos
        # ofs += 3       

        # franka0 : task space target - ee (local)
        # buf[:, ofs : ofs + 3] = self.franka0_poly_end_rel_goal_pos
        # ofs += 3

        # # franka1 : task space target - ee (local)
        # buf[:, ofs : ofs + 3] = self.franka1_hole_end_rel_goal_pos
        # ofs += 3     

        # # franka0 : quat target - ee (local)
        # buf[:, ofs : ofs + 4] = self.franka0_poly_end_rel_goal_quat
        # ofs += 4

        # # franka0 : quat target - ee (local)
        # buf[:, ofs : ofs + 4] = self.franka1_hole_end_rel_goal_quat
        # ofs += 4        

        # # franka0 : distance (target - ee)
        # buf[:, ofs : ofs + 1] = torch.norm((self.franka0_poly_end_pos - self.franka0_goal_pos))
        # ofs += 1        

        # # franka1 : distance (target - ee)
        # buf[:, ofs : ofs + 1] = torch.norm((self.franka1_hole_end_pos - self.franka1_goal_pos))
        # ofs += 1
        
        # franka : Force/Torque sensor
        for i, s_type in enumerate(['fx', 'fy', 'fz', 'tx', 'ty', 'tz']):
            buf[:, ofs + i : ofs + i + 1] = self.sensor_avg_data[s_type]            
        ofs += 6               

        # franka : gain ratio
        buf[:, ofs : ofs + 1] = (self.gains_ratio-1).unsqueeze(-1)
        ofs += 1   

        # this should help the critic predict the future rewards better and anticipate the episode termination
        buf[:, ofs : ofs + 1] = torch.log(self.progress_buf / 10 + 1).unsqueeze(-1)
        ofs += 1
        buf[:, ofs : ofs + 1] = torch.log(self.successes + 1).unsqueeze(-1)
        ofs += 1

        # state_str = [f"{state.item():.3f}" for state in buf[0, : self.full_state_size]]
        # print(' '.join(state_str))

        # this is where we will add the reward observation
        reward_obs_ofs = ofs
        ofs += 1

        assert ofs == self.full_state_size
        return ofs, reward_obs_ofs
    
    def clamp_obs(self, obs_buf: Tensor) -> None:
        if self.clamp_abs_observations > 0:
            obs_buf.clamp_(-self.clamp_abs_observations, self.clamp_abs_observations)

    def get_random_quat(self, env_ids):
        # https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py
        # https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L261

        uvw = torch_rand_float(0, 1.0, (len(env_ids), 3), device=self.device)
        q_w = torch.sqrt(1.0 - uvw[:, 0]) * (torch.sin(2 * np.pi * uvw[:, 1]))
        q_x = torch.sqrt(1.0 - uvw[:, 0]) * (torch.cos(2 * np.pi * uvw[:, 1]))
        q_y = torch.sqrt(uvw[:, 0]) * (torch.sin(2 * np.pi * uvw[:, 2]))
        q_z = torch.sqrt(uvw[:, 0]) * (torch.cos(2 * np.pi * uvw[:, 2]))
        new_rot = torch.cat((q_x.unsqueeze(-1), q_y.unsqueeze(-1), q_z.unsqueeze(-1), q_w.unsqueeze(-1)), dim=-1)

        return new_rot

    def get_identity_quat(self, env_ids):
        # https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py
        # https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L261

        uvw = torch.zeros(len(env_ids), 3, device=self.device)        
        q_w = torch.sqrt(1.0 - uvw[:, 0]) * (torch.sin(2 * np.pi * uvw[:, 1]))
        q_x = torch.sqrt(1.0 - uvw[:, 0]) * (torch.cos(2 * np.pi * uvw[:, 1]))
        q_y = torch.sqrt(uvw[:, 0]) * (torch.sin(2 * np.pi * uvw[:, 2]))
        q_z = torch.sqrt(uvw[:, 0]) * (torch.cos(2 * np.pi * uvw[:, 2]))
        new_rot = torch.cat((q_x.unsqueeze(-1), q_y.unsqueeze(-1), q_z.unsqueeze(-1), q_w.unsqueeze(-1)), dim=-1)                

        return new_rot

    def reset_target_goal(self, env_ids: Tensor) -> None:                
        pass
        # self.reset_target_goal_noise(env_ids)     
        # self.set_target()        

    def deferred_set_actor_root_state_tensor_indexed(self, obj_indices: List[Tensor]) -> None:
        self.set_actor_root_state_object_indices.extend(obj_indices)

    def set_actor_root_state_tensor_indexed(self) -> None:
        object_indices: List[Tensor] = self.set_actor_root_state_object_indices
        if not object_indices:
            # nothing to set
            return

        unique_object_indices = torch.unique(torch.cat(object_indices).to(torch.int32))

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state),
            gymtorch.unwrap_tensor(unique_object_indices),
            len(unique_object_indices),
        )

        self.set_actor_root_state_object_indices = []

    def _disable_gravity(self):
        """Disable gravity."""        
        sim_params = self.gym.get_sim_params(self.sim)
        sim_params.gravity.z = 0.0
        self.gym.set_sim_params(self.sim, sim_params)

    def _enable_gravity(self):
        """Enable gravity."""
        sim_params = self.gym.get_sim_params(self.sim)
        sim_params.gravity.z = -9.81
        self.gym.set_sim_params(self.sim, sim_params)

    def reset_franka(self, env_ids: Tensor) -> None:  
        is_randomize = False
        if is_randomize:                      
            delta_max = self.joint_dof_limits_upper - self.dof_pos
            delta_min = self.joint_dof_limits_lower - self.dof_pos
            
            rand_dof_pos_floats = torch_rand_float( 0.0, 1.0, (len(env_ids), self.num_dofs), device=self.device)
            rand_dof_vel_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_dofs), device=self.device)
            
            rand_dof_pos_delta = delta_min + (delta_max - delta_min) * rand_dof_pos_floats   
            rand_dof_vel_delta = delta_min + (delta_max - delta_min) * rand_dof_vel_floats    
            
            noise_dof_pos_coeff = torch.ones_like(self.dof_pos, device=self.device) * self.reset_dof_pos_noise_arm
            noise_dof_vel_coeff = torch.ones_like(self.dof_pos, device=self.device) * self.reset_dof_vel_noise_arm

            self.dof_pos[env_ids, :] = self.default_dof_pos + noise_dof_pos_coeff * rand_dof_pos_delta 
            self.dof_vel[env_ids, :] = self.default_dof_vel + noise_dof_vel_coeff * rand_dof_vel_delta 
        else:
            self.dof_pos[env_ids, :] = self.default_dof_pos
            self.dof_vel[env_ids, :] = self.default_dof_vel
            
        # flattened list of franka actors that we need to reset
        franka_indices = self.franka_indices[env_ids].to(torch.int32).flatten()
        
        # self.gym.set_dof_position_target_tensor_indexed(self.sim, gymtorch.unwrap_tensor(prev_targets), franka_indices_gym, num_franka_indices)
        self.gym.set_dof_state_tensor_indexed(
            self.sim, 
            gymtorch.unwrap_tensor(self.dof_state), 
            gymtorch.unwrap_tensor(franka_indices), 
            len(franka_indices)
        )

        self.gym.simulate(self.sim)
        self.refresh_tensor_sim_step()

    def _reset_object(self, only_respone):

        table_offset = torch.tensor([0.5, 0.0, 0.15], device=self.device).repeat((self.num_envs, 1))   

        obj_pose = {'holi_top':{}, 'holi_bottom_08':{}}      
        obj_pose['holi_top']['pos']        = torch.tensor([0.0, 0.0, 0.05],     device=self.device).repeat((self.num_envs, 1)) + table_offset 
        obj_pose['holi_top']['quat']       = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat((self.num_envs, 1))
        obj_pose['holi_bottom_08']['pos']  = torch.tensor([0.0, 0.0, 0.0],      device=self.device).repeat((self.num_envs, 1)) + table_offset 
        obj_pose['holi_bottom_08']['quat'] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat((self.num_envs, 1))
        
        obj_pose_offset = {}
        obj_pose_offset['pos']  = 1.0 * 0.05 * 2.0 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5) * torch.tensor([1, 4, 0], device=self.device)
        obj_pose_offset['quat'] = 1.0 * 0.20 * 2.0 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5) * torch.tensor([0, 0, 1], device=self.device)
        obj_pose_offset['quat'] = torch_utils.quat_from_euler_xyz(
            obj_pose_offset['quat'][:,0],
            obj_pose_offset['quat'][:,1],
            obj_pose_offset['quat'][:,2],
        )
        
        obj_pose_noise = {'holi_top':{}, 'holi_bottom_08':{}}
        for name in ['holi_top', 'holi_bottom_08']:  
            if not only_respone:
                obj_pose_noise[name] = {}
                obj_pose_noise[name]['pos']  = 0.0 * 0.01 * 2.0 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5) * torch.tensor([1, 1, 0], device=self.device)
                obj_pose_noise[name]['quat'] = 0.0 * 0.05 * 2.0 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5) * torch.tensor([0, 0, 1], device=self.device)
                obj_pose_noise[name]['quat'] = torch_utils.quat_from_euler_xyz(
                    obj_pose_noise[name]['quat'][:,0],
                    obj_pose_noise[name]['quat'][:,1],
                    obj_pose_noise[name]['quat'][:,2],
                )

                self.object_reset_pos[name]  = obj_pose[name]['pos'] + obj_pose_offset['pos'] + obj_pose_noise[name]['pos']            
                self.object_reset_quat[name] = quat_mul(obj_pose[name]['quat'], obj_pose_offset['quat'])        
                
            self.root_pos[:,    self.actor_id_env[name], 0:3] = self.object_reset_pos[name].clone()
            self.root_quat[:,   self.actor_id_env[name], 0:4] = self.object_reset_quat[name].clone()
            self.root_linvel[:, self.actor_id_env[name], 0:3] = 0.0
            self.root_angvel[:, self.actor_id_env[name], 0:3] = 0.0   
            
            # Set object root state
            actor_id_sim = self.actor_id_sim[name].clone().to(dtype=torch.int32)            
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_state),
                gymtorch.unwrap_tensor(actor_id_sim),
                len(actor_id_sim),
            )
            self.gym.simulate(self.sim)  
        self.refresh_tensor_sim_step()

        self.set_target()
            
    def _stabilize_franka(self):      
        actor_id_sim = self.actor_id_sim['franka'].clone().to(dtype=torch.int32)

        # Set zero-velocity
        self.dof_vel[:, :] = 0.0
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(actor_id_sim),
            len(actor_id_sim),
        )

        # Set zero-torque
        self.torques[:, :] = 0.0
        self.gym.set_dof_actuation_force_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.torques),
            gymtorch.unwrap_tensor(actor_id_sim),
            len(actor_id_sim),
        )

        # Simulate one step to apply changes
        self.gym.simulate(self.sim)
        self.refresh_tensor_sim_step()

    def move_gripper_to_target_pose(self, sim_steps):
        """Move gripper to control target pose."""        
        for _ in range(sim_steps): 
            self.generate_ctrl_signals(only_move=True)
            self.refresh_tensor_sim_step() 
            self.render()            
            if self.viewer:
                self.draw_franka()

        self._stabilize_franka()        

    def _move_gripper_to_grasp_pose(self, env_ids, sim_steps):
        """Define grasp pose for plug and move gripper to pose."""        
       
        for name in ['holi_top']:
            # range [0.0, 0.0, 0.0275] + [rand_pos, 0, rand_pos]
            offset_pos  = torch.tensor([0.0, 0.0, 0.0275], dtype=torch.float32, device=self.device).repeat((self.num_envs, 1))
            offset_pos += 0.0 * ((torch.rand((len(env_ids), 3), device=self.device) - 0.5) * 2.0) * torch.tensor([1, 0, 1], device=self.device)            

            # range [-rand_rot_angle, rand_rot_angle]
            rand_rot_angle = 0.0   
            half_angles = 0.5 * rand_rot_angle * ((torch.rand(len(env_ids), device=self.device) - 0.5 ) * 2.0)
            sin_angles, cos_angles = torch.sin(half_angles), torch.cos(half_angles)
            offset_rot = torch.stack([torch.zeros_like(sin_angles), sin_angles, torch.zeros_like(sin_angles), cos_angles], dim=1) # y-axis rotation
            
            x_rot_pi_quat = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=self.device).repeat((self.num_envs, 1))
            
            self.obj_grasp_point_pos[name] = self.object_pos[name] + quat_rotate(self.object_quat[name], offset_pos)                                    
            self.obj_grasp_point_rot[name] = torch_utils.quat_mul(torch_utils.quat_mul(self.object_quat[name], x_rot_pi_quat), offset_rot)            
            
        self.cur_eef_targets_w_actions[:,0:3] = self.obj_grasp_point_pos['holi_top']
        self.cur_eef_targets_w_actions[:,3:7] = self.obj_grasp_point_rot['holi_top']

        if self.hand_type == "Gripper":
            self.cur_hand_targets[:,0:self.num_hand_dofs] = to_torch([0.035, 0.035], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        elif self.hand_type == "TheHand":
            self.cur_hand_targets[:,0:self.num_hand_dofs] = to_torch([0.0]*self.num_hand_dofs, dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        elif self.hand_type == "ShadowHand":
            self.cur_hand_targets[:,0:self.num_hand_dofs] = to_torch([0.0]*self.num_hand_dofs, dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.move_gripper_to_target_pose(sim_steps=sim_steps)
        self._reset_object(True)

    def _open_gripper(self, env_ids, sim_steps):
        self.set_hand_actions('open')
        self.move_gripper_to_target_pose(sim_steps=sim_steps)

    def _close_gripper(self, env_ids, sim_steps):
        self.set_hand_actions('close')
        self.move_gripper_to_target_pose(sim_steps=sim_steps)

    def reset_noise_bias(self, env_ids: Tensor) -> None:
        self.curriculum_level = 200*(0.01 - self.success_tolerance) # level : 0 -> 1
        
        self.curriculum_biased_noise['pos'] = 0.000 + 0.001*self.curriculum_level   
        # self.curriculum_biased_noise['pos'] = 0.0

        self.curriculum_biased_noise['quat'] = 0.00 + 0.04*self.curriculum_level
        # self.curriculum_biased_noise['quat'] = 0.0

        is_hard = False
        noise_bias = {}
        for name in ['holi_top', 'holi_bottom_08']:
            noise_bias[name] = {}
            noise_bias[name]['pos'] = (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5) * 2.0
            if is_hard:
                beta_dist = torch.distributions.Beta(0.5, 0.5)        
                beta_samples = beta_dist.sample((self.num_envs, 3)).to(self.device)
                noise_bias[name]['pos'] = (beta_samples - 0.5) * 2.0

            noise_bias[name]['pos'] *= self.curriculum_biased_noise['pos']
            
            obj_euler_xyz_noise = self.curriculum_biased_noise['quat'] * 2.0 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)
            noise_bias[name]['quat'] = torch_utils.quat_from_euler_xyz(
                obj_euler_xyz_noise[:,0],
                obj_euler_xyz_noise[:,1],
                obj_euler_xyz_noise[:,2],
            )

            # biased error
            self.obj_estimation_biased_noise[name]['pos']  = noise_bias[name]['pos']
            self.obj_estimation_biased_noise[name]['quat'] = noise_bias[name]['quat']

    def reset_idx(self, env_ids: Tensor) -> None:
        # randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)        

        self.frame_since_restarts[env_ids] = 0        
        self.phase[env_ids] = 0           
        
        self.reset_target_goal(env_ids)
        self.reset_franka(env_ids)        
        
        self._disable_gravity()
        self._reset_object(False)          
        self._move_gripper_to_grasp_pose(env_ids, sim_steps=180)      
        self._close_gripper(env_ids, sim_steps=60)
        self._enable_gravity()    
        
        self.reset_noise_bias(env_ids)
        self.pose_estimate()
        
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

        self.prev_episode_successes[env_ids] = self.successes[env_ids]        
        self.successes[env_ids] = 0
        
        self.prev_episode_true_objective[env_ids] = self.true_objective[env_ids]
        self.true_objective[env_ids] = 0
        
        self.near_goal_steps[env_ids] = 0                

        for key in self.rewards_episode.keys():            
            self.rewards_episode[key][env_ids] = 0

        self.extras["scalars"] = dict()
        self.extras["scalars"]["success_tolerance"] = self.success_tolerance        

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)     
        
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)        
        if len(reset_env_ids) > 0:            
            self.reset_idx(reset_env_ids)        
        
        self.set_actor_root_state_tensor_indexed()

        if self.cfg_ctrl["ctrl_type"] == "joint_space_impedance":                                                
            targets = self.prev_eef_targets[:, :self.num_dofs] + self.action_scale * self.con_dt * self.actions[:, 0:self.num_dofs]
            self.cur_eef_targets[:, :self.num_dofs] = tensor_clamp(targets, self.joint_dof_limits_lower[0:self.num_dofs], self.joint_dof_limits_upper[:self.num_dofs])
        
        elif self.cfg_ctrl["ctrl_type"] == "task_space_impedance":                        
            pos_actions = 0.4 * self.action_scale * self.actions[:, 0:3] 
            rot_actions = 1.0 * self.action_scale * self.actions[:, 3:6] 
            if self.compliance_adjust:
                gain_actions = 0.1 * self.actions[:, 6]   

                self.gains_ratio = self.gains_ratio * gain_actions
                self.gains_ratio = torch.clamp(self.gains_ratio, min=0.5, max=1.5)
                self.task_prop_gains = self.gains_ratio.unsqueeze(-1) * self.task_prop_gains_base        

            self.target_goal_pos, self.target_goal_quat = self.get_target()            
            
            # ======================= pos target w actions  =======================
            
            self.cur_obj_targets[:, 0:3] = self.target_goal_pos
            self.cur_eef_targets[:, 0:3] = self.cur_obj_targets[:, 0:3] + self.transform_obj_to_eef['pos']            
            
            if self.action_type == "Relative":                   
                self.cur_eef_targets_w_actions[:, 0:3] = self.eef_end_pos             + pos_actions[:, 0:3]                
                self.cur_obj_targets_w_actions[:, 0:3] = self.obj_end_pos['holi_top'] + pos_actions[:, 0:3]                                    
            else:                
                self.cur_eef_targets_w_actions[:, 0:3] = self.cur_eef_targets[:, 0:3] + pos_actions[:, 0:3]              
                self.cur_obj_targets_w_actions[:, 0:3] = self.cur_obj_targets[:, 0:3] + pos_actions[:, 0:3]                      
                
            # ======================= quat target w actions =======================
            
            franka_angle = torch.norm(rot_actions[:, 0:3], p=2, dim=-1)
            franka_axis = rot_actions[:, 0:3] / franka_angle.unsqueeze(-1)
            franka_rot_actions_quat = torch_utils.quat_from_angle_axis(franka_angle, franka_axis)
            franka_rot_actions_quat = torch.where(
                franka_angle.unsqueeze(-1).repeat(1, 4) > 1.0e-6,
                franka_rot_actions_quat,
                torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1),
            )
            
            self.cur_obj_targets[:, 3:7] = self.target_goal_quat            
            self.cur_eef_targets[:, 3:7] = torch_utils.quat_mul(self.transform_obj_to_eef['quat'], self.cur_obj_targets[:, 3:7])            
            # self.cur_eef_targets[:, 3:7] = self.target_goal_quat[name]
            if self.action_type == "Relative":
                self.cur_eef_targets_w_actions[:, 3:7] = torch_utils.quat_mul(franka_rot_actions_quat, self.eef_end_rot)
                self.cur_obj_targets_w_actions[:, 3:7] = torch_utils.quat_mul(franka_rot_actions_quat, self.obj_end_rot['holi_top'])
            else:
                # self.cur_eef_targets_w_actions[:, 3:7] = torch_utils.quat_mul(franka_rot_actions_quat, self.cur_eef_targets[:, 3:7])
                # self.cur_obj_targets_w_actions[:, 3:7] = torch_utils.quat_mul(franka_rot_actions_quat, self.cur_obj_targets[:, 3:7])                     
                self.cur_eef_targets_w_actions[:, 3:7] = torch_utils.quat_mul(franka_rot_actions_quat, self.eef_end_rot)
                self.cur_obj_targets_w_actions[:, 3:7] = torch_utils.quat_mul(franka_rot_actions_quat, self.obj_end_rot['holi_top'])               
            
            self.cur_eef_targets_w_actions[:, 3:7] = self._normalize_quat(self.cur_eef_targets_w_actions[:, 3:7])
            self.cur_obj_targets_w_actions[:, 3:7] = self._normalize_quat(self.cur_obj_targets_w_actions[:, 3:7])            

        if self.control_type == "Pos":            
            pos_targets = self.cur_eef_targets # target need to joint pos, current is pos+quat 
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(pos_targets))
            self.gym.simulate(self.sim)

        self.set_hand_actions('close')

    def set_hand_actions(self, mode):        
        if self.hand_type == "Gripper":                        
            if mode == 'open':                    
                self.cur_hand_targets[:,0:self.num_hand_dofs] = to_torch([0.035, 0.035], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
            elif mode == 'close':
                self.cur_hand_targets[:,0:self.num_hand_dofs] = to_torch([0.0, 0.0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        elif self.hand_type == "TheHand":                        
            if mode == 'open':                    
                self.cur_hand_targets[:,0:self.num_hand_dofs] = to_torch([0.0]*self.num_hand_dofs, dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
            elif mode == 'close':
                self.cur_hand_targets[:,0:self.num_hand_dofs] = to_torch([0.0]*self.num_hand_dofs, dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        elif self.hand_type == "ShadowHand":                        
            if mode == 'open':                    
                self.cur_hand_targets[:,0:self.num_hand_dofs] = to_torch([0.0]*self.num_hand_dofs, dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
            elif mode == 'close':
                self.cur_hand_targets[:,0:self.num_hand_dofs] = to_torch([0.0]*self.num_hand_dofs, dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

    def generate_ctrl_signals(self, only_move=False):
        # Get desired Jacobian                
        if self.cfg_ctrl['all']['jacobian_type'] == 'geometric':
            self.jacobian_tf[self.end_link] = self.jacobian[self.end_link]            
        elif self.cfg_ctrl['all']['jacobian_type'] == 'analytic':
            self.jacobian_tf[self.end_link] = fc.get_analytic_jacobian(                    
                jacobian=self.jacobian[self.end_link],                
                num_envs=self.num_envs,
                device=self.device)
        
        self._set_dof_torque('normal', only_move)                        
        # self._set_dof_torque('ext_force_comp') 
        
        # self.torques = torch.zeros_like(self.torques)         
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))                
        self.gym.simulate(self.sim)  

    def _set_dof_torque(self, mode, only_move):        
        cur_contact_force = torch.zeros((self.num_envs,3))

        force_sensor_data = torch.cat([
            self.sensor_queue['fx'][0].view(self.num_envs, 1),
            self.sensor_queue['fy'][0].view(self.num_envs, 1),
            self.sensor_queue['fz'][0].view(self.num_envs, 1)], dim=1)
        
        torque_sensor_data = torch.cat([
            self.sensor_queue['tx'][0].view(self.num_envs, 1),
            self.sensor_queue['ty'][0].view(self.num_envs, 1),
            self.sensor_queue['tz'][0].view(self.num_envs, 1)], dim=1)
        
        mask = self.eef_end_rot[:,0] * self.cur_eef_targets_w_actions[:,3] < 0  # Boolean mask
        self.cur_eef_targets_w_actions[mask, 3:7] *= -1 
        
        torques, wrench = fc.compute_dof_torque( 
            cfg_ctrl=self.cfg_ctrl,
            name='franka',
            mode=mode,
            only_move=only_move,
            p_gain=self.task_prop_gains,
            d_gain=self.task_deriv_gains,
            f_gain=self.f_gain,
            dof_pos=self.dof_pos,
            dof_vel=self.dof_vel,
            pos =self.eef_end_pos,
            quat=self.eef_end_rot,
            linvel=self.robot_linvel[self.end_link],
            angvel=self.robot_angvel[self.end_link],
            jacobian=self.jacobian_tf[self.end_link],
            mass_matrix=self.mass_matrix,            
            ctrl_target_pos=self.cur_eef_targets_w_actions[:,0:3],
            ctrl_target_quat=self.cur_eef_targets_w_actions[:,3:7],
            ctrl_hand_target=self.cur_hand_targets[:,0:self.num_hand_dofs],
            cur_target_contact_force = self.target_contact_force,
            cur_contact_force = cur_contact_force,  
            ext_force_global = force_sensor_data,
            ext_torque_global = torque_sensor_data,
            hand_type = self.hand_type,         
            device=self.device)
    
        torques = torch.max(torch.min(torques, self.franka_effort_max), self.franka_effort_min)            
        self.torques[:, 0:self.num_dofs] = torques                    

    def pd_control(self):        
        # need to be initialized earlier, not here. 
        # for pd control
        self.p_gain_val = 40.0
        self.d_gain_val = 5.0
        self.p_gain = torch.ones((self.num_envs, self.num_dofs), device=self.device, dtype=torch.float) * self.p_gain_val
        self.d_gain = torch.ones((self.num_envs, self.num_dofs), device=self.device, dtype=torch.float) * self.d_gain_val                                      
        self.pd_dof_pos = torch.zeros((self.num_envs, self.num_dofs), device=self.device, dtype=torch.float) * self.p_gain_val    

        self.pd_dof_pos[:, :] = self.dof_pos.clone()        
                
        cur_targets = self.cur_eef_targets_w_actions.clone()        
        dof_pos_diff = (cur_targets - self.pd_dof_pos)
        
        torques = self.p_gain * (dof_pos_diff) - self.d_gain * self.dof_vel        
        
        self.torques = torques.clone()        
        self.torques = torch.max(torch.min(self.torques, self.franka_effort_max), self.franka_effort_min)                       

        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques)) 
        self.gym.simulate(self.sim)                

    def simulation_steps(self):
        if self.control_type == "Pos":
            self.refresh_tensor_sim_step()            
        elif self.control_type == "Effort":                     
            for i in range(self.num_simulation_steps):
                if self.cfg_ctrl["ctrl_type"] == "joint_space_impedance":
                    self.pd_control()
                elif self.cfg_ctrl["ctrl_type"] == "task_space_impedance" or self.cfg_ctrl["ctrl_type"] == "operational_space_motion":
                    self.generate_ctrl_signals()
                elif self.cfg_ctrl["ctrl_type"] == "hybrid_force_motion":
                    self.generate_ctrl_signals()

                self.refresh_tensor_sim_step()   
            self.refresh_tensor_con_step()

    def refresh_tensor_sim_step(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)        
        self.gym.refresh_rigid_body_state_tensor(self.sim)        
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)  
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos  = self.goal_states[:, 0:3]
        self.goal_rot  = self._normalize_quat(self.goal_states[:, 3:7])                        
        if self.hand_type == "Gripper":
            if self.end_link == 'panda_firgertip_midpoint':
                self.jacobian[self.end_link]     = 0.5 * (self.jacobian['panda_leftfinger_tip']  + self.jacobian['panda_rightfinger_tip'])
                self.robot_pos[self.end_link]    = 0.5 * (self.robot_pos['panda_leftfinger_tip'] + self.robot_pos['panda_rightfinger_tip'])                
                self.robot_quat[self.end_link]   = self._normalize_quat(self.robot_quat['panda_fingertip_centered'])
                self.robot_linvel[self.end_link] = self.robot_linvel['panda_fingertip_centered']
                self.robot_angvel[self.end_link] = self.robot_angvel['panda_fingertip_centered']                                        
        
        self.eef_end_rot = self._normalize_quat(self.robot_quat[self.end_link])                    
        self.eef_end_pos = self.robot_pos[self.end_link]
        self.eef_end_linvel = self.robot_linvel[self.end_link]
        self.eef_end_angvel = self.robot_angvel[self.end_link]
        
        half_offset = {}
        half_offset['holi_top']       = torch.tensor([0.0, 0.0, 0.0],  dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        half_offset['holi_bottom_08'] = torch.tensor([0.0, 0.0, 0.02], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))                
        ### [original] and [noise] 
        for name in ['holi_top', 'holi_bottom_08']:            
            self.obj_end_rot[name]     = self._normalize_quat(quat_mul(self.obj_estimation_noise[name]['quat'], self.object_quat[name]))
            self.obj_end_rot_org[name] = self._normalize_quat(self.object_quat[name])            
            self.obj_end_pos[name]     = self.object_pos[name] + quat_rotate(self.obj_end_rot_org[name], half_offset[name]+self.obj_estimation_noise[name]['pos']) + quat_rotate(self.obj_end_rot[name], half_offset[name])             
            self.obj_end_pos_org[name] = self.object_pos[name] + quat_rotate(self.obj_end_rot_org[name], self.obj_end_offset[name]) 
            self.obj_end_linvel[name]  = self.object_linvel[name]
            self.obj_end_angvel[name]  = self.object_angvel[name]
        
        for i, s_type in enumerate(['fx', 'fy', 'fz', 'tx', 'ty', 'tz']):
            self.sensor_queue[s_type].appendleft(self.ft_sensor[:,i])                
        
    def refresh_tensor_con_step(self): 
        pass

    def _compute_sensor_data(self):
        self.sensor_data_scaler = 1
        self.sensor_avg_data = {}
        for s_type in ['fx', 'fy', 'fz', 'tx', 'ty', 'tz']:
            self.sensor_avg_data[s_type] = (sum(self.sensor_queue[s_type]) / (self.sensor_data_scaler * len(self.sensor_queue[s_type]))).view(self.num_envs,1)

    def post_physics_step(self): 
        self.frame_since_restart += 1
        self.frame_since_restarts += 1
        self.randomize_buf += 1        
        self.progress_buf += 1        
        self.episode_progress_buf += 1          
        self._extra_curriculum()       
        self._compute_sensor_data()
        
        rewards, is_success = self.compute_franka_reward()     
        self._compute_phase()
        self.set_target()                     
        
        obs_buf, reward_obs_ofs = self.compute_observations()        
                        
        resets = self._compute_resets()
        self.reset_buf[:] = resets

        # add rewards to observations
        reward_obs_scale = 0.01
        obs_buf[:, reward_obs_ofs : reward_obs_ofs + 1] = rewards.unsqueeze(-1) * reward_obs_scale

        self.prev_actions = self.actions.clone()
        self.prev_eef_targets[:, :] = self.cur_eef_targets[:, :].clone()
        self.prev_eef_targets_w_actions[:, :] = self.cur_eef_targets_w_actions[:, :].clone()
       
        self.clamp_obs(obs_buf)
        self._eval_stats(is_success)
        # if self.save_states:
        #     self.accumulate_env_states()

        if self.plot_sensor:
            self.draw_sensor_plot()

        if self.viewer:
            self.draw_franka()
    
    def draw_sensor_plot(self):
        # raw data
        raw_data = {}
        for s_type in ['fx', 'fy', 'fz', 'tx', 'ty', 'tz']:
            raw_data[s_type] = self.sensor_queue[s_type][0].view(self.num_envs,1) 

        self.sensor_avg_data_plot = torch.roll(self.sensor_avg_data_plot, shifts=-1, dims=0)        

        # raw data
        # self.sensor_avg_data_plot[-1, 0] = 0.1 * raw_data['fx']
        # self.sensor_avg_data_plot[-1, 1] = 0.1 * raw_data['fy']
        # self.sensor_avg_data_plot[-1, 2] = 0.1 * raw_data['fz']
        # self.sensor_avg_data_plot[-1, 3] = 0.1 * raw_data['tx']
        # self.sensor_avg_data_plot[-1, 4] = 0.1 * raw_data['ty']
        # self.sensor_avg_data_plot[-1, 5] = 0.1 * raw_data['tz']

        # avg data + action data
        # self.sensor_avg_data_plot[-1, 0] = self.sensor_avg_data['fx']
        # self.sensor_avg_data_plot[-1, 1] = self.sensor_avg_data['fy']
        # self.sensor_avg_data_plot[-1, 2] = self.sensor_avg_data['fz']
        # self.sensor_avg_data_plot[-1, 3] = self.actions[...,0]
        # self.sensor_avg_data_plot[-1, 4] = self.actions[...,1]
        # self.sensor_avg_data_plot[-1, 5] = self.actions[...,2]
        
        # avg data + action data * gain ratio
        self.sensor_avg_data_plot[-1, 0] = self.sensor_avg_data['fx']
        self.sensor_avg_data_plot[-1, 1] = self.sensor_avg_data['fy']
        self.sensor_avg_data_plot[-1, 2] = self.sensor_avg_data['fz']
        self.sensor_avg_data_plot[-1, 3] = self.actions[...,0] 
        self.sensor_avg_data_plot[-1, 4] = self.actions[...,1] 
        self.sensor_avg_data_plot[-1, 5] = self.actions[...,2] 
        
        # self.sensor_avg_data_plot[-1, 3] = self.actions[...,0] * self.gains_ratio 
        # self.sensor_avg_data_plot[-1, 4] = self.actions[...,1] * self.gains_ratio 
        # self.sensor_avg_data_plot[-1, 5] = self.actions[...,2] * self.gains_ratio 
        # self.sensor_avg_data_plot[-1, 3] = self.gains_ratio
        # self.sensor_avg_data_plot[-1, 4] = self.gains_ratio
        # self.sensor_avg_data_plot[-1, 5] = self.gains_ratio
    
        for i in range(6):                
            self.lines0[i].set_xdata(range(100))
            self.lines0[i].set_ydata(self.sensor_avg_data_plot[:, i].cpu().numpy())
            # self.lines1[i].set_xdata(range(100))
            # self.lines1[i].set_ydata(self.sensor_avg_data_plot[:, i].cpu().numpy())
            
        plt.draw()
        plt.pause(0.01)
        plt.show()        

    def draw_franka(self):        
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        # color map
        color_red   = (1.0,0.0,0.0)
        color_green = (0.0,1.0,0.0)
        color_blue  = (0.0,0.0,1.0)       
        
        # base axis
        px_offset = torch.tensor([0.05, 0.0, 0.0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        py_offset = torch.tensor([0.0, 0.05, 0.0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        pz_offset = torch.tensor([0.0, 0.0, 0.05], dtype=torch.float, device=self.device).repeat((self.num_envs, 1)) 

        ### target 
        for i in range(self.num_envs):
            ### obj target
            # sphere_pose = gymapi.Transform()
            # sphere_pose.p = gymapi.Vec3(0.5*self.cur_obj_targets[i][0], 0.5*self.cur_obj_targets[i][1], 0.5*self.cur_obj_targets[i][2])                                        
            # sphere_pose.r = gymapi.Quat(0, 0, 0, 1)
            # sphere_geom = gymutil.WireframeSphereGeometry(0.001, 12, 12, sphere_pose, color=color_red)                
            # gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

            ### eef target with action
            # sphere_pose = gymapi.Transform()
            # sphere_pose.p = gymapi.Vec3(0.5*self.cur_eef_targets_w_actions[i][0], 0.5*self.cur_eef_targets_w_actions[i][1], 0.5*self.cur_eef_targets_w_actions[i][2])
            # sphere_pose.r = gymapi.Quat(0, 0, 0, 1)                
            # sphere_geom = gymutil.WireframeSphereGeometry(0.001, 12, 12, sphere_pose, color=color_green)                
            # gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

            ### obj target with action     
            sphere_pose = gymapi.Transform()
            sphere_pose.p = gymapi.Vec3(0.5*self.cur_obj_targets_w_actions[i][0], 0.5*self.cur_obj_targets_w_actions[i][1], 0.5*self.cur_obj_targets_w_actions[i][2])
            sphere_pose.r = gymapi.Quat(0, 0, 0, 1)                
            sphere_geom = gymutil.WireframeSphereGeometry(0.001, 12, 12, sphere_pose, color=color_green)                
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

        
        ### obj end (peg)
        # for i in range(self.num_envs):
        #     sphere_pose = gymapi.Transform()
        #     sphere_pose.p = gymapi.Vec3(0.5*self.obj_end_pos['holi_top'][i][0], 0.5*self.obj_end_pos['holi_top'][i][1], 0.5*self.obj_end_pos['holi_top'][i][2])                                        
        #     sphere_pose.r = gymapi.Quat(0, 0, 0, 1)
        #     sphere_geom = gymutil.WireframeSphereGeometry(0.002, 16, 16, sphere_pose, color=(1, 0, 1))                
        #     gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

        # obj_px = self.obj_end_pos['holi_top'] + quat_rotate(self.obj_end_rot['holi_top'], px_offset)
        # obj_py = self.obj_end_pos['holi_top'] + quat_rotate(self.obj_end_rot['holi_top'], py_offset)
        # obj_pz = self.obj_end_pos['holi_top'] + quat_rotate(self.obj_end_rot['holi_top'], pz_offset)       
        # for i in range(self.num_envs):     
        #     p0_ = gymapi.Vec3(self.obj_end_pos['holi_top'][i][0], self.obj_end_pos['holi_top'][i][1], self.obj_end_pos['holi_top'][i][2])  
        #     px_ = gymapi.Vec3(obj_px[i][0], obj_px[i][1], obj_px[i][2])  
        #     py_ = gymapi.Vec3(obj_py[i][0], obj_py[i][1], obj_py[i][2])  
        #     pz_ = gymapi.Vec3(obj_pz[i][0], obj_pz[i][1], obj_pz[i][2])
            
        #     gymutil.draw_line(p0_, px_, gymapi.Vec3(1, 0, 0), self.gym, self.viewer, self.envs[i])
        #     gymutil.draw_line(p0_, py_, gymapi.Vec3(0, 1, 0), self.gym, self.viewer, self.envs[i])
        #     gymutil.draw_line(p0_, pz_, gymapi.Vec3(0, 0, 1), self.gym, self.viewer, self.envs[i])

        ### obj end (hole)
        # for i in range(self.num_envs):
        #     sphere_pose = gymapi.Transform()
        #     sphere_pose.p = gymapi.Vec3(0.5*self.obj_end_pos['holi_bottom_08'][i][0], 0.5*self.obj_end_pos['holi_bottom_08'][i][1], 0.5*self.obj_end_pos['holi_bottom_08'][i][2])                                        
        #     sphere_pose.r = gymapi.Quat(0, 0, 0, 1)
        #     sphere_geom = gymutil.WireframeSphereGeometry(0.002, 16, 16, sphere_pose, color=(1, 0, 1))                
        #     gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

        # obj_px = self.obj_end_pos['holi_bottom_08'] + quat_rotate(self.obj_end_rot['holi_bottom_08'], px_offset)
        # obj_py = self.obj_end_pos['holi_bottom_08'] + quat_rotate(self.obj_end_rot['holi_bottom_08'], py_offset)
        # obj_pz = self.obj_end_pos['holi_bottom_08'] + quat_rotate(self.obj_end_rot['holi_bottom_08'], pz_offset)
        # for i in range(self.num_envs):     
        #     p0_ = gymapi.Vec3(self.obj_end_pos['holi_bottom_08'][i][0], self.obj_end_pos['holi_bottom_08'][i][1], self.obj_end_pos['holi_bottom_08'][i][2])  
        #     px_ = gymapi.Vec3(obj_px[i][0], obj_px[i][1], obj_px[i][2])  
        #     py_ = gymapi.Vec3(obj_py[i][0], obj_py[i][1], obj_py[i][2])  
        #     pz_ = gymapi.Vec3(obj_pz[i][0], obj_pz[i][1], obj_pz[i][2])
            
        #     gymutil.draw_line(p0_, px_, gymapi.Vec3(1, 0, 0), self.gym, self.viewer, self.envs[i])
        #     gymutil.draw_line(p0_, py_, gymapi.Vec3(0, 1, 0), self.gym, self.viewer, self.envs[i])
        #     gymutil.draw_line(p0_, pz_, gymapi.Vec3(0, 0, 1), self.gym, self.viewer, self.envs[i])
              

        ### franka end-link                
        # for i in range(self.num_envs):
        #     sphere_pose = gymapi.Transform()
        #     sphere_pose.p = gymapi.Vec3(0.5*self.eef_end_pos[i][0], 0.5*self.eef_end_pos[i][1], 0.5*self.eef_end_pos[i][2])                                        
        #     sphere_pose.r = gymapi.Quat(0, 0, 0, 1)
        #     sphere_geom = gymutil.WireframeSphereGeometry(0.002, 16, 16, sphere_pose, color=(1.0, 0.0, 1.0))                
        #     gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

        # eef_px = self.eef_end_pos + quat_rotate(self.eef_end_rot, px_offset)
        # eef_py = self.eef_end_pos + quat_rotate(self.eef_end_rot, py_offset)
        # eef_pz = self.eef_end_pos + quat_rotate(self.eef_end_rot, pz_offset)
        # for i in range(self.num_envs):     
        #     p0_ = gymapi.Vec3(self.eef_end_pos[i][0], self.eef_end_pos[i][1], self.eef_end_pos[i][2])  
        #     px_ = gymapi.Vec3(eef_px[i][0], eef_px[i][1], eef_px[i][2])  
        #     py_ = gymapi.Vec3(eef_py[i][0], eef_py[i][1], eef_py[i][2])  
        #     pz_ = gymapi.Vec3(eef_pz[i][0], eef_pz[i][1], eef_pz[i][2])
            
        #     gymutil.draw_line(p0_, px_, gymapi.Vec3(1, 0, 0), self.gym, self.viewer, self.envs[i])
        #     gymutil.draw_line(p0_, py_, gymapi.Vec3(0, 1, 0), self.gym, self.viewer, self.envs[i])
        #     gymutil.draw_line(p0_, pz_, gymapi.Vec3(0, 0, 1), self.gym, self.viewer, self.envs[i]) 
    

        ### grasp pose
        # for name in ['holi_top']:
        #     for i in range(self.num_envs):
        #         sphere_pose = gymapi.Transform()
        #         sphere_pose.p = gymapi.Vec3(0.5*self.obj_grasp_point_pos[name][i][0], 0.5*self.obj_grasp_point_pos[name][i][1], 0.5*self.obj_grasp_point_pos[name][i][2])                                        
        #         sphere_pose.r = gymapi.Quat(0, 0, 0, 1)
        #         sphere_geom = gymutil.WireframeSphereGeometry(0.0005, 16, 16, sphere_pose, color=(0, 1, 1))                
        #         gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

        #     target_px = self.obj_grasp_point_pos[name] + quat_rotate(self.obj_grasp_point_rot[name], px_offset)
        #     target_py = self.obj_grasp_point_pos[name] + quat_rotate(self.obj_grasp_point_rot[name], py_offset)
        #     target_pz = self.obj_grasp_point_pos[name] + quat_rotate(self.obj_grasp_point_rot[name], pz_offset)
        #     for i in range(self.num_envs):     
        #         p0_ = gymapi.Vec3(self.obj_grasp_point_pos[name][i][0], self.obj_grasp_point_pos[name][i][1], self.obj_grasp_point_pos[name][i][2])  
        #         px_ = gymapi.Vec3(target_px[i][0], target_px[i][1], target_px[i][2])  
        #         py_ = gymapi.Vec3(target_py[i][0], target_py[i][1], target_py[i][2])  
        #         pz_ = gymapi.Vec3(target_pz[i][0], target_pz[i][1], target_pz[i][2])
                
        #         gymutil.draw_line(p0_, px_, gymapi.Vec3(0, 1, 1), self.gym, self.viewer, self.envs[i])
        #         gymutil.draw_line(p0_, py_, gymapi.Vec3(0, 1, 1), self.gym, self.viewer, self.envs[i])
        #         gymutil.draw_line(p0_, pz_, gymapi.Vec3(0, 1, 1), self.gym, self.viewer, self.envs[i]) 


        ### [original] target axis        
        for i in range(self.num_envs):
            sphere_pose = gymapi.Transform()
            sphere_pose.p = gymapi.Vec3(0.5*self.target_goal_pos_org[i][0], 0.5*self.target_goal_pos_org[i][1], 0.5*self.target_goal_pos_org[i][2])                                        
            sphere_pose.r = gymapi.Quat(0, 0, 0, 1)
            sphere_geom = gymutil.WireframeSphereGeometry(0.001, 12, 12, sphere_pose, color=(1, 0, 1))                
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

        target_px = self.target_goal_pos_org + quat_rotate(self.target_goal_quat_org, px_offset)
        target_py = self.target_goal_pos_org + quat_rotate(self.target_goal_quat_org, py_offset)
        target_pz = self.target_goal_pos_org + quat_rotate(self.target_goal_quat_org, pz_offset)
        for i in range(self.num_envs):     
            p0_ = gymapi.Vec3(self.target_goal_pos_org[i][0], self.target_goal_pos_org[i][1], self.target_goal_pos_org[i][2])  
            px_ = gymapi.Vec3(target_px[i][0], target_px[i][1], target_px[i][2])  
            py_ = gymapi.Vec3(target_py[i][0], target_py[i][1], target_py[i][2])  
            pz_ = gymapi.Vec3(target_pz[i][0], target_pz[i][1], target_pz[i][2])
                
            gymutil.draw_line(p0_, px_, gymapi.Vec3(1, 0, 0), self.gym, self.viewer, self.envs[i])
            gymutil.draw_line(p0_, py_, gymapi.Vec3(0, 1, 0), self.gym, self.viewer, self.envs[i])
            gymutil.draw_line(p0_, pz_, gymapi.Vec3(0, 0, 1), self.gym, self.viewer, self.envs[i]) 
        

        ### [noise] target axis            
        # for i in range(self.num_envs):
        #     sphere_pose = gymapi.Transform()
        #     sphere_pose.p = gymapi.Vec3(0.5*self.target_goal_pos[i][0], 0.5*self.target_goal_pos[i][1], 0.5*self.target_goal_pos[i][2])                                        
        #     sphere_pose.r = gymapi.Quat(0, 0, 0, 1)
        #     sphere_geom = gymutil.WireframeSphereGeometry(0.0005, 16, 16, sphere_pose, color=(0, 1, 1))                
        #     gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

        # target_px = self.target_goal_pos + quat_rotate(self.target_goal_quat, px_offset)
        # target_py = self.target_goal_pos + quat_rotate(self.target_goal_quat, py_offset)
        # target_pz = self.target_goal_pos + quat_rotate(self.target_goal_quat, pz_offset)
        # for i in range(self.num_envs):     
        #     p0_ = gymapi.Vec3(self.target_goal_pos[i][0], self.target_goal_pos[i][1], self.target_goal_pos[i][2])  
        #     px_ = gymapi.Vec3(target_px[i][0], target_px[i][1], target_px[i][2])  
        #     py_ = gymapi.Vec3(target_py[i][0], target_py[i][1], target_py[i][2])  
        #     pz_ = gymapi.Vec3(target_pz[i][0], target_pz[i][1], target_pz[i][2])
            
        #     gymutil.draw_line(p0_, px_, gymapi.Vec3(1, 0, 1), self.gym, self.viewer, self.envs[i])
        #     gymutil.draw_line(p0_, py_, gymapi.Vec3(1, 0, 1), self.gym, self.viewer, self.envs[i])
        #     gymutil.draw_line(p0_, pz_, gymapi.Vec3(1, 0, 1), self.gym, self.viewer, self.envs[i])               

        ### obj estimated pose
        est_offset = {'holi_top':[], 'holi_bottom_08':[]}                
        est_offset['holi_top'].append(torch.tensor([-0.025, -0.025,  0.0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1)))
        est_offset['holi_top'].append(torch.tensor([-0.025,  0.025,  0.0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1)))
        est_offset['holi_top'].append(torch.tensor([ 0.025,  0.025,  0.0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1)))
        est_offset['holi_top'].append(torch.tensor([ 0.025, -0.025,  0.0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1)))
        est_offset['holi_top'].append(torch.tensor([-0.025, -0.025,  0.04], dtype=torch.float, device=self.device).repeat((self.num_envs, 1)))
        est_offset['holi_top'].append(torch.tensor([-0.025,  0.025,  0.04], dtype=torch.float, device=self.device).repeat((self.num_envs, 1)))
        est_offset['holi_top'].append(torch.tensor([ 0.025,  0.025,  0.04], dtype=torch.float, device=self.device).repeat((self.num_envs, 1)))
        est_offset['holi_top'].append(torch.tensor([ 0.025, -0.025,  0.04], dtype=torch.float, device=self.device).repeat((self.num_envs, 1)))

        est_offset['holi_bottom_08'].append(torch.tensor([-0.025, -0.025,  0.0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1)))
        est_offset['holi_bottom_08'].append(torch.tensor([-0.025,  0.025,  0.0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1)))
        est_offset['holi_bottom_08'].append(torch.tensor([ 0.025,  0.025,  0.0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1)))
        est_offset['holi_bottom_08'].append(torch.tensor([ 0.025, -0.025,  0.0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1)))
        est_offset['holi_bottom_08'].append(torch.tensor([-0.025, -0.025, -0.04], dtype=torch.float, device=self.device).repeat((self.num_envs, 1)))
        est_offset['holi_bottom_08'].append(torch.tensor([-0.025,  0.025, -0.04], dtype=torch.float, device=self.device).repeat((self.num_envs, 1)))
        est_offset['holi_bottom_08'].append(torch.tensor([ 0.025,  0.025, -0.04], dtype=torch.float, device=self.device).repeat((self.num_envs, 1)))
        est_offset['holi_bottom_08'].append(torch.tensor([ 0.025, -0.025, -0.04], dtype=torch.float, device=self.device).repeat((self.num_envs, 1)))

        est_p = []
        for i in range(8):
            est_p.append({})            

        line_pair = [
            (0,1), (1,2), (2,3), (3,0),
            (4,5), (5,6), (6,7), (7,4),
            (0,1), (1,5), (5,4), (4,0),
            (1,5), (5,6), (6,2), (2,1),
            (2,6), (6,7), (7,3), (3,2),
            (0,4), (4,7), (7,3), (3,0),
            ]

        ### obj pose estimation box
        for name in ['holi_top', 'holi_bottom_08']:
            for i in range(self.num_envs):
                est_p_vec3 = []
                for j in range(8):                                        
                    est_p[j][name] = self.obj_end_pos[name] + quat_rotate(self.obj_end_rot[name], est_offset[name][j])

                    sphere_pose = gymapi.Transform()
                    sphere_pose.p = gymapi.Vec3(0.5*est_p[j][name][i][0], 0.5*est_p[j][name][i][1], 0.5*est_p[j][name][i][2])                                        
                    sphere_pose.r = gymapi.Quat(0, 0, 0, 1)
                    sphere_geom = gymutil.WireframeSphereGeometry(0.0005, 16, 16, sphere_pose, color=(0.2, 1, 1))                
                    gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

                    est_p_vec3.append(gymapi.Vec3(est_p[j][name][i][0], est_p[j][name][i][1], est_p[j][name][i][2]))

                for st, ed in line_pair:
                    gymutil.draw_line(est_p_vec3[st], est_p_vec3[ed], gymapi.Vec3(1.0, 0.2, 0.2), self.gym, self.viewer, self.envs[i])

       