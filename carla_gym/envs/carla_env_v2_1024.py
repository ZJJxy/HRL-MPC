"""
@author: Majid Moghadam
UCSC - ASL
"""

import gym
import cv2
import json
from PIL import ImageGrab
import time
import pylab
import keyboard
import torch
import itertools
from tools.modules import *
from config import cfg
from agents.local_planner.frenet_optimal_trajectory import FrenetPlanner as MotionPlanner
from agents.local_planner.frenet_optimal_trajectory import closest
from agents.low_level_controller.controller import VehiclePIDController
from agents.tools.misc import get_speed
from agents.low_level_controller.controller import IntelligentDriverModel
from PORF.reward_func import PORF

MODULE_WORLD = 'WORLD'
MODULE_HUD = 'HUD'
MODULE_INPUT = 'INPUT'
MODULE_TRAFFIC = 'TRAFFIC'
TENSOR_ROW_NAMES = ['EGO', 'LEADING', 'FOLLOWING', 'LEFT', 'LEFT_UP', 'LEFT_DOWN','LLEFT', 'LLEFT_UP', 'LLEFT_DOWN',
                    'RIGHT', 'RIGHT_UP', 'RIGHT_DOWN', 'RRIGHT', 'RRIGHT_UP', 'RRIGHT_DOWN']
# 设置录制参数
INIT_POSITION = (323, 335)
SCREEN_SIZE = (1200, 500)
FILENAME = 'recorded_video.avi'
FPS = 30.0

RLHF_TRAIN_SAMPLE = False
CWD = os.getcwd()
JSON_PATH = os.path.join(CWD, "RLHF", "json")
LOG_PATH = os.path.join(CWD, "RLHF", "logs")
import csv

def euclidean_distance(v1, v2):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(v1, v2)]))

def inertial_to_body_frame(ego_location, xi, yi, psi):
    Xi = np.array([xi, yi])  # inertial frame
    R_psi_T = np.array([[np.cos(psi), np.sin(psi)],  # Rotation matrix transpose
                        [-np.sin(psi), np.cos(psi)]])
    Xt = np.array([ego_location[0],  # Translation from inertial to body frame
                   ego_location[1]])
    Xb = np.matmul(R_psi_T, Xi - Xt)
    return Xb


def closest_wp_idx(ego_state, fpath, f_idx, w_size=10):
    """
    given the ego_state and frenet_path this function returns the closest WP in front of the vehicle that is within the w_size
    """

    min_dist = 300  # in meters (Max 100km/h /3.6) * 2 sn
    ego_location = [ego_state[0], ego_state[1]]
    closest_wp_index = 0  # default WP
    w_size = w_size if w_size <= len(fpath.t) - 2 - f_idx else len(fpath.t) - 2 - f_idx
    for i in range(w_size):
        temp_wp = [fpath.x[f_idx + i], fpath.y[f_idx + i]]
        temp_dist = euclidean_distance(ego_location, temp_wp)
        if temp_dist <= min_dist and inertial_to_body_frame(ego_location, temp_wp[0], temp_wp[1], ego_state[2])[0] > 0.0:
            closest_wp_index = i
            min_dist = temp_dist

    return f_idx + closest_wp_index


class CarlaGymEnv(gym.Env):
    # metadata = {'render.modes': ['human']}
    def __init__(self):
        self.__version__ = "9.9.2"

        # simulation
        self.verbosity = 0
        self.auto_render = False  # automatically render the environment
        self.human_ctrl = False
        self.human_ctrl_buffer = []
        self.out = None
        self.video_path = None
        self.n_step = 0
        try:
            self.global_route = np.load(
                'road_maps/global_route_town04.npy')  # track waypoints (center lane of the second lane from left)
        except IOError:
            self.global_route = None

        # constraints
        self.targetSpeed = float(cfg.GYM_ENV.TARGET_SPEED)
        self.avgSpeed = float(cfg.GYM_ENV.TARGET_SPEED)
        self.maxSpeed = float(cfg.GYM_ENV.MAX_SPEED)
        self.maxAcc = float(cfg.GYM_ENV.MAX_ACC)
        self.LANE_WIDTH = float(cfg.CARLA.LANE_WIDTH)
        self.N_SPAWN_CARS = int(cfg.TRAFFIC_MANAGER.N_SPAWN_CARS)

        # frenet
        self.f_idx = 0
        self.init_s = None  # initial frenet s value - will be updated in reset function
        self.ego_s = None
        self.ego_d = None
        self.max_s = int(cfg.CARLA.MAX_S)
        self.radar_distance = 60
        self.track_length = int(cfg.GYM_ENV.TRACK_LENGTH)
        self.look_back = int(cfg.GYM_ENV.LOOK_BACK)
        self.time_step = int(cfg.GYM_ENV.TIME_STEP)
        self.loop_break = int(cfg.GYM_ENV.LOOP_BREAK)
        self.effective_distance_from_vehicle_ahead = int(cfg.GYM_ENV.DISTN_FRM_VHCL_AHD)
        self.lanechange = False
        self.is_first_path = True

        # RL
        self.w_speed = int(cfg.RL.W_SPEED)
        self.w_r_speed = int(cfg.RL.W_R_SPEED)

        self.min_speed_gain = float(cfg.RL.MIN_SPEED_GAIN)
        self.min_speed_loss = float(cfg.RL.MIN_SPEED_LOSS)
        self.lane_change_reward = float(cfg.RL.LANE_CHANGE_REWARD)
        self.lane_change_penalty = float(cfg.RL.LANE_CHANGE_PENALTY)

        self.off_the_road_penalty = int(cfg.RL.OFF_THE_ROAD)
        self.collision_penalty = int(cfg.RL.COLLISION)

        if cfg.GYM_ENV.FIXED_REPRESENTATION:
            self.low_state = np.array([[-1 for _ in range(self.look_back)] for _ in range(16)])
            self.high_state = np.array([[1 for _ in range(self.look_back)] for _ in range(16)])
        else:
            self.low_state = np.array(
                [[-1 for _ in range(self.look_back)] for _ in range(int(self.N_SPAWN_CARS + 1) * 2 + 1)])
            self.high_state = np.array(
                [[1 for _ in range(self.look_back)] for _ in range(int(self.N_SPAWN_CARS + 1) * 2 + 1)])

        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(self.time_step + 1, 15),
                                                dtype=np.float32)
        action_low = np.array([-1, -1])
        action_high = np.array([1, 1])
        self.action_space = gym.spaces.Box(low=action_low, high=action_high, shape=(2,), dtype=np.float32)
        # [cn, ..., c1, c0, normalized yaw angle, normalized speed error] => ci: coefficients
        self.state = np.zeros_like(self.observation_space.sample())
        self.state_r = None

        # instances
        self.ego = None
        self.ego_deq_s = deque(maxlen=50)
        self.ego_deq_d = deque(maxlen=50)
        self.ego_los_sensor = None
        self.module_manager = None
        self.world_module = None
        self.traffic_module = None
        self.hud_module = None
        self.input_module = None
        self.control_module = None
        self.init_transform = None  # ego initial transform to recover at each episode
        self.acceleration_ = 0
        self.eps_rew = 0

        """
        ['EGO', 'LEADING', 'FOLLOWING', 'LEFT', 'LEFT_UP', 'LEFT_DOWN', 'LLEFT', 'LLEFT_UP',
        'LLEFT_DOWN', 'RIGHT', 'RIGHT_UP', 'RIGHT_DOWN', 'RRIGHT', 'RRIGHT_UP', 'RRIGHT_DOWN']
        """
        self.actor_enumerated_dict = {}
        self.actor_dyenumerated_dict = {}
        self.actor_step_dict = {}
        self.actor_step_traindata = []
        self.DEQSIZE = 200
        self.deq_ego_speed = deque(maxlen=self.DEQSIZE)
        self.deq_leading_s = deque(maxlen=self.DEQSIZE)
        self.deq_follow_s = deque(maxlen=self.DEQSIZE)
        self.deq_left_s = deque(maxlen=self.DEQSIZE)
        self.deq_leftup_s = deque(maxlen=self.DEQSIZE)
        self.deq_leftdown_s = deque(maxlen=self.DEQSIZE)
        self.deq_lleft_s = deque(maxlen=self.DEQSIZE)
        self.deq_lleftup_s = deque(maxlen=self.DEQSIZE)
        self.deq_llftdown_s = deque(maxlen=self.DEQSIZE)
        self.deq_right_s = deque(maxlen=self.DEQSIZE)
        self.deq_rightup_s = deque(maxlen=self.DEQSIZE)
        self.deq_rightdown_s = deque(maxlen=self.DEQSIZE)
        self.deq_rright_s = deque(maxlen=self.DEQSIZE)
        self.deq_rrightup_s = deque(maxlen=self.DEQSIZE)
        self.deq_rrightdown_s = deque(maxlen=self.DEQSIZE)
        self.deq_list = [self.deq_ego_speed, self.deq_leading_s, self.deq_follow_s, self.deq_left_s,
                         self.deq_leftup_s, self.deq_leftdown_s, self.deq_lleft_s,self.deq_lleftup_s,
                         self.deq_llftdown_s, self.deq_right_s,self.deq_rightup_s, self.deq_rightdown_s,
                         self.deq_rright_s, self.deq_rrightup_s, self.deq_rrightdown_s]

        self.deq_ego_speed_step = deque()
        self.deq_leading_step = deque()
        self.deq_follow_step = deque()
        self.deq_left_step = deque()
        self.deq_leftup_step = deque()
        self.deq_leftdown_step = deque()
        self.deq_lleft_step = deque()
        self.deq_lleftup_step = deque()
        self.deq_llftdown_step = deque()
        self.deq_right_step = deque()
        self.deq_rightup_step = deque()
        self.deq_rightdown_step = deque()
        self.deq_rright_step = deque()
        self.deq_rrightup_step = deque()
        self.deq_rrightdown_step = deque()
        self.deq_step_list = [self.deq_ego_speed_step, self.deq_leading_step, self.deq_follow_step,
                              self.deq_left_step, self.deq_leftup_step, self.deq_leftdown_step,
                              self.deq_lleft_step, self.deq_lleftup_step, self.deq_llftdown_step,
                              self.deq_right_step,self.deq_rightup_step, self.deq_rightdown_step,
                              self.deq_rright_step, self.deq_rrightup_step, self.deq_rrightdown_step]

        self.actor_enumeration = []
        self.side_window = 5  # times 2 to make adjacent window

        self.motionPlanner = None
        self.vehicleController = None
        self.episode = 0
        self.scores, self.scores_r, self.scores_g, self.scores_a, self.episodes, self.episodes_r, self.average, self.average_r, self.average_g, self.average_a = [], [], [], [], [], [], [], [], [], []


        if float(cfg.CARLA.DT) > 0:
            self.dt = float(cfg.CARLA.DT)
        else:
            self.dt = 0.05

        if self.human_ctrl:
            keyboard.add_hotkey('left', self.on_left_pressed)
            keyboard.add_hotkey('right', self.on_right_pressed)

        # reward func
        self.porf = PORF()

    def on_left_pressed(self):
        self.human_ctrl_buffer.append(-1)
        print("left键被按下")

    def on_right_pressed(self):
        self.human_ctrl_buffer.append(1)
        print("right键被按下")

    def seed(self, seed=None):
        pass

    def get_vehicle_ahead(self, ego_s, ego_d, ego_init_d, ego_target_d):
        """
        This function returns the values for the leading actor in front of the ego vehicle. When there is lane-change
        it is important to consider actor in the current lane and target lane. If leading actor in the current lane is
        too close than it is considered to be vehicle_ahead other wise target lane is prioritized.
        """

        distance = self.effective_distance_from_vehicle_ahead
        others_s = [0 for _ in range(self.N_SPAWN_CARS)]
        others_d = [0 for _ in range(self.N_SPAWN_CARS)]
        for i, actor in enumerate(self.traffic_module.actors_batch):
            act_s, act_d = actor['Frenet State'][0][-1], actor['Frenet State'][1]
            others_s[i] = act_s
            others_d[i] = act_d

        init_lane_d_idx = \
            np.where((abs(np.array(others_d) - ego_d) < 1.75) * (abs(np.array(others_d) - ego_init_d) < 1))[0]

        init_lane_strict_d_idx = \
            np.where((abs(np.array(others_d) - ego_d) < 0.4) * (abs(np.array(others_d) - ego_init_d) < 1))[0]

        target_lane_d_idx = \
            np.where((abs(np.array(others_d) - ego_d) < 3.3) * (abs(np.array(others_d) - ego_target_d) < 1))[0]

        if len(init_lane_d_idx) and len(target_lane_d_idx) == 0:
            return None # no vehicle ahead
        else:
            init_lane_s = np.array(others_s)[init_lane_d_idx]
            init_s_idx = np.concatenate((np.array(init_lane_d_idx).reshape(-1, 1), (init_lane_s - ego_s).reshape(-1, 1),)
                                        , axis=1)
            sorted_init_s_idx = init_s_idx[init_s_idx[:, 1].argsort()]

            init_lane_strict_s = np.array(others_s)[init_lane_strict_d_idx]
            init_strict_s_idx = np.concatenate(
                (np.array(init_lane_strict_d_idx).reshape(-1, 1), (init_lane_strict_s - ego_s).reshape(-1, 1),)
                , axis=1)
            sorted_init_strict_s_idx = init_strict_s_idx[init_strict_s_idx[:, 1].argsort()]

            target_lane_s = np.array(others_s)[target_lane_d_idx]
            target_s_idx = np.concatenate((np.array(target_lane_d_idx).reshape(-1, 1),
                                                (target_lane_s - ego_s).reshape(-1, 1),), axis=1)
            sorted_target_s_idx = target_s_idx[target_s_idx[:, 1].argsort()]

            if any(sorted_init_s_idx[:, 1][sorted_init_s_idx[:, 1] <= 10] > 0):
                vehicle_ahead_idx = int(sorted_init_s_idx[:, 0][sorted_init_s_idx[:, 1] > 0][0])
            elif any(sorted_init_strict_s_idx[:, 1][sorted_init_strict_s_idx[:, 1] <= distance] > 0):
                vehicle_ahead_idx = int(sorted_init_strict_s_idx[:, 0][sorted_init_strict_s_idx[:, 1] > 0][0])
            elif any(sorted_target_s_idx[:, 1][sorted_target_s_idx[:, 1] <= distance] > 0):
                vehicle_ahead_idx = int(sorted_target_s_idx[:, 0][sorted_target_s_idx[:, 1] > 0][0])
            else:
                return None

            # print(others_s[vehicle_ahead_idx] - ego_s, others_d[vehicle_ahead_idx], ego_d)

            return self.traffic_module.actors_batch[vehicle_ahead_idx]['Actor']

    def get_vehicle_left(self, ego_s, ego_d, ego_init_d, ego_target_d):
        distance = self.effective_distance_from_vehicle_ahead
        others_s = [0 for _ in range(self.N_SPAWN_CARS)]
        others_d = [0 for _ in range(self.N_SPAWN_CARS)]
        for i, actor in enumerate(self.traffic_module.actors_batch):
            act_s, act_d = actor['Frenet State'][0][-1], actor['Frenet State'][1]
            others_s[i] = act_s
            others_d[i] = act_d
        init_lane_d_idx = \
            np.where(((np.array(others_d) - ego_d) < -3) * ((np.array(others_d) - ego_d) > -4))[0]

        if len(init_lane_d_idx) == 0:
            return None # no vehicle left
        else:
            init_lane_s = np.array(others_s)[init_lane_d_idx]

    def enumerate_actors(self, ego_s, ego_d, ego_deq_s, ego_deq_d):
        """
        Given the traffic actors and ego_state this fucntion enumerate actors, calculates their relative positions with
        to ego and assign them to actor_enumerated_dict.
        Keys to be updated: ['LEADING', 'FOLLOWING', 'LEFT', 'LEFT_UP', 'LEFT_DOWN', 'LLEFT', 'LLEFT_UP',
        'LLEFT_DOWN', 'RIGHT', 'RIGHT_UP', 'RIGHT_DOWN', 'RRIGHT', 'RRIGHT_UP', 'RRIGHT_DOWN']
        """
        # 采样一帧
        self.actor_enumeration = []
        enumerated_dict = {}
        enumerated_dict['EGO'] = {'NORM_S': [], 'NORM_D': [],'S': [], 'D': [], 'SPEED': []}
        enumerated_dict['EGO']['S'].append(ego_s)
        enumerated_dict['EGO']['D'].append(ego_d)
        enumerated_dict['EGO']['NORM_S'].append((ego_s - self.init_s) / self.track_length)
        enumerated_dict['EGO']['NORM_D'].append(round((ego_d + self.LANE_WIDTH) / (3 * self.LANE_WIDTH), 2))
        last_speed = get_speed(self.ego)
        enumerated_dict['EGO']['SPEED'].append(last_speed / self.maxSpeed)

        others_s = [0 for _ in range(self.N_SPAWN_CARS)]
        others_d = [0 for _ in range(self.N_SPAWN_CARS)]
        others_id = [0 for _ in range(self.N_SPAWN_CARS)]
        for i, actor in enumerate(self.traffic_module.actors_batch):
            act_s, act_d = actor['Frenet State']
            others_s[i] = act_s[-1]
            others_d[i] = act_d
            others_id[i] = actor['Actor'].id

        def append_actor(x_lane_d_idx, actor_names=None):
            # actor names example: ['left', 'leftUp', 'leftDown']
            x_lane_s = np.array(others_s)[x_lane_d_idx]
            x_lane_id = np.array(others_id)[x_lane_d_idx]
            s_idx = np.concatenate((np.array(x_lane_d_idx).reshape(-1, 1), (x_lane_s - ego_s).reshape(-1, 1),
                                    x_lane_id.reshape(-1, 1)), axis=1)
            sorted_s_idx = s_idx[s_idx[:, 1].argsort()] # [idx s id]

            # 在side_window范围内取纵向距离最近的车辆id
            # self.actor_enumeration.append(
            #     others_id[int(sorted_s_idx[:, 0][abs(sorted_s_idx[:, 1]) < self.side_window][0])] if (any(abs(sorted_s_idx[:, 1][abs(sorted_s_idx[:, 1]) <= self.side_window]) >= -self.side_window)) else -1)
            # # 大于side_window取纵向距离最近的车辆id
            # self.actor_enumeration.append(
            #     others_id[int(sorted_s_idx[:, 0][sorted_s_idx[:, 1] > self.side_window][0])] if (any(sorted_s_idx[:, 1][sorted_s_idx[:, 1] > 0] > self.side_window)) else -1)
            # # 小于-side_window取纵向距离最近的车辆id
            # self.actor_enumeration.append(
            #     others_id[int(sorted_s_idx[:, 0][sorted_s_idx[:, 1] < -self.side_window][-1])] if (any(sorted_s_idx[:, 1][sorted_s_idx[:, 1] < 0] < -self.side_window)) else -1)
            # 在side_window范围内取纵向距离最近的车辆id
            self.actor_enumeration.append(
                others_id[int(sorted_s_idx[:, 0][abs(sorted_s_idx[:, 1]) < self.side_window][0])] if (any(abs(sorted_s_idx[:, 1][abs(sorted_s_idx[:, 1]) <= self.side_window]) >= -self.side_window)) else -1)
            # 大于side_window取纵向距离最近的车辆id
            self.actor_enumeration.append(
                others_id[int(sorted_s_idx[:, 0][sorted_s_idx[:, 1] > self.side_window][0])] if (any(sorted_s_idx[:, 1][sorted_s_idx[:, 1] > 0] > self.side_window) and any(sorted_s_idx[:, 1][sorted_s_idx[:, 1] > 0] < self.radar_distance)) else -1)
            # 小于-side_window取纵向距离最近的车辆id
            self.actor_enumeration.append(
                others_id[int(sorted_s_idx[:, 0][sorted_s_idx[:, 1] < -self.side_window][-1])] if (any(sorted_s_idx[:, 1][sorted_s_idx[:, 1] < 0] < -self.side_window) and any(sorted_s_idx[:, 1][sorted_s_idx[:, 1] > 0] > -self.radar_distance)) else -1)

        # --------------------------------------------- ego lane -------------------------------------------------
        same_lane_d_idx = np.where(abs(np.array(others_d) - ego_d) < 1)[0] # 筛选出横向距离小于1m的actor
        if len(same_lane_d_idx) == 0:
            self.actor_enumeration.append(-2)
            self.actor_enumeration.append(-2)

        else:
            same_lane_s = np.array(others_s)[same_lane_d_idx]
            same_lane_id = np.array(others_id)[same_lane_d_idx]
            same_s_idx = np.concatenate((np.array(same_lane_d_idx).reshape(-1, 1), (same_lane_s - ego_s).reshape(-1, 1),
                                         same_lane_id.reshape(-1, 1)), axis=1)
            sorted_same_s_idx = same_s_idx[same_s_idx[:, 1].argsort()] # 按照纵向距离排序 离ego最近的[0]->[idx, s, id]
            # print(sorted_same_s_idx)
            # 前车id
            self.actor_enumeration.append(others_id[int(sorted_same_s_idx[:, 0][sorted_same_s_idx[:, 1] > 0][0])] if (any(sorted_same_s_idx[:, 1] > 0)) else -1)
            # 后车id 无候车则 -1
            self.actor_enumeration.append(others_id[int(sorted_same_s_idx[:, 0][sorted_same_s_idx[:, 1] < 0][-1])] if (any(sorted_same_s_idx[:, 1] < 0)) else -1)
            # print(self.actor_enumeration)

        # --------------------------------------------- left lane -------------------------------------------------
        left_lane_d_idx = np.where(((np.array(others_d) - ego_d) < -3) * ((np.array(others_d) - ego_d) > -4))[0]
        # 左侧无车道 -2
        if ego_d < -1.75:
            self.actor_enumeration += [-2, -2, -2]
        # 左侧无车 -1
        elif len(left_lane_d_idx) == 0:
            self.actor_enumeration += [-1, -1, -1]
        # 左侧有车 车辆idx
        else:
            append_actor(left_lane_d_idx)

        # ------------------------------------------- two left lane -----------------------------------------------
        lleft_lane_d_idx = np.where(((np.array(others_d) - ego_d) < -6.5) * ((np.array(others_d) - ego_d) > -7.5))[0]

        if ego_d < 1.75:
            self.actor_enumeration += [-2, -2, -2]

        elif len(lleft_lane_d_idx) == 0:
            self.actor_enumeration += [-1, -1, -1]

        else:
            append_actor(lleft_lane_d_idx)

            # ---------------------------------------------- rigth lane --------------------------------------------------
        right_lane_d_idx = np.where(((np.array(others_d) - ego_d) > 3) * ((np.array(others_d) - ego_d) < 4))[0]
        if ego_d > 5.25:
            self.actor_enumeration += [-2, -2, -2]

        elif len(right_lane_d_idx) == 0:
            self.actor_enumeration += [-1, -1, -1]

        else:
            append_actor(right_lane_d_idx)

        # ------------------------------------------- two rigth lane --------------------------------------------------
        rright_lane_d_idx = np.where(((np.array(others_d) - ego_d) > 6.5) * ((np.array(others_d) - ego_d) < 7.5))[0]
        if ego_d > 1.75:
            self.actor_enumeration += [-2, -2, -2]
        elif len(rright_lane_d_idx) == 0:
            self.actor_enumeration += [-1, -1, -1]
        else:
            append_actor(rright_lane_d_idx)

        # Fill enumerated actor values

        actor_id_s_d = {}
        norm_s = []
        # norm_d = []
        for actor in self.traffic_module.actors_batch:
            actor_id_s_d[actor['Actor'].id] = actor['Frenet State']
        actor_id_vehicle = {}
        for actor in self.traffic_module.actors_batch:
            actor_id_vehicle[actor['Actor'].id] = actor['Actor']

        # 显示选取的车辆
        for i, actor_id in enumerate(self.actor_enumeration):
            if actor_id >= 0:
                vehicle = actor_id_vehicle[actor_id]
                loc = vehicle.get_location()
                loc.z += 1.5
                self.world_module.world.debug.draw_point(loc, size=0.05, color=carla.Color(0, 255, 0), life_time=3)

        for i, actor_id in enumerate(self.actor_enumeration):
            if actor_id >= 0:
                actor_norm_s = []
                act_s_hist, act_d = actor_id_s_d[actor_id]  # act_s_hist:list act_d:float
                for act_s, ego_s in zip(list(act_s_hist)[-self.look_back:], list(ego_deq_s)[-self.look_back:]):
                    actor_norm_s.append((act_s - ego_s) / self.max_s)
                norm_s.append(actor_norm_s)
            #    norm_d[i] = (act_d - ego_d) / (3 * self.LANE_WIDTH)
            # -1:empty lane, -2:no lane
            else:
                norm_s.append(actor_id)

        # How to fill actor_s when there is no lane or lane is empty. relative_norm_s to ego vehicle
        # emp_ln_max = 0.03
        # emp_ln_min = -0.03
        # no_ln_down = -0.03
        # no_ln_up = 0.004
        # no_ln = 0.001
        emp_ln_max = 1.0
        emp_ln_min = -1.0
        no_ln_down = 0.0
        no_ln_up = 0.0
        no_ln = 0.0
        if norm_s[0] not in (-1, -2):
            enumerated_dict['LEADING'] = {'S': norm_s[0]}
        else:
            enumerated_dict['LEADING'] = {'S': [emp_ln_max]}

        if norm_s[1] not in (-1, -2):
            enumerated_dict['FOLLOWING'] = {'S': norm_s[1]}
        else:
            enumerated_dict['FOLLOWING'] = {'S': [emp_ln_min]}

        if norm_s[2] not in (-1, -2):
            enumerated_dict['LEFT'] = {'S': norm_s[2]}
        else:
            enumerated_dict['LEFT'] = {'S': [emp_ln_min] if norm_s[2] == -1 else [no_ln]}

        if norm_s[3] not in (-1, -2):
            enumerated_dict['LEFT_UP'] = {'S': norm_s[3]}
        else:
            enumerated_dict['LEFT_UP'] = {'S': [emp_ln_max] if norm_s[3] == -1 else [no_ln_up]}

        if norm_s[4] not in (-1, -2):
            enumerated_dict['LEFT_DOWN'] = {'S': norm_s[4]}
        else:
            enumerated_dict['LEFT_DOWN'] = {'S': [emp_ln_min] if norm_s[4] == -1 else [no_ln_down]}

        if norm_s[5] not in (-1, -2):
            enumerated_dict['LLEFT'] = {'S': norm_s[5]}
        else:
            enumerated_dict['LLEFT'] = {'S': [emp_ln_min] if norm_s[5] == -1 else [no_ln]}

        if norm_s[6] not in (-1, -2):
            enumerated_dict['LLEFT_UP'] = {'S': norm_s[6]}
        else:
            enumerated_dict['LLEFT_UP'] = {'S': [emp_ln_max] if norm_s[6] == -1 else [no_ln_up]}

        if norm_s[7] not in (-1, -2):
            enumerated_dict['LLEFT_DOWN'] = {'S': norm_s[7]}
        else:
            enumerated_dict['LLEFT_DOWN'] = {'S': [emp_ln_min] if norm_s[7] == -1 else [no_ln_down]}

        if norm_s[8] not in (-1, -2):
            enumerated_dict['RIGHT'] = {'S': norm_s[8]}
        else:
            enumerated_dict['RIGHT'] = {'S': [emp_ln_min] if norm_s[8] == -1 else [no_ln]}

        if norm_s[9] not in (-1, -2):
            enumerated_dict['RIGHT_UP'] = {'S': norm_s[9]}
        else:
            enumerated_dict['RIGHT_UP'] = {'S': [emp_ln_max] if norm_s[9] == -1 else [no_ln_up]}

        if norm_s[10] not in (-1, -2):
            enumerated_dict['RIGHT_DOWN'] = {'S': norm_s[10]}
        else:
            enumerated_dict['RIGHT_DOWN'] = {'S': [emp_ln_min] if norm_s[10] == -1 else [no_ln_down]}

        if norm_s[11] not in (-1, -2):
            enumerated_dict['RRIGHT'] = {'S': norm_s[11]}
        else:
            enumerated_dict['RRIGHT'] = {'S': [emp_ln_min] if norm_s[11] == -1 else [no_ln]}

        if norm_s[12] not in (-1, -2):
            enumerated_dict['RRIGHT_UP'] = {'S': norm_s[12]}
        else:
            enumerated_dict['RRIGHT_UP'] = {'S': [emp_ln_max] if norm_s[12] == -1 else [no_ln_up]}

        if norm_s[13] not in (-1, -2):
            enumerated_dict['RRIGHT_DOWN'] = {'S': norm_s[13]}
        else:
            enumerated_dict['RRIGHT_DOWN'] = {'S': [emp_ln_min] if norm_s[13] == -1 else [no_ln_down]}

        return enumerated_dict

    def fix_representation(self, dict):
        """
        Given the traffic actors fill the desired tensor with appropriate values and time_steps
        """
        self.actor_enumerated_dict = {
        'EGO': {'SPEED': []},
        'LEADING': {'S': []},
        'FOLLOWING': {'S': []},
        'LEFT': {'S': []},
        'LEFT_UP': {'S': []},
        'LEFT_DOWN': {'S': []},
        'LLEFT': {'S': []},
        'LLEFT_UP': {'S': []},
        'LLEFT_DOWN': {'S': []},
        'RIGHT': {'S': []},
        'RIGHT_UP': {'S': []},
        'RIGHT_DOWN': {'S': []},
        'RRIGHT': {'S': []},
        'RRIGHT_UP': {'S': []},
        'RRIGHT_DOWN': {'S': []}
        }
        self.actor_enumerated_dict['EGO']['SPEED'].extend(dict['EGO']['SPEED'][-1] for _ in range(self.look_back))
        self.actor_enumerated_dict['LEADING']['S'].extend(dict['LEADING']['S'][-1] for _ in range(self.look_back))
        self.actor_enumerated_dict['FOLLOWING']['S'].extend(dict['FOLLOWING']['S'][-1] for _ in range(self.look_back))
        self.actor_enumerated_dict['LEFT']['S'].extend(dict['LEFT']['S'][-1] for _ in range(self.look_back))
        self.actor_enumerated_dict['LEFT_UP']['S'].extend(dict['LEFT_UP']['S'][-1] for _ in range(self.look_back))
        self.actor_enumerated_dict['LEFT_DOWN']['S'].extend(dict['LEFT_DOWN']['S'][-1] for _ in range(self.look_back))
        self.actor_enumerated_dict['LLEFT']['S'].extend(dict['LLEFT']['S'][-1] for _ in range(self.look_back))
        self.actor_enumerated_dict['LLEFT_UP']['S'].extend(dict['LLEFT_UP']['S'][-1] for _ in range(self.look_back))
        self.actor_enumerated_dict['LLEFT_DOWN']['S'].extend(dict['LLEFT_DOWN']['S'][-1] for _ in range(self.look_back))
        self.actor_enumerated_dict['RIGHT']['S'].extend(dict['RIGHT']['S'][-1] for _ in range(self.look_back))
        self.actor_enumerated_dict['RIGHT_UP']['S'].extend(dict['RIGHT_UP']['S'][-1] for _ in range(self.look_back))
        self.actor_enumerated_dict['RIGHT_DOWN']['S'].extend(dict['RIGHT_DOWN']['S'][-1] for _ in range(self.look_back))
        self.actor_enumerated_dict['RRIGHT']['S'].extend(dict['RRIGHT']['S'][-1] for _ in range(self.look_back))
        self.actor_enumerated_dict['RRIGHT_UP']['S'].extend(dict['RRIGHT_UP']['S'][-1] for _ in range(self.look_back))
        self.actor_enumerated_dict['RRIGHT_DOWN']['S'].extend(dict['RRIGHT_DOWN']['S'][-1] for _ in range(self.look_back))
        # for act_values in self.actor_enumerated_dict.values():
        #     act_values['S'].extend(act_values['S'][-1] for _ in range(self.look_back - len(act_values['S'])))

        _range = np.arange(-self.look_back, -1, int(np.ceil(self.look_back / self.time_step)), dtype=int) # add last observation
        _range = np.append(_range, -1)[-5:]

        lstm_obs = np.concatenate((np.array(self.actor_enumerated_dict['EGO']['SPEED'])[_range],
                                   np.array(self.actor_enumerated_dict['LEADING']['S'])[_range],
                                   np.array(self.actor_enumerated_dict['FOLLOWING']['S'])[_range],
                                   np.array(self.actor_enumerated_dict['LEFT']['S'])[_range],
                                   np.array(self.actor_enumerated_dict['LEFT_UP']['S'])[_range],
                                   np.array(self.actor_enumerated_dict['LEFT_DOWN']['S'])[_range],
                                   np.array(self.actor_enumerated_dict['LLEFT']['S'])[_range],
                                   np.array(self.actor_enumerated_dict['LLEFT_UP']['S'])[_range],
                                   np.array(self.actor_enumerated_dict['LLEFT_DOWN']['S'])[_range],
                                   np.array(self.actor_enumerated_dict['RIGHT']['S'])[_range],
                                   np.array(self.actor_enumerated_dict['RIGHT_UP']['S'])[_range],
                                   np.array(self.actor_enumerated_dict['RIGHT_DOWN']['S'])[_range],
                                   np.array(self.actor_enumerated_dict['RRIGHT']['S'])[_range],
                                   np.array(self.actor_enumerated_dict['RRIGHT_UP']['S'])[_range],
                                   np.array(self.actor_enumerated_dict['RRIGHT_DOWN']['S'])[_range]),
                                  axis=0)

        return lstm_obs.reshape(self.observation_space.shape[1], -1).transpose()  # state shape:(5,15)

    def Dynamic_state_update(self, dict):
        """
        Given the traffic actors fill the desired tensor with appropriate values and time_steps
        """
        self.actor_dyenumerated_dict = {
        'EGO': {'SPEED': []},
        'LEADING': {'S': []},
        'FOLLOWING': {'S': []},
        'LEFT': {'S': []},
        'LEFT_UP': {'S': []},
        'LEFT_DOWN': {'S': []},
        'LLEFT': {'S': []},
        'LLEFT_UP': {'S': []},
        'LLEFT_DOWN': {'S': []},
        'RIGHT': {'S': []},
        'RIGHT_UP': {'S': []},
        'RIGHT_DOWN': {'S': []},
        'RRIGHT': {'S': []},
        'RRIGHT_UP': {'S': []},
        'RRIGHT_DOWN': {'S': []}
        }
        cur_speed = get_speed(self.ego)
        self.deq_ego_speed.append(cur_speed / self.maxSpeed)
        self.deq_leading_s.append(dict['LEADING']['S'][-1])
        self.deq_follow_s.append(dict['FOLLOWING']['S'][-1])
        self.deq_left_s.append(dict['LEFT']['S'][-1])
        self.deq_leftup_s.append(dict['LEFT_UP']['S'][-1])
        self.deq_leftdown_s.append(dict['LEFT_DOWN']['S'][-1])
        self.deq_lleft_s.append(dict['LLEFT']['S'][-1])
        self.deq_lleftup_s.append(dict['LEFT_UP']['S'][-1])
        self.deq_llftdown_s.append(dict['LLEFT_DOWN']['S'][-1])
        self.deq_right_s.append(dict['RIGHT']['S'][-1])
        self.deq_rightup_s.append(dict['RIGHT_UP']['S'][-1])
        self.deq_rightdown_s.append(dict['RIGHT_DOWN']['S'][-1])
        self.deq_rright_s.append(dict['RRIGHT']['S'][-1])
        self.deq_rrightup_s.append(dict['RRIGHT_UP']['S'][-1])
        self.deq_rrightdown_s.append(dict['RRIGHT_DOWN']['S'][-1])

        # 填充
        for deq in self.deq_list:
            if len(deq) < self.DEQSIZE:
                for _ in range(self.DEQSIZE - len(deq)):
                    deq.appendleft(deq[0])

        self.actor_dyenumerated_dict['EGO']['SPEED'] = list(self.deq_ego_speed)
        self.actor_dyenumerated_dict['LEADING']['S'] = list(self.deq_leading_s)
        self.actor_dyenumerated_dict['FOLLOWING']['S'] = list(self.deq_follow_s)
        self.actor_dyenumerated_dict['LEFT']['S'] = list(self.deq_left_s)
        self.actor_dyenumerated_dict['LEFT_UP']['S'] = list(self.deq_leftup_s)
        self.actor_dyenumerated_dict['LEFT_DOWN']['S'] = list(self.deq_leftdown_s)
        self.actor_dyenumerated_dict['LLEFT']['S'] = list(self.deq_lleft_s)
        self.actor_dyenumerated_dict['LLEFT_UP']['S'] = list(self.deq_lleftup_s)
        self.actor_dyenumerated_dict['LLEFT_DOWN']['S'] = list(self.deq_llftdown_s)
        self.actor_dyenumerated_dict['RIGHT']['S'] = list(self.deq_right_s)
        self.actor_dyenumerated_dict['RIGHT_UP']['S'] = list(self.deq_rightup_s)
        self.actor_dyenumerated_dict['RIGHT_DOWN']['S'] = list(self.deq_rightdown_s)
        self.actor_dyenumerated_dict['RRIGHT']['S'] = list(self.deq_rright_s)
        self.actor_dyenumerated_dict['RRIGHT_UP']['S'] = list(self.deq_rrightup_s)
        self.actor_dyenumerated_dict['RRIGHT_DOWN']['S'] = list(self.deq_rrightdown_s)

        _range = np.arange(-self.DEQSIZE, -1, int(np.ceil(self.DEQSIZE / 5)), dtype=int) # add last observation
        _range = np.append(_range, -1)[-5:]

        lstm_obs = np.concatenate((np.array(self.actor_dyenumerated_dict['EGO']['SPEED'])[_range],
                                   np.array(self.actor_dyenumerated_dict['LEADING']['S'])[_range],
                                   np.array(self.actor_dyenumerated_dict['FOLLOWING']['S'])[_range],
                                   np.array(self.actor_dyenumerated_dict['LEFT']['S'])[_range],
                                   np.array(self.actor_dyenumerated_dict['LEFT_UP']['S'])[_range],
                                   np.array(self.actor_dyenumerated_dict['LEFT_DOWN']['S'])[_range],
                                   np.array(self.actor_dyenumerated_dict['LLEFT']['S'])[_range],
                                   np.array(self.actor_dyenumerated_dict['LLEFT_UP']['S'])[_range],
                                   np.array(self.actor_dyenumerated_dict['LLEFT_DOWN']['S'])[_range],
                                   np.array(self.actor_dyenumerated_dict['RIGHT']['S'])[_range],
                                   np.array(self.actor_dyenumerated_dict['RIGHT_UP']['S'])[_range],
                                   np.array(self.actor_dyenumerated_dict['RIGHT_DOWN']['S'])[_range],
                                   np.array(self.actor_dyenumerated_dict['RRIGHT']['S'])[_range],
                                   np.array(self.actor_dyenumerated_dict['RRIGHT_UP']['S'])[_range],
                                   np.array(self.actor_dyenumerated_dict['RRIGHT_DOWN']['S'])[_range]),
                                  axis=0)

        return lstm_obs.reshape(self.observation_space.shape[1], -1).transpose()  # state shape:(5,15)

    def step_state_update(self, dict):
        self.actor_step_dict = {
        'EGO': {'SPEED': []},
        'LEADING': {'S': []},
        'FOLLOWING': {'S': []},
        'LEFT': {'S': []},
        'LEFT_UP': {'S': []},
        'LEFT_DOWN': {'S': []},
        'LLEFT': {'S': []},
        'LLEFT_UP': {'S': []},
        'LLEFT_DOWN': {'S': []},
        'RIGHT': {'S': []},
        'RIGHT_UP': {'S': []},
        'RIGHT_DOWN': {'S': []},
        'RRIGHT': {'S': []},
        'RRIGHT_UP': {'S': []},
        'RRIGHT_DOWN': {'S': []}
        }
        cur_speed = get_speed(self.ego)
        self.deq_ego_speed_step.append(cur_speed / self.maxSpeed)
        self.deq_leading_step.append(dict['LEADING']['S'][-1])
        self.deq_follow_step.append(dict['FOLLOWING']['S'][-1])
        self.deq_left_step.append(dict['LEFT']['S'][-1])
        self.deq_leftup_step.append(dict['LEFT_UP']['S'][-1])
        self.deq_leftdown_step.append(dict['LEFT_DOWN']['S'][-1])
        self.deq_lleft_step.append(dict['LLEFT']['S'][-1])
        self.deq_lleftup_step.append(dict['LEFT_UP']['S'][-1])
        self.deq_llftdown_step.append(dict['LLEFT_DOWN']['S'][-1])
        self.deq_right_step.append(dict['RIGHT']['S'][-1])
        self.deq_rightup_step.append(dict['RIGHT_UP']['S'][-1])
        self.deq_rightdown_step.append(dict['RIGHT_DOWN']['S'][-1])
        self.deq_rright_step.append(dict['RRIGHT']['S'][-1])
        self.deq_rrightup_step.append(dict['RRIGHT_UP']['S'][-1])
        self.deq_rrightdown_step.append(dict['RRIGHT_DOWN']['S'][-1])

        for deq in self.deq_step_list:
            if len(deq) < 5:
                for _ in range(5 - len(deq)):
                    deq.appendleft(deq[0])

        self.actor_step_dict['EGO']['SPEED'] = list(self.deq_ego_speed_step)
        self.actor_step_dict['LEADING']['S'] = list(self.deq_leading_step)
        self.actor_step_dict['FOLLOWING']['S'] = list(self.deq_follow_step)
        self.actor_step_dict['LEFT']['S'] = list(self.deq_left_step)
        self.actor_step_dict['LEFT_UP']['S'] = list(self.deq_leftup_step)
        self.actor_step_dict['LEFT_DOWN']['S'] = list(self.deq_leftdown_step)
        self.actor_step_dict['LLEFT']['S'] = list(self.deq_lleft_step)
        self.actor_step_dict['LLEFT_UP']['S'] = list(self.deq_lleftup_step)
        self.actor_step_dict['LLEFT_DOWN']['S'] = list(self.deq_llftdown_step)
        self.actor_step_dict['RIGHT']['S'] = list(self.deq_right_step)
        self.actor_step_dict['RIGHT_UP']['S'] = list(self.deq_rightup_step)
        self.actor_step_dict['RIGHT_DOWN']['S'] = list(self.deq_rightdown_step)
        self.actor_step_dict['RRIGHT']['S'] = list(self.deq_rright_step)
        self.actor_step_dict['RRIGHT_UP']['S'] = list(self.deq_rrightup_step)
        self.actor_step_dict['RRIGHT_DOWN']['S'] = list(self.deq_rrightdown_step)

        _range = np.arange(-len(self.deq_step_list[0]), -1, int(np.ceil(len(self.deq_step_list[0]) / 5)), dtype=int)  # add last observation
        _range = np.append(_range, -1)[-5:]
        if len(_range) < 5:
            exp_l = np.array([_range[0] for _ in range(5-len(_range))], dtype=int)
            _range = np.concatenate((exp_l, _range), axis=0)
        lstm_obs = np.concatenate((np.array(self.actor_step_dict['EGO']['SPEED'])[_range],
                                   np.array(self.actor_step_dict['LEADING']['S'])[_range],
                                   np.array(self.actor_step_dict['FOLLOWING']['S'])[_range],
                                   np.array(self.actor_step_dict['LEFT']['S'])[_range],
                                   np.array(self.actor_step_dict['LEFT_UP']['S'])[_range],
                                   np.array(self.actor_step_dict['LEFT_DOWN']['S'])[_range],
                                   np.array(self.actor_step_dict['LLEFT']['S'])[_range],
                                   np.array(self.actor_step_dict['LLEFT_UP']['S'])[_range],
                                   np.array(self.actor_step_dict['LLEFT_DOWN']['S'])[_range],
                                   np.array(self.actor_step_dict['RIGHT']['S'])[_range],
                                   np.array(self.actor_step_dict['RIGHT_UP']['S'])[_range],
                                   np.array(self.actor_step_dict['RIGHT_DOWN']['S'])[_range],
                                   np.array(self.actor_step_dict['RRIGHT']['S'])[_range],
                                   np.array(self.actor_step_dict['RRIGHT_UP']['S'])[_range],
                                   np.array(self.actor_step_dict['RRIGHT_DOWN']['S'])[_range]),
                                  axis=0)
        return lstm_obs.reshape(self.observation_space.shape[1], -1).transpose()  # state shape:(5,15)


    def step(self, action=None):
        self.n_step += 1
        self.actor_enumerated_dict['EGO'] = {'NORM_S': [], 'NORM_D': [], 'S': [], 'D': [], 'SPEED': []}
        self.actor_dyenumerated_dict['EGO'] = {'NORM_S': [], 'NORM_D': [], 'S': [], 'D': [], 'SPEED': []}
        self.actor_step_dict['EGO'] = {'NORM_S': [], 'NORM_D': [], 'S': [], 'D': [], 'SPEED': []}
        if self.verbosity: print('ACTION'.ljust(15), '{:+8.6f}, {:+8.6f}'.format(float(action[0]), float(action[1])))
        if self.is_first_path:  # Episode start is bypassed
            action = [0, -1]
            self.is_first_path = False
        if self.human_ctrl and not self.is_first_path:
            if self.human_ctrl_buffer:
                action[0] = self.human_ctrl_buffer[-1]
                self.human_ctrl_buffer.clear()
            else:
                action[0] = 0
        """
                **********************************************************************************************************************
                *********************************************** Motion Planner *******************************************************
                **********************************************************************************************************************
        """

        temp = [self.ego.get_velocity(), self.ego.get_acceleration()]
        init_speed = speed = get_speed(self.ego)
        acc_vec = self.ego.get_acceleration()
        acc = math.sqrt(acc_vec.x ** 2 + acc_vec.y ** 2 + acc_vec.z ** 2)
        psi = math.radians(self.ego.get_transform().rotation.yaw)
        ego_state = [self.ego.get_location().x, self.ego.get_location().y, speed, acc, psi, temp, self.max_s]
        with open("./data_Excel/traindata_step_velocity_acc_psi.csv", "a+", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([int(self.n_step), float(speed), float(acc), float(psi)])


        fpath, self.lanechange, off_the_road, v_exp = self.motionPlanner.run_step_single_path(ego_state, self.f_idx, df_n=action[0], Tf=5,Vf_n=action[1])

        wps_to_go = len(fpath.t) - 3  # -2 bc len gives # of items not the idx of last item + 2wp controller is used
        self.f_idx = 1

        """
                **********************************************************************************************************************
                ************************************************* Controller *********************************************************
                **********************************************************************************************************************
        """
        # initialize flags
        collision = track_finished = False
        elapsed_time = lambda previous_time: time.time() - previous_time
        path_start_time = time.time()
        ego_init_d, ego_target_d = fpath.d[0], fpath.d[-1]
        # follows path until end of WPs for max 1.5 * path_time or loop counter breaks unless there is a langechange
        loop_counter = 0
        for deq in self.deq_step_list:
            deq.clear()
        while self.f_idx < wps_to_go and (elapsed_time(path_start_time) < self.motionPlanner.D_T * 1.5 or
                                          loop_counter < self.loop_break or self.lanechange):

            loop_counter += 1
            ego_state = [self.ego.get_location().x, self.ego.get_location().y,
                         math.radians(self.ego.get_transform().rotation.yaw), 0, 0, temp, self.max_s]
            self.f_idx = closest_wp_idx(ego_state, fpath, self.f_idx)
            cmdWP = [fpath.x[self.f_idx], fpath.y[self.f_idx]]
            cmdWP2 = [fpath.x[self.f_idx + 1], fpath.y[self.f_idx + 1]]

            # overwrite command speed using IDM
            self.ego_s = self.motionPlanner.estimate_frenet_state(ego_state, self.f_idx)[0]  # estimated current ego_s
            self.ego_d = fpath.d[self.f_idx]
            self.ego_deq_s.append(self.ego_s)
            self.ego_deq_d.append(self.ego_d)
            vehicle_ahead = self.get_vehicle_ahead(self.ego_s, self.ego_d, ego_init_d, ego_target_d)
            # 变速
            # self.targetSpeed = v_exp
            cmdSpeed = self.IDM.run_step(vd=self.targetSpeed, vehicle_ahead=vehicle_ahead)
            # if self.lanechange:
            #     cmdSpeed = self.IDM.run_step(vd=v_exp, vehicle_ahead=vehicle_ahead)
            # else:
            #     IDM_v = self.IDM.run_step(vd=self.avgSpeed, vehicle_ahead=vehicle_ahead)
            #     cmdSpeed = IDM_v

            # control = self.vehicleController.run_step(cmdSpeed, cmdWP)  # calculate control
            control = self.vehicleController.run_step_2_wp(cmdSpeed, cmdWP, cmdWP2)  # calculate control
            self.ego.apply_control(control)  # apply control

            """
                    **********************************************************************************************************************
                    *********************************************** Draw Waypoints *******************************************************
                    **********************************************************************************************************************
            """

            if self.world_module.args.play_mode != 0:
                for i in range(len(fpath.t)):
                    self.world_module.points_to_draw['path wp {}'.format(i)] = [
                        carla.Location(x=fpath.x[i], y=fpath.y[i]),
                        'COLOR_ALUMINIUM_0']
                self.world_module.points_to_draw['ego'] = [self.ego.get_location(), 'COLOR_SCARLET_RED_0']
                self.world_module.points_to_draw['waypoint ahead'] = carla.Location(x=cmdWP[0], y=cmdWP[1])
                self.world_module.points_to_draw['waypoint ahead 2'] = carla.Location(x=cmdWP2[0], y=cmdWP2[1])


            for i in range(len(fpath.t)):
                t = carla.Location(x=fpath.x[i], y=fpath.y[i], z=0.2)
                self.world_module.world.debug.draw_point(t, size=0.05, life_time=1)
            """
                    **********************************************************************************************************************
                    ************************************************ Update Carla ********************************************************
                    **********************************************************************************************************************
            """
            self.module_manager.tick()  # Update carla world
            # 视角
            self.view_()
            if self.auto_render:
                self.render()

            collision_hist = self.world_module.get_collision_history()
            # if ego off-the road or collided
            if any(collision_hist):
                collision = True
                break

            distance_traveled = self.ego_s - self.init_s
            if distance_traveled < -5:
                distance_traveled = self.max_s + distance_traveled
            if distance_traveled >= self.track_length:
                track_finished = True

            """
                    **********************************************************************************************************************
                    ************************************************ Update State Queue ********************************************************
                    **********************************************************************************************************************
            """
            if not cfg.GYM_ENV.FIXED_REPRESENTATION:
                frame = self.enumerate_actors(self.ego_s, self.ego_d, self.ego_deq_s, self.ego_deq_d)
                self.state = self.Dynamic_state_update(frame)
            if RLHF_TRAIN_SAMPLE:
                frame = self.enumerate_actors(self.ego_s, self.ego_d, self.ego_deq_s, self.ego_deq_d)
                self.state_r = self.step_state_update(frame)

        """
                *********************************************************************************************************************
                *********************************************** RL Observation ******************************************************
                *********************************************************************************************************************
        """

        if cfg.GYM_ENV.FIXED_REPRESENTATION:
            frame = self.enumerate_actors(self.ego_s, self.ego_d, self.ego_deq_s, self.ego_deq_d)
            self.state = self.fix_representation(frame)
            # if self.verbosity == 2:
            #     print(3 * '---EPS UPDATE---')
            #     print(TENSOR_ROW_NAMES[0].ljust(15),
            #           #      '{:+8.6f}  {:+8.6f}'.format(self.state[-1][1], self.state[-1][0]))
            #          '{:+8.6f}'.format(self.state[-1][0]))
            #     for idx in range(1, self.state.shape[1]):
            #         print(TENSOR_ROW_NAMES[idx].ljust(15), '{:+8.6f}'.format(self.state[-1][idx]))


        # if self.verbosity == 3: print(self.state)
        """
                **********************************************************************************************************************
                ********************************************* RL Reward Function *****************************************************
                **********************************************************************************************************************
        """
        # 与全局平均速度的误差
        last_speed = get_speed(self.ego)
        # e_speed = abs(self.avgSpeed - last_speed)
        l_speed = abs(self.targetSpeed - last_speed)
        # r_speed = 0.5 * self.w_r_speed * np.exp(-e_speed ** 2 / self.maxSpeed * self.w_speed)  # 0<= r_speed <= self.w_r_speed
        r_speed =  self.w_r_speed * np.exp(-l_speed ** 2 / self.maxSpeed * self.w_speed)
        #  first two path speed change increases regardless so we penalize it differently

        spd_change_percentage = (last_speed - init_speed) / init_speed if init_speed != 0 else -1
        r_laneChange = 0

        if self.lanechange and spd_change_percentage < self.min_speed_gain:
            r_laneChange = -1 * r_speed * self.lane_change_penalty  # <= 0

        elif self.lanechange:
            r_speed *= self.lane_change_reward

        positives = r_speed
        negatives = r_laneChange
        reward = positives + negatives  # r_speed * (1 - lane_change_penalty) <= reward <= r_speed * lane_change_reward
        # print(self.n_step, self.eps_rew)
        # RLHF 多个step state整合
        if RLHF_TRAIN_SAMPLE:
            self.actor_step_traindata.append(self.state_r.tolist())
        """
                **********************************************************************************************************************
                ********************************************* Episode Termination ****************************************************
                **********************************************************************************************************************
        """

        done = False
        if collision:
            self.episode += 1
            # print('Collision happened!')
            reward = self.collision_penalty
            done = True
            self.eps_rew += reward
            # print('eps rew: ', self.n_step, self.eps_rew)
            if self.verbosity: print('REWARD'.ljust(15), '{:+8.6f}'.format(reward))

        elif track_finished:
            # print('Finished the race')
            # reward = 10
            self.episode += 1
            done = True
            if off_the_road:
                reward = self.off_the_road_penalty
            self.eps_rew += reward
            # print('eps rew: ', self.n_step, self.eps_rew)
            if self.verbosity: print('REWARD'.ljust(15), '{:+8.6f}'.format(reward))

        # elif off_the_road:
        #     # print('Collision happened!')
        #     reward = self.off_the_road_penalty
        #     # done = True
        #     self.eps_rew += reward
        #     # print('eps rew: ', self.n_step, self.eps_rew)
        #     if self.verbosity: print('REWARD'.ljust(15), '{:+8.6f}'.format(reward))

        if done == False:
            self.eps_rew += reward
            # print(self.n_step, self.eps_rew)
            if self.verbosity: print('REWARD'.ljust(15), '{:+8.6f}'.format(reward))
        else:
            if RLHF_TRAIN_SAMPLE:
                step_js = {
                    "state": self.actor_step_traindata,
                    "r": reward,
                    "step": self.n_step
                }
                json_file_path = os.path.join(JSON_PATH, f'{self.episode - 1}.json')
                with open(json_file_path, "w") as f:
                    json.dump(step_js, f, indent=4)
                self.traffic_module.clean_actor()
                self.world_module.world.tick()
                self.world_module.client.stop_recorder()
            with open("./data_Excel/eps_epsR_lcCount_lcPos_lastV_col.csv", "a+", newline='') as f:
                writer = csv.writer(f)
                if collision:
                    writer.writerow([self.episode, float(self.eps_rew), self.lan_change_count, self.lan_change_positive, last_speed, 1])
                else:
                    writer.writerow(
                        [self.episode, float(self.eps_rew), self.lan_change_count, self.lan_change_positive, last_speed, 0])
            self.PlotReward(self.eps_rew, self.episode)

        return self.state, reward, done, {'reserved': 0}

    def reset(self):
        if RLHF_TRAIN_SAMPLE:
            self.actor_step_traindata = []
            log_file_path = os.path.join(LOG_PATH, f'{self.episode}.log')
            self.world_module.client.start_recorder(log_file_path)
        self.vehicleController.reset()
        self.world_module.reset()
        self.init_s = self.world_module.init_s
        init_d = self.world_module.init_d
        self.ego_s = self.init_s
        self.ego_d = init_d
        self.ego_deq_s.clear()
        self.ego_deq_d.clear()
        self.ego_deq_s.append(self.ego_s)
        self.ego_deq_d.append(self.ego_d)
        self.traffic_module.reset(self.init_s, init_d)
        self.motionPlanner.reset(self.init_s, self.world_module.init_d, df_n=0, Tf=4, Vf_n=0, optimal_path=False)
        self.f_idx = 0

        self.n_step = 0  # initialize episode steps count
        self.eps_rew = 0
        self.is_first_path = True
        # actors_norm_s_d = []  # relative frenet consecutive s and d values wrt ego
        # init_norm_d = round((init_d + self.LANE_WIDTH) / (3 * self.LANE_WIDTH), 2)
        # ego_s_list = [self.init_s for _ in range(self.look_back)]
        # ego_d_list = [init_d for _ in range(self.look_back)]

        self.actor_enumerated_dict['EGO'] = {'NORM_S': [], 'NORM_D': [], 'S': [], 'D': [], 'SPEED': []}
        self.actor_dyenumerated_dict['EGO'] = {'NORM_S': [], 'NORM_D': [], 'S': [], 'D': [], 'SPEED': []}
        self.actor_step_dict['EGO'] = {'NORM_S': [], 'NORM_D': [], 'S': [], 'D': [], 'SPEED': []}
        # 起始状态清空状态队列
        for deq in self.deq_list:
            deq.clear()
        for deq in self.deq_step_list:
            deq.clear()

        if cfg.GYM_ENV.FIXED_REPRESENTATION:
            frame = self.enumerate_actors(self.ego_s, self.ego_d, self.ego_deq_s, self.ego_deq_d)
            self.state = self.fix_representation(frame)
            # if self.verbosity == 2:
            #     print(3 * '---RESET---')
            #     print(TENSOR_ROW_NAMES[0].ljust(15),
            #           #      '{:+8.6f}  {:+8.6f}'.format(self.state[-1][1], self.state[-1][0]))
            #           '{:+8.6f}'.format(self.state[-1][0]))
            #     for idx in range(1, self.state.shape[1]):
            #         print(TENSOR_ROW_NAMES[idx].ljust(15), '{:+8.6f}'.format(self.state[-1][idx]))
        else:
            frame = self.enumerate_actors(self.ego_s, self.ego_d, self.ego_deq_s, self.ego_deq_d)
            self.state = self.Dynamic_state_update(frame)
            # Could be debugged to be used
            # pad the feature lists to recover from the cases where the length of path is less than look_back time

            # self.state = self.non_fix_representation(speeds, ego_norm_s, ego_norm_d, actors_norm_s_d)

        # ---
        # Ego starts to move slightly after being relocated when a new episode starts. Probably, ego keeps a fraction of previous acceleration after
        # being relocated. To solve this, the following procedure is needed.
        self.ego.set_simulate_physics(enabled=False)
        # for _ in range(5):
        self.module_manager.tick()
        self.ego.set_simulate_physics(enabled=True)
        # ----
        return self.state

    def PlotReward(self, score, episode):
        self.scores_r.append(score)
        self.episodes_r.append(episode)
        self.average_r.append(sum(self.scores_r[-50:]) / len(self.scores_r[-50:]))
        pylab.plot(self.episodes_r, self.scores_r, 'C1')
        pylab.plot(self.episodes_r, self.average_r, 'b')
        pylab.ylabel('Score', fontsize=18)
        pylab.xlabel('Episodes', fontsize=18)
        dqn = 'RL_1000'

        try:
            pylab.savefig(dqn + ".png")
        except OSError:
            pass

        return str(self.average_r[-1])[:5]

    def begin_modules(self, args):
        self.verbosity = args.verbosity

        # define and register module instances
        self.module_manager = ModuleManager()
        width, height = [int(x) for x in args.carla_res.split('x')]
        # 连接到carla server
        self.world_module = ModuleWorld(MODULE_WORLD, args, timeout=10.0, module_manager=self.module_manager,
                                        width=width, height=height)
        self.traffic_module = TrafficManager(MODULE_TRAFFIC, module_manager=self.module_manager)
        self.module_manager.register_module(self.world_module)
        self.module_manager.register_module(self.traffic_module)
        if args.play_mode:
            self.hud_module = ModuleHUD(MODULE_HUD, width, height, module_manager=self.module_manager)
            self.module_manager.register_module(self.hud_module)
            self.input_module = ModuleInput(MODULE_INPUT, module_manager=self.module_manager)
            self.module_manager.register_module(self.input_module)

        # generate and save global route if it does not exist in the road_maps folder
        if self.global_route is None:
            self.global_route = np.empty((0, 3))
            distance = 1
            for i in range(1520):
                wp = self.world_module.town_map.get_waypoint(carla.Location(x=406, y=-100, z=0.1),
                                                             project_to_road=True).next(distance=distance)[0]
                distance += 2
                self.global_route = np.append(self.global_route,
                                              [[wp.transform.location.x, wp.transform.location.y,
                                                wp.transform.location.z]], axis=0)
                # To visualize point clouds
                self.world_module.points_to_draw['wp {}'.format(wp.id)] = [wp.transform.location, 'COLOR_CHAMELEON_0']
            np.save('road_maps/global_route_town04', self.global_route)

        self.motionPlanner = MotionPlanner()

        # Start Modules
        self.motionPlanner.start(self.global_route)
        self.world_module.update_global_route_csp(self.motionPlanner.csp)
        self.traffic_module.update_global_route_csp(self.motionPlanner.csp)
        self.module_manager.start_modules()  # 创建ego_car
        # self.motionPlanner.reset(self.world_module.init_s, self.world_module.init_d)

        self.ego = self.world_module.hero_actor
        self.ego_los_sensor = self.world_module.los_sensor
        self.vehicleController = VehiclePIDController(self.ego, args_lateral={'K_P': 1.5, 'K_D': 0.0, 'K_I': 0.0}) # P:1.5
        self.IDM = IntelligentDriverModel(self.ego)

        self.module_manager.tick()  # Update carla world

        self.init_transform = self.ego.get_transform()

    def enable_auto_render(self):
        self.auto_render = True

    def enable_huamn_ctrl(self):
        self.human_ctrl = True

    def render(self, mode='human'):
        self.module_manager.render(self.world_module.display)

    def view_(self):
        transform = self.ego.get_transform()
        spectator = self.world_module.world.get_spectator()
        spectator.set_transform(carla.Transform(transform.location + carla.Location(z=70),
                                                carla.Rotation(pitch=-80, yaw=transform.rotation.yaw+90)))


    def destroy(self):
        print('Destroying environment...')
        if self.world_module is not None:
            self.world_module.destroy()
            self.traffic_module.destroy()