import os
import random

# import git
import gym
import pygame

gym.logger.set_level(40)
import carla_gym
import inspect
import argparse
import numpy as np
import os.path as osp
from pathlib import Path
currentPath = osp.dirname(osp.abspath(inspect.getfile(inspect.currentframe())))
# sys.path.insert(1, currentPath + '/agents/stable_baselines/')
import shutil


from stable_baselines3 import PPO, A2C
from stable_baselines3 import DQN
# from stable_baselines import PPO2
from stable_baselines3.common.evaluation import evaluate_policy

# from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
# from baselines.trpo_mpi import trpo_mpi
# from stable_baselines.common.policies import MlpPolicy as CommonMlpPolicy


from config import cfg, log_config_to_file, cfg_from_list, cfg_from_yaml_file
def parse_args_cfgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, default="tools/cfgs/config.yaml", help='specify the config for training')
    parser.add_argument('--env', help='environment ID', type=str, default='CarlaGymEnv-v2')
    parser.add_argument('--log_interval', help='Log interval (model)', type=int, default=100)
    parser.add_argument('--agent_id', type=int, default=1),
    parser.add_argument('--num_timesteps', type=float, default=1e7),
    parser.add_argument('--save_path', help='Path to save trained model to', default=None, type=str)
    parser.add_argument('--log_path', help='Directory to save learning curve data.', default=None, type=str)
    parser.add_argument('--play_mode', type=int, help='Display mode: 0:off, 1:2D, 2:3D ', default=0)
    parser.add_argument('--verbosity', help='Terminal mode: 0:Off, 1:Action,Reward 2:All', default=0, type=int)
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--test_model', help='test model file name', type=str, default='')
    parser.add_argument('--test_last', help='test model best or last?', action='store_true', default=False)
    parser.add_argument('--carla_host', metavar='H', default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
    parser.add_argument('-p', '--carla_port', metavar='P', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    parser.add_argument('--tm_port', default=8000, type=int, help='Traffic Manager TCP port to listen to (default: 8000)')
    parser.add_argument('--carla_res', metavar='WIDTHxHEIGHT', default='1280x720', help='window resolution (default: 1280x720)')


    args = parser.parse_args()

    args.num_timesteps = int(args.num_timesteps)

    if args.test and args.cfg_file is None:
        path = 'logs/agent_{}/'.format(args.agent_id)
        conf_list = [cfg_file for cfg_file in os.listdir(path) if '.yaml' in cfg_file]
        args.cfg_file = path + conf_list[0]

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    # visualize all test scenarios
    if args.test:
        args.play_mode = True

    return args, cfg



TRAIN_A2C = True
TRAIN_TRPO = False

TRAIN = True

# import stat
# os.chmod("D:/Python_PRO/Frenet_Traj_MPC/data_Excel/traindata_step_tarV_actV_tx_ty_ax_ay.csv", stat.S_IRWXU)

if __name__ == '__main__':
    args, cfg = parse_args_cfgs()
    print('Env is starting')

    if os.path.exists("./data_Excel/testdata_step_Jerk_wz.csv"):
        os.remove("./data_Excel/testdata_step_Jerk_wz.csv")
    if os.path.exists("./data_Excel/traindata_step_velocity.csv"):
        os.remove("./data_Excel/traindata_step_velocity.csv")
    if os.path.exists("./data_Excel/eps_epsR.csv"):
        os.remove("./data_Excel/eps_epsR.csv")
    if os.path.exists("./data_Excel/traindata_step_r_done.csv"):
        os.remove("./data_Excel/traindata_step_r_done.csv")
    if os.path.exists("./data_Excel/traindata_step_velocity_acc_psi.csv"):
        os.remove("./data_Excel/traindata_step_velocity_acc_psi.csv")
    if os.path.exists("./data_Excel/idm_expv_v_ax_ay_wz_comfort_col.csv"):
        os.remove("./data_Excel/idm_expv_v_ax_ay_wz_comfort_col.csv")
    if os.path.exists("./data_Excel/traindata_step_tarV_actV_tx_ty_ax_ay.csv"):
        os.remove("./data_Excel/traindata_step_tarV_actV_tx_ty_ax_ay.csv")

    env = gym.make(args.env)
    if args.play_mode:
        env.enable_auto_render()
    pygame.init()
    env.begin_modules(args)
    obs = env.reset()
    try:
        # print("train A2C")
        # # model = A2C("MlpPolicy", env, verbose=0)
        # model = A2C.load(f"./model/A2C/{3000*13}.zip", env=env)
        # for i in range(14, 17):
        #     # print(model.policy)
        #     # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        #     # print(f"Before training: mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
        #     model.learn(total_timesteps=3000, reset_num_timesteps=False)
        #     # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        #     # print(f"After training: mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
        #     model.save(f"./model/A2C/{3000*i}")
        #     print(i)
        if TRAIN:
            print("train PPO")
            model = PPO("MlpPolicy", env, verbose=0)
            model = A2C.load(f"./model/PPO/{3000*4}.zip", env=env)
            for i in range(5, 8):
                # print(model.policy)
                # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
                # print(f"Before training: mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
                model.learn(total_timesteps=500, reset_num_timesteps=False)
                # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
                # print(f"After training: mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
                model.save(f"./model/PPO/{3000*i}")
                print(i)
        else:
            print("eval PPO2")
            # model = PPO("MlpPolicy", env, verbose=0)
            model = PPO.load(f"./traindata/PPO_Dytest/{3000*5}.zip", env=env)
            model.learn(total_timesteps=500, reset_num_timesteps=False)
            # print("eval A2C")
            # # model = PPO("MlpPolicy", env, verbose=0)
            # model = PPO.load(f"./traindata/A2C_Dytest/{48000}.zip", env=env)
            # model.learn(total_timesteps=500, reset_num_timesteps=False)

    finally:
        env.destroy()