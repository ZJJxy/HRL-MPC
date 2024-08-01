import os
import csv
import random
import datetime
import git
import gym
import pygame
import copy
from tqdm import tqdm

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

import argparse
from itertools import count

import gym
import scipy.optimize

import torch
from TRPO.models import *
from TRPO.replay_memory import Memory
from TRPO.running_state import ZFilter
from torch.autograd import Variable
from TRPO.trpo import trpo_step
from TRPO.utils import *


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

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')


steps_done = 0
EPS_START = 0.99
EPS_threshold = 0.5
EPS_END = 0.01
EPS_DECAY = 0.999
pastepisode = 0
def selectAction(state, model, episode: int):
    global steps_done
    global EPS_threshold
    global pastepisode
    sample = random.random()
    # 阈值不断下降
    # eps_threshold = EPS_END + (EPS_START - EPS_END) * \
    #                 math.exp(-1. * steps_done / EPS_DECAY)
    if pastepisode != episode:
        EPS_threshold = max(EPS_threshold * EPS_DECAY, EPS_END)
    pastepisode = episode

    steps_done += 1
    if sample > EPS_threshold:
        with torch.no_grad():
            mu, std = model.forward(state)
            action_dist = torch.distributions.Normal(mu, std)
            action = action_dist.sample().squeeze()
            return action
    else:
        action = torch.rand(2)  # 随机动作
        return action


class TRPOContinuous():
    def __init__(self, state_space, action_space, critic_lr, gamma, lmbda, kl_constraint, alpha):
        state_dim = state_space
        action_dim = action_space
        self.actor = PolicyLSTM(num_inputs=state_dim, num_outputs=action_dim)
        self.critic = ValueLSTM(num_inputs=state_dim)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.lmbda = lmbda
        self.kl_constraint = kl_constraint
        self.alpha = alpha

    # 广义优势估计
    def GAE_advantage(self, td_err):
        td_err = td_err.detach().numpy()
        advantage_list = []
        advantage = 0
        for delta in td_err[::-1]:
            advantage = self.gamma * self.lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(advantage_list)

    def take_action(self, state):
        # print(state)
        state = torch.tensor([state])
        # print(state.size())
        mu, std = self.actor(state)
        action_dist = torch.distributions.Normal(mu, std)
        action = action_dist.sample()
        action = action.numpy().flatten()
        return action

    def compute_surrogate_obj(self, states, actions, advantage, old_log_probs, actor):
        mu, std = actor(states)
        action_dists = torch.distributions.Normal(mu, std)
        log_probs = action_dists.log_prob(actions)
        ratio = torch.exp(log_probs - old_log_probs)
        return torch.mean(ratio * advantage)

    def conjugate_gradient(self, grad, states, old_action_dists):
        x = torch.zeros_like(grad)
        r = grad.clone()
        p = grad.clone()
        rdotr = torch.dot(r, r)
        for i in range(10):
            Hp = self.hessian_matrix_vector_product(states, old_action_dists,
                                                    p)
            alpha = rdotr / torch.dot(p, Hp)
            x += alpha * p
            r -= alpha * Hp
            new_rdotr = torch.dot(r, r)
            if new_rdotr < 1e-10:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        return x

    def hessian_matrix_vector_product(self, states, old_action_dists, vector, damping=0.1):
        mu, std = self.actor(states)
        new_action_dists = torch.distributions.Normal(mu, std)
        kl = torch.mean(torch.distributions.kl.kl_divergence(old_action_dists, new_action_dists))
        kl_grad = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
        grad2 = torch.autograd.grad(kl_grad_vector_product, self.actor.parameters())
        grad2_vector = torch.cat([grad.contiguous().view(-1) for grad in grad2])
        return grad2_vector + damping * vector

    def line_search(self, states, actions, advantage, old_log_probs,
                    old_action_dists, max_vec):
        old_para = torch.nn.utils.convert_parameters.parameters_to_vector(
            self.actor.parameters())
        old_obj = self.compute_surrogate_obj(states, actions, advantage,
                                             old_log_probs, self.actor)
        for i in range(15):
            coef = self.alpha**i
            new_para = old_para + coef * max_vec
            new_actor = copy.deepcopy(self.actor)
            torch.nn.utils.convert_parameters.vector_to_parameters(
                new_para, new_actor.parameters())
            mu, std = new_actor(states)
            new_action_dists = torch.distributions.Normal(mu, std)
            kl_div = torch.mean(
                torch.distributions.kl.kl_divergence(old_action_dists,
                                                     new_action_dists))
            new_obj = self.compute_surrogate_obj(states, actions, advantage,
                                                 old_log_probs, new_actor)
            if new_obj > old_obj and kl_div < self.kl_constraint:
                return new_para
        return old_para

    def policy_learn(self, states, actions, old_action_dists, old_log_probs, advantage):
        surrogate_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_probs, self.actor)
        grads = torch.autograd.grad(surrogate_obj, self.actor.parameters())
        obj_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        descent_direction = self.conjugate_gradient(obj_grad, states, old_action_dists)
        Hd = self.hessian_matrix_vector_product(states, old_action_dists, descent_direction)
        max_coef = torch.sqrt(2 * self.kl_constraint / (torch.dot(descent_direction, Hd) + 1e-8))
        new_para = self.line_search(states, actions, advantage, old_log_probs, old_action_dists, descent_direction * max_coef)
        torch.nn.utils.convert_parameters.vector_to_parameters(new_para, self.actor.parameters())

    def update(self, batch):
        rewards = torch.Tensor(batch.reward).view(-1, 1)
        masks = torch.Tensor(batch.mask).view(-1, 1)
        actions = torch.Tensor(np.concatenate(batch.action, 0))
        states = torch.Tensor(batch.state)
        next_states = torch.Tensor(batch.next_state)
        # rewards = (rewards + 8.0) / 8.0  # 对奖励进行修改,方便训练
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - masks)
        td_error = td_target - self.critic(states)

        advantage = self.GAE_advantage(td_error)
        mu, std = self.actor(states)
        old_action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        old_log_probs = old_action_dists.log_prob(actions)
        critic_loss = torch.mean(nn.functional.mse_loss(self.critic(states), td_target.detach()))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.policy_learn(states, actions, old_action_dists, old_log_probs,
                          advantage)


if __name__ == '__main__':
    args, cfg = parse_args_cfgs()
    print('Env is starting')
    env = gym.make(args.env)
    if args.play_mode:
        env.enable_auto_render()
    pygame.init()
    env.begin_modules(args)
    obs = env.reset()
    num_actions = env.action_space.shape[0]
    num_space = env.observation_space.shape[0]
    # print("num_act:{}".format(num_actions))
    # print("num_space:{}".format(num_space))
    # env.seed(args.seed)
    # torch.manual_seed(args.seed)

    TRAIN = True
    # modelname = "TRPO_lstmDyRtest_0_1000"
    # modelname = "TRPO_lstmT40_2000_4000"
    # modelname = "TRPO_lstmIDMPlus_1000_2000"
    # modelname = "RewardV205_TEST_0_2000"
    modelname = "MODEL_TEST_0_1000"
    agent = TRPOContinuous(action_space=num_actions, state_space=15,
                           critic_lr=1e-2, gamma=0.9, lmbda=0.9, kl_constraint=0.5, alpha=0.5) # LR DEFAULT 1e-2 KL 0.00005
    # agent.actor.load_state_dict(torch.load("./traindata/trpo-lstm-bezir-mpc/TRPO_lstmDyRtest_7000_9000_2023_09_18_23_43/policy_LSTM_last.pkl"))
    # agent.critic.load_state_dict(torch.load("./traindata/trpo-lstm-bezir-mpc/TRPO_lstmDyRtest_7000_9000_2023_09_18_23_43/value_LSTM_last.pkl"))
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

    time_now = str(datetime.datetime.now()).split(" ")
    date_str = time_now[0].split("-")
    time_str = time_now[1].split(":")
    line = "_"
    time_str_now = line.join(date_str + time_str[:2])
    fpath = f"./model/{modelname}_{time_str_now}/"
    if not os.path.exists(fpath):
        os.makedirs(fpath)

    init_episode = int(modelname.split("_")[2])
    done_episode = int(modelname.split("_")[3])
    env.episode = init_episode
    num_episodes = done_episode - init_episode
    num_step = 0
    try:
        with tqdm(total=int(num_episodes)) as pbar:
            for i_episode in range(init_episode, init_episode+num_episodes):
                memory = Memory()
                num_traj = 0
                batch_traj = 1
                # 采样batch_traj条链
                while num_traj < batch_traj:
                    obs = env.reset()
                    state = obs
                    reward_sum = 0
                    step = 0
                    done = False
                    while not done:
                        action = agent.take_action(state)
                        next_state, reward, done, _ = env.step(action)
                        reward_sum += reward

                        mask = 1
                        step += 1
                        num_step += 1
                        if done:
                            mask = 0
                        memory.push(state, np.array([action]), mask, next_state, reward)
                        state = next_state
                        with open("./data_Excel/traindata_step_r_done.csv", "a+", newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([num_step, float(reward), int(mask)])
                    pbar.set_postfix({'step': '%d' % (step), 'reward_sum': '%d' % (reward_sum)})
                    num_traj += 1
                batch = memory.sample()
                if TRAIN:
                    agent.update(batch)
                pbar.update(1)
                if i_episode%200 == 0:
                    torch.save(agent.actor.state_dict(), f'{fpath}/policy_LSTM_{i_episode}.pkl')
                    torch.save(agent.critic.state_dict(), f'{fpath}/value_LSTM_{i_episode}.pkl')
            torch.save(agent.actor.state_dict(), f'{fpath}/policy_LSTM_last.pkl')
            torch.save(agent.critic.state_dict(), f'{fpath}/value_LSTM_last.pkl')
    finally:
        env.destroy()