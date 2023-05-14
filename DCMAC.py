import numpy as np
import random
import gym
import torch
import seaborn as sns
from torch.nn import init
from tqdm.auto import tqdm
from dataclasses import dataclass
import torch.nn.utils as utils
from torch import nn
import pandas as pd
from torch import optim
from typing import Any
from copy import deepcopy
import torch.nn.functional as F
from collections import deque

from citylearn.citylearn import CityLearnEnv
from envs.CityLearnModEnv import CityLearnModEnv


class Constants:
    schema_path = 'data/citylearn_challenge_2022_phase_1/schema.json'
    eval_schema_path = 'data/citylearn_challenge_2022_phase_2/schema.json'


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.BUFFER_SIZE = buffer_size
        self.BATCH_SIZE = batch_size
        self.replay_buffer = deque(maxlen=self.BUFFER_SIZE)
        self.actions_cache = None

    def __len__(self):
        return len(self.replay_buffer)

    def __getitem__(self, item):
        return self.replay_buffer[item]

    def add_experience(self, state, action, action_mean, action_std, reward, next_state):
        experience = {"state": state, "action": action.cpu(), "action_mean": action_mean.cpu(),
                      "action_std": action_std.cpu(), "reward": reward, "next_state": next_state}
        if self.actions_cache is None:
            self.actions_cache = action.unsqueeze(0).cpu()
        else:
            self.actions_cache = torch.cat((self.actions_cache, action.unsqueeze(0).cpu()), dim=0)

        if len(self.replay_buffer) == self.BUFFER_SIZE:
            self.actions_cache = self.actions_cache[1:]

        self.replay_buffer.append(experience)

    def get_all_actions(self):
        return self.actions_cache

    def sample_experiences(self):
        if len(self.replay_buffer) < self.BATCH_SIZE:
            raise Exception("BUFFER SIZE SMALLER THEN BATCH SIZE")

        experiences = random.sample(self.replay_buffer, k=self.BATCH_SIZE)

        states = torch.tensor(np.stack([state['state'] for state in experiences]).astype(np.float32))
        actions = torch.stack([action['action'].squeeze(0) for action in experiences])
        action_means = torch.stack([action['action_mean'].squeeze(0) for action in experiences])
        action_stds = torch.stack([action['action_std'].squeeze(0) for action in experiences])

        try:
            rewards = torch.tensor(np.array([reward['reward'] for reward in experiences]).astype(np.float32)).resize(
                self.BATCH_SIZE, 1)
        except:
            print([reward['reward'] for reward in experiences])

        next_states = torch.tensor(np.stack([state['next_state'] for state in experiences]).astype(np.float32))

        return states, actions, action_means, action_stds, rewards, next_states

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, num_agents, max_std, min_std):
        super().__init__()
        self.max_std = max_std
        self.min_std = min_std


        self.head_layer = nn.Sequential(
            nn.Linear(state_dim, 120),
            nn.ReLU(),
            nn.Linear(120, 120),
            nn.ReLU(),
        )

        self.mean_layer = nn.Sequential(
            nn.Linear(120, action_dim * num_agents),
        )

        self.log_std_network = nn.Sequential(
            nn.Linear(120, action_dim * num_agents),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

        self.action_dim = action_dim
        self.num_agents = num_agents

    def forward(self, ob):
        single_input = False
        if ob.size(0) == 1:
            ob = torch.cat([ob, ob])
            single_input = True

        head = self.head_layer(ob)
        mean = self.mean_layer(head)
        mean = torch.clamp(mean, -3, 3)
        mean = torch.tanh(mean.view(mean.size(0), -1, self.action_dim))


        std = self.log_std_network(head)
        std = torch.clamp(std, max=np.log(self.max_std), min=np.log(self.min_std))
        std = std.exp().unsqueeze(2).expand_as(mean)

        if single_input:
            mean = mean[0].unsqueeze(0)
            std = std[0].unsqueeze(0)

        return mean, std

# Define the critic network architecture
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, num_agents):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_dim + (num_agents * action_dim), 120),
            nn.ReLU(),
            nn.Linear(120, 120),
            nn.ReLU(),
            nn.Linear(120, 1)
        )

    def forward(self, state, action):
        state_action = torch.cat((state, action), dim=-1)
        x = self.critic(state_action)
        return x


@dataclass
class DCMACAgent:
    components: int
    state_dim: int
    action_dim: int
    max_epsilon_decay_steps: int
    max_noise_decay_steps: int

    batch_size: int = 32
    memory_size: int = 20000
    warm_up_steps: int = 500
    gamma: float = 0.99
    actor_lr: float = 0.00001
    critic_lr: float = 0.00002
    epsilon_init: float = 1.0
    min_epsilon: float = 0.0
    tau = 0.005

    max_actor_std: float = 0.1
    min_actor_std: float = 0.001

    noise_sigma_init: float = 1
    noise_sigma_min: float = 0.1
    device: str = None
    importance_weight_clip: float = 2

    def __post_init__(self):
        self.reset()

    def reset(self):
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.num_agents = self.components
        self.centralized_actor_network = Actor(self.state_dim, self.action_dim, self.num_agents, self.max_actor_std,
                                               self.min_actor_std)
        self.centralized_critic_network = Critic(self.state_dim, self.action_dim, self.num_agents)
        self.centralized_critic_network_target = Critic(self.state_dim, self.action_dim, self.num_agents)

        self.centralized_actor_network.to(self.device)
        self.centralized_critic_network.to(self.device)
        self.centralized_critic_network_target.to(self.device)

        self.actor_optim = optim.Adam(self.centralized_actor_network.parameters(), lr=self.actor_lr)
        self.critic_optim = optim.Adam(self.centralized_critic_network.parameters(), lr=self.critic_lr)

        self.replay_buffer = ReplayBuffer(self.memory_size, self.batch_size)
        self.counter = 0

        self.epsilon = self.epsilon_init
        self.ep_reduction = (self.epsilon - self.min_epsilon) / float(self.max_epsilon_decay_steps)

        self.noise_sigma = self.noise_sigma_init
        self.noise_reduction = (self.noise_sigma - self.noise_sigma_min) / float(self.max_noise_decay_steps)

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space

    def store_experience(self, obs, actions, actions_mean, actions_std, rewards, next_obs):
        self.replay_buffer.add_experience(obs, actions, actions_mean, actions_std, rewards, next_obs)

    def compute_action(self, observation, agent_id, rand=False, evaluating=False):
        with torch.no_grad():
            """Get observation return action"""
            observation = torch.tensor(np.array(observation).astype(np.float32))
            observation = observation.to(self.device)

            if not evaluating and (len(self.replay_buffer) < self.warm_up_steps):
                # generate random sample data
                action_means = 2 * torch.rand(self.num_agents, self.action_dim) - 1
                action_stds = torch.rand(self.num_agents, self.action_dim) * 0.1
                dist = torch.distributions.normal.Normal(action_means, action_stds)
                actions = dist.sample()
            elif evaluating:
                if len(observation.size()) < 2:
                    observation = observation.resize(1, self.state_dim)
                action_means, action_stds = self.centralized_actor_network(observation)
                actions = action_means
            else:
                if len(observation.size()) < 2:
                    observation = observation.resize(1, self.state_dim)

                action_means, action_stds = self.centralized_actor_network(observation)
                dist = torch.distributions.normal.Normal(action_means, action_stds)

                if np.random.uniform() < self.epsilon or rand:
                    noise_dist = torch.distributions.normal.Normal(action_means, self.noise_sigma)
                    noise = noise_dist.sample()
                    dist = torch.distributions.normal.Normal(action_means + noise, action_stds)

                actions = dist.sample()

            actions = torch.clamp(actions,-3,3)
            actions = torch.tanh(actions)

            return actions, action_means, action_stds


    def update_actor_critic(self):
        self.counter += 1
        actor_loss = 0
        critic_loss = 0

        if len(self.replay_buffer) < self.warm_up_steps:
            return actor_loss, critic_loss

        obs, actions, action_means, action_stds, rewards, next_obs = self.replay_buffer.sample_experiences()

        obs = obs.to(self.device)
        actions = actions.to(self.device)
        action_means = action_means.to(self.device)
        action_stds = action_stds.to(self.device)
        rewards = rewards.to(self.device)
        next_obs = next_obs.to(self.device)

        # Predict action given state
        pred_action_means, pred_action_stds = self.centralized_actor_network(obs)
        dist = torch.distributions.normal.Normal(pred_action_means, pred_action_stds)
        p = torch.clamp(dist.rsample(),-3, 3)
        pred_actions = torch.tanh(p)

        log_probs = dist.log_prob(pred_actions).sum(dim=1)

        # Calculate weights for prioritized experience replay
        with torch.no_grad():
            pi_dist = torch.distributions.normal.Normal(pred_action_means, pred_action_stds)
            pi_probs = pi_dist.log_prob(pred_actions).sum(dim=1).exp()
            mu_dist = torch.distributions.normal.Normal(action_means, action_stds)
            mu_probs = mu_dist.log_prob(pred_actions).sum(dim=1).exp()

            weights = pi_probs / mu_probs
            weights = torch.clamp(weights, max=self.importance_weight_clip)

        # update critic network
        with torch.no_grad():
            next_pred_action_means, next_pred_action_stds = self.centralized_actor_network(next_obs)
            next_dist = torch.distributions.normal.Normal(next_pred_action_means, next_pred_action_stds)
            z = torch.clamp(next_dist.sample(),-3,3)
            next_actions = torch.tanh(z)
            td_target = rewards + self.gamma * self.centralized_critic_network_target(next_obs,
                                                                                      next_actions.squeeze(-1))

            # td_target = rewards + self.gamma * self.centralized_critic_network(next_obs,
            #                                                                    next_actions.squeeze(-1))
        td_error = td_target - self.centralized_critic_network(obs, actions.squeeze(-1))
        advantages = td_error.detach()

        self.critic_optim.zero_grad()

        critic_loss = (td_error ** 2) * weights
        critic_loss = critic_loss.mean()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm(self.centralized_critic_network.parameters(), max_norm=1)
        self.critic_optim.step()

        # update actor network
        self.actor_optim.zero_grad()
        actor_loss = (log_probs * advantages * weights).mean()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.centralized_actor_network.parameters(), max_norm=1)
        self.actor_optim.step()

        return actor_loss.detach().cpu(), critic_loss.detach().cpu()

    def decay_epsilon(self):
        self.epsilon = self.epsilon - self.ep_reduction if self.epsilon - self.ep_reduction >= self.min_epsilon else \
            self.min_epsilon

    def decay_noise(self):
        self.noise_sigma = self.noise_sigma - self.noise_reduction if self.noise_sigma - self.noise_reduction >= \
                                                                      self.noise_sigma_min else self.noise_sigma_min

    def update_target_critic(self, step, soft=True):
        if not soft:
            if step % self.target_update_freq == 0:
                self.centralized_critic_network_target = deepcopy(self.centralized_critic_network)

        else:
            for target_param, param in zip(self.centralized_critic_network_target.parameters(),
                                           self.centralized_critic_network.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)


def test(env, agent: DCMACAgent):
    t = []

    # we reset the environment
    nb_iter = 8759
    ob = env.reset()
    test_actions = []
    for n in tqdm(range(nb_iter), desc="testing"):
        actions, action_means, action_stds = agent.compute_action(ob, None, evaluating=True)
        test_actions.append(actions.squeeze(-1))
        next_obs, rewards, dones, info = env.step(actions.squeeze(0).cpu())
        ob = next_obs

    return env.env.evaluate(), torch.stack(test_actions)


def train(agent: DCMACAgent, episodes, max_steps, env, test_env, show_progress=False, progress_check=5,
          model_name="default"):
    num_ep = 0
    ep_logs = []
    agent.reset()

    for i in tqdm(range(episodes), desc="ep"):
        ep_rewards = []
        ep_steps = []
        cumulative_reward = []
        train_actions = []

        smooth_ep_return = deque(maxlen=10)
        smooth_actor_losses = deque(maxlen=10)
        smooth_critic_losses = deque(maxlen=10)

        total_actor_loss = []
        total_critic_loss = []

        ob = env.reset()

        num_ep += 1
        ret = 0
        actor_loss = 0
        critic_loss = 0

        for t in tqdm(range(max_steps), desc="step"):

            if len(agent.replay_buffer) < agent.warm_up_steps:
                actions, action_means, action_stds = agent.compute_action(None, None, rand=True)
            else:
                actions, action_means, action_stds = agent.compute_action(ob, None)

            train_actions.append(actions)

            next_obs, reward, done, info = env.step(actions.squeeze(0).cpu())

            agent.store_experience(ob, actions.squeeze(0).cpu(), action_means, action_stds, reward, next_obs)

            al, cl = agent.update_actor_critic()

            ob = next_obs

            agent.decay_epsilon()
            agent.decay_noise()
            agent.update_target_critic(t, soft=True)

            # Record rewards
            ret += float(reward)

            cumulative_reward.append(ret)
            smooth_ep_return.append(ret / (t + 1))

            ep_rewards.append(np.mean(smooth_ep_return))
            ep_steps.append(t)

            # Record losses
            actor_loss += al
            critic_loss += cl

            smooth_critic_losses.append(cl)
            smooth_actor_losses.append(al)

            total_actor_loss.append(np.mean(smooth_actor_losses))
            total_critic_loss.append(np.mean(smooth_critic_losses))

            if show_progress and False:
                print(f'Step:{t}  epsilon:{agent.epsilon}  '
                      f'Smoothed Training Return:{np.mean(smooth_ep_return)}')

        if num_ep % progress_check == 0:
            test_ret = test(test_env, agent)
            if show_progress:
                print('==========================')
                print(f'Step:{t} Testing Return: {sum(test_ret[0]) / 2}')

        run_log = pd.DataFrame({'return': ep_rewards,
                                'total_return': cumulative_reward,
                                'steps': ep_steps,
                                'episode': np.full((len(ep_steps)), num_ep),
                                'epsilon': agent.epsilon_init,
                                'actor_loss': total_actor_loss,
                                'critic_loss': total_critic_loss})
        ep_logs.append(run_log)

    ep_logs = pd.concat(ep_logs, ignore_index=True)
    ep_logs.to_csv('DCMAC_performance.csv', index=False)
    torch.save(agent.centralized_actor_network.state_dict(), "actor_model_" + model_name)
    torch.save(agent.centralized_critic_network.state_dict(), "critic_model_" + model_name)
    ep_logs['Agent'] = "DCMAC"

    return ep_logs, test(test_env, agent)
