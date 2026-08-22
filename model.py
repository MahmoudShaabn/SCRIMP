import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from alg_parameters import *
from net import SCRIMPNet


class Model(object):
    """model wrapper for SCRIMPNet"""

    def __init__(self, env_id, device, is_global=False):
        """initialization"""
        self.ID = env_id
        self.device = device
        self.network = SCRIMPNet().to(device)
        self.is_global = is_global
        if is_global:
            self.net_optimizer = torch.optim.Adam(self.network.parameters(), lr=TrainingParameters.LR)
            self.scaler = GradScaler()

    def step(self, obs, vector, valid_actions, input_state, no_reward, message, num_agent):
        """sample actions and compute values for rollout worker"""
        self.network.eval()
        with torch.no_grad():
            obs = torch.from_numpy(obs).to(self.device)
            vector = torch.from_numpy(vector).to(self.device)
            valid_actions = torch.from_numpy(valid_actions).to(self.device)

            policy, value_in, value_ex, blocking, policy_sig, output_state, _, raw_message, c_i = \
                self.network(obs, vector, input_state, message)

            # Mask invalid actions
            valid_policy = policy * valid_actions
            valid_policy_sum = torch.sum(valid_policy, dim=-1, keepdim=True)
            valid_policy = valid_policy / (valid_policy_sum + 1e-8)

            dist = torch.distributions.Categorical(valid_policy)
            actions = dist.sample()
            
            num_invalid = (valid_policy_sum == 0).sum().item()

            values_in = value_in.cpu().numpy()
            values_ex = value_ex.cpu().numpy()
            values_all = values_in + values_ex

            return actions.cpu().numpy(), policy.cpu().numpy(), values_in, values_ex, values_all, \
                blocking.cpu().numpy(), output_state, num_invalid, raw_message, c_i

    def evaluate(self, obs, vector, valid_actions, input_state, greedy, no_reward, message, num_agent):
        """sample actions during evaluation"""
        self.network.eval()
        with torch.no_grad():
            obs = torch.from_numpy(obs).to(self.device)
            vector = torch.from_numpy(vector).to(self.device)
            valid_actions = torch.from_numpy(valid_actions).to(self.device)

            policy, value_in, value_ex, blocking, policy_sig, output_state, _, raw_message, c_i = \
                self.network(obs, vector, input_state, message)

            valid_policy = policy * valid_actions
            valid_policy_sum = torch.sum(valid_policy, dim=-1, keepdim=True)
            valid_policy = valid_policy / (valid_policy_sum + 1e-8)

            if greedy:
                actions = torch.argmax(valid_policy, dim=-1).cpu().numpy().squeeze(0)
            else:
                dist = torch.distributions.Categorical(valid_policy)
                actions = dist.sample().cpu().numpy().squeeze(0)

            num_invalid = (valid_policy_sum == 0).sum().item()
            values_all = (value_in + value_ex).cpu().numpy()

            return actions, blocking.cpu().numpy(), output_state, num_invalid, values_all, \
                policy.cpu().numpy(), raw_message, c_i

    def value(self, obs, vector, input_state, no_reward, message):
        """compute critic values for GAE advantage computation"""
        self.network.eval()
        with torch.no_grad():
            obs = torch.from_numpy(obs).to(self.device)
            vector = torch.from_numpy(vector).to(self.device)

            _, value_in, value_ex, _, _, _, _, _, _ = self.network(obs, vector, input_state, message)

            values_in = value_in.cpu().numpy()
            values_ex = value_ex.cpu().numpy()
            values_all = values_in + values_ex

            return values_in, values_ex, values_all

    def generate_state(self, obs, vector, input_state, message):
        """generate hidden state for imitation learning"""
        self.network.eval()
        with torch.no_grad():
            obs = torch.from_numpy(obs).to(self.device)
            vector = torch.from_numpy(vector).to(self.device)

            _, _, _, _, _, output_state, _, message_out, _ = self.network(obs, vector, input_state, message)
            return output_state, message_out

    def train(self, mb_obs, mb_vector, mb_returns_in, mb_returns_ex, mb_returns_all, mb_values_in,
              mb_values_ex, mb_values_all, mb_actions, mb_ps, mb_hidden_state,
              mb_train_valid, mb_blocking, mb_message, mb_c_i):
        """PPO optimization step with communication loss penalty"""
        self.network.train()

        obs = torch.from_numpy(mb_obs).to(self.device)
        vector = torch.from_numpy(mb_vector).to(self.device)
        returns_in = torch.from_numpy(mb_returns_in).to(self.device).float()
        returns_ex = torch.from_numpy(mb_returns_ex).to(self.device).float()
        returns_all = torch.from_numpy(mb_returns_all).to(self.device).float()
        old_values_all = torch.from_numpy(mb_values_all).to(self.device).float()
        actions = torch.from_numpy(mb_actions).to(self.device).long()
        old_ps = torch.from_numpy(mb_ps).to(self.device).float()
        input_state = (torch.from_numpy(mb_hidden_state[:, 0]).to(self.device),
                       torch.from_numpy(mb_hidden_state[:, 1]).to(self.device))
        train_valid = torch.from_numpy(mb_train_valid).to(self.device).float()
        blocking_target = torch.from_numpy(mb_blocking).to(self.device).float()
        message = torch.from_numpy(mb_message).to(self.device)

        with autocast():
            policy, value_in, value_ex, blocking, _, _, policy_logits, _, c_i = \
                self.network(obs, vector, input_state, message)

            # PPO Policy Loss
            advantages = returns_all - old_values_all
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            valid_policy = policy * train_valid
            valid_policy_sum = torch.sum(valid_policy, dim=-1, keepdim=True)
            valid_policy = valid_policy / (valid_policy_sum + 1e-8)

            action_dist = torch.distributions.Categorical(valid_policy)
            new_log_probs = action_dist.log_prob(actions)
            old_log_probs = torch.log(torch.gather(old_ps, -1, actions.unsqueeze(-1)).squeeze(-1) + 1e-8)

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - TrainingParameters.CLIP_RANGE,
                                1.0 + TrainingParameters.CLIP_RANGE) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Critic Value Losses
            value_in = value_in.squeeze(-1)
            value_ex = value_ex.squeeze(-1)
            value_loss_in = F.mse_loss(value_in, returns_in)
            value_loss_ex = F.mse_loss(value_ex, returns_ex)
            value_loss = value_loss_in + value_loss_ex

            # Blocking and Entropy
            blocking_loss = F.binary_cross_entropy(blocking.squeeze(-1), blocking_target)
            entropy = action_dist.entropy().mean()

            # Communication Regularization Loss
            lambda_comm = getattr(TrainingParameters, 'LAMBDA_COMM', 0.05)
            comm_loss = lambda_comm * torch.mean(c_i)

            total_loss = (policy_loss +
                          TrainingParameters.VALUE_COEF * value_loss +
                          TrainingParameters.BLOCKING_COEF * blocking_loss -
                          TrainingParameters.ENTROPY_COEF * entropy +
                          comm_loss)

        self.net_optimizer.zero_grad()
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.net_optimizer)
        nn.utils.clip_grad_norm_(self.network.parameters(), TrainingParameters.MAX_GRAD_NORM)
        self.scaler.step(self.net_optimizer)
        self.scaler.update()

        return [total_loss.item(), policy_loss.item(), value_loss.item(), comm_loss.item()]

    def imitation_train(self, mb_obs, mb_vector, mb_actions, mb_hidden_state, mb_message):
        """imitation learning optimization step"""
        self.network.train()

        obs = torch.from_numpy(mb_obs).to(self.device)
        vector = torch.from_numpy(mb_vector).to(self.device)
        actions = torch.from_numpy(mb_actions).to(self.device).long()
        input_state = (torch.from_numpy(mb_hidden_state[:, 0]).to(self.device),
                       torch.from_numpy(mb_hidden_state[:, 1]).to(self.device))
        message = torch.from_numpy(mb_message).to(self.device)

        with autocast():
            policy, _, _, _, _, _, _, _, _ = self.network(obs, vector, input_state, message)
            imitation_loss = F.cross_entropy(policy.reshape(-1, EnvParameters.N_ACTIONS), actions.reshape(-1))

        self.net_optimizer.zero_grad()
        self.scaler.scale(imitation_loss).backward()
        self.scaler.unscale_(self.net_optimizer)
        nn.utils.clip_grad_norm_(self.network.parameters(), TrainingParameters.MAX_GRAD_NORM)
        self.scaler.step(self.net_optimizer)
        self.scaler.update()

        return imitation_loss.item()

    def set_weights(self, weights):
        """load network weights"""
        self.network.load_state_dict(weights)