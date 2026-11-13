"""On-policy, discrete-action PPO actor-critic implemented with PyTorch."""
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
        )
        self.policy_head = nn.Linear(128, n_actions)
        self.value_head = nn.Linear(128, 1)

    def forward(self, obs):
        hidden = self.body(obs)
        return self.policy_head(hidden), self.value_head(hidden).squeeze(-1)


class PPOAgent:
    def __init__(self, obs_dim, actions=("left", "forward", "right"),
                 lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, entropy_coef=0.02, value_coef=0.5,
                 update_epochs=4, minibatch_size=64, device="cpu"):
        self.actions = tuple(actions)
        self.device = torch.device(device)
        self.model = ActorCritic(obs_dim, len(self.actions)).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size

    @torch.no_grad()
    def choose_action(self, observation, deterministic=False):
        obs = torch.as_tensor(observation, dtype=torch.float32,
                              device=self.device).unsqueeze(0)
        logits, value = self.model(obs)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = Categorical(logits=logits).sample()
        log_prob = Categorical(logits=logits).log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item())

    def update(self, buffer, bootstrap_value=0.0):
        """PPO update with GAE; done includes terminal and time-limit endpoints."""
        n = len(buffer)
        obs = np.stack([step[0] for step in buffer])
        actions = np.asarray([step[1] for step in buffer], dtype=np.int64)
        log_probs = np.asarray([step[2] for step in buffer], dtype=np.float32)
        values = np.asarray([step[3] for step in buffer], dtype=np.float32)
        rewards = np.asarray([step[4] for step in buffer], dtype=np.float32)
        dones = np.asarray([step[5] for step in buffer], dtype=bool)
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0
        next_value = float(bootstrap_value)

        for t in range(n - 1, -1, -1):
            nonterminal = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma*next_value*nonterminal - values[t]
            gae = delta + self.gamma*self.gae_lambda*nonterminal*gae
            advantages[t] = gae
            next_value = float(values[t])
        returns = advantages + values

        obs_t = torch.from_numpy(obs).to(self.device)
        act_t = torch.from_numpy(actions).to(self.device)
        oldlog_t = torch.from_numpy(log_probs).to(self.device)
        ret_t = torch.from_numpy(returns).to(self.device)
        adv_t = torch.from_numpy(advantages).to(self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std(unbiased=False) + 1e-8)

        for _ in range(self.update_epochs):
            for indices in torch.randperm(n, device=self.device).split(self.minibatch_size):
                logits, val = self.model(obs_t[indices])
                dist = Categorical(logits=logits)
                newlog = dist.log_prob(act_t[indices])
                ratio = torch.exp(newlog - oldlog_t[indices])
                unclipped = ratio * adv_t[indices]
                clipped = ratio.clamp(1-self.clip_epsilon, 1+self.clip_epsilon) * adv_t[indices]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = (val-ret_t[indices]).square().mean()
                entropy = dist.entropy().mean()
                loss = (policy_loss + self.value_coef*value_loss
                        - self.entropy_coef*entropy)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
        return float(advantages.mean()), float(returns.mean())

    def save(self, path):
        torch.save({"model_state_dict": self.model.state_dict(),
                    "actions": self.actions}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if tuple(checkpoint["actions"]) != self.actions:
            raise ValueError("Saved action list does not match the current agent")
        self.model.load_state_dict(checkpoint["model_state_dict"])
