import random
import numpy as np
import torch
import torch.nn.functional as F
import copy
#from noise import OUNoise
import torch.optim as optim
from model import Actor, Critic
from memory_sample import ReplayBuffer

class DDPGAgent:
    def __init__(self, config):
        self.config = config
        self.seed = config.seed
        
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(config.action_size, config.state_size, config.actor_hidden_units, config.seed).to(config.device)
        self.actor_target = Actor(config.action_size, config.state_size, config.actor_hidden_units, config.seed).to(config.device)
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(), lr=config.actor_learning_rate)
        
        # Critic Network (w/ Target Network)
        self.critic_local = Critic(config.action_size, config.state_size, config.critic_hidden_units, config.seed).to(config.device)
        self.critic_target = Critic(config.action_size, config.state_size, config.critic_hidden_units, config.seed).to(config.device)

        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(), lr=config.critic_learning_rate)
        
        # ----------------------- initialize target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, 1)
        self.soft_update(self.actor_local, self.actor_target, 1)

        self.noise = OUNoise(config.action_size, config.seed)
        
        if config.shared_replay_buffer:
            self.memory = config.memory
        else:
            self.memory = ReplayBuffer(config.action_size, config.buffer_size, config.batch_size, config.seed, config.device)
        
    def reset(self):
        self.noise.reset()
        
    def act(self, states):            
        """Returns actions for given state as per current policy."""
        states = torch.from_numpy(states).float().to(self.config.device)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()
        actions += self.noise.sample()
        return np.clip(actions, -1, 1)
                
    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """

        states, actions, rewards, next_states, dones = experiences
        
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
#         torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.config.tau)
        self.soft_update(self.actor_local, self.actor_target, self.config.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)



################
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.1):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

            
