class Config:
    def __init__(self):
        self.seed = 2
        self.hidden_units = None
        self.num_agents = 1          
        self.shared_replay_buffer = False
        self.memory = None        
        self.actor_hidden_units = (64, 64)
        self.actor_learning_rate = 1e-4
        self.critic_hidden_units = (64, 64)
        self.critic_learning_rate = 3e-4        
        self.tau = 1e-3
        self.weight_decay = 0
        self.states = None
        self.state_size = None
        self.action_size = None
        self.learning_rate = 0.001
        self.gate = None
        self.batch_size = 256
        self.buffer_size = int(1e5)
        self.gamma = 0.999
        self.update_every = 16
