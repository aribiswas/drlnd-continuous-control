# Report

## The agent

A Deep Deterministic Policy Gradient (DDPG) agent is used to train the Reacher environment. DDPG agents can operate on continuous action spaces which is a critical requirement for this environment. 

During the training process, the agent interacts with the environment and stores experiences in an offline experience buffer. Learning from experiences is through a stochastic gradient descent process. The agent uses an actor-critic model, where the critic approximates the state-action value function Q(s,a) and the actor models a deterministic policy that maximizes the expected value of Q(s,a) as the training progresses.

Both the actor and critic are modeled by neural networks as explained in the following sections.

### Actor

The actor in a DDPG agent is a deterministic actor. It is modeled by a neural network that takes states as input and outputs the actions. In this implementation, the actor neural network has 2 fully connected layers with sizes 64 and 32. The output of the actor is bounded between -1 and 1 through a tanh layer. This network structure is simple and you will alter find that this is sufficient to train the agent.

<pre><code>
class DeterministicActor(nn.Module):
    
    def __init__(self, num_obs, num_act, seed=0):
        
        torch.manual_seed(seed)

        super(DeterministicActor, self).__init__()

        self.num_obs = num_obs

        # layers
        self.fc1 = nn.Linear(num_obs,64)
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32,num_act)
        self.tanh = nn.Tanh()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        
    def forward(self, state):
        """
        Perform forward pass through the network.
        
        """
        
        # convert to torch
        if isinstance(state, numpy.ndarray):
            x = torch.from_numpy(state).float().to(self.device)
        elif isinstance(state, torch.Tensor):
            x = state
        else:
            raise TypeError("Input must be a numpy array or torch Tensor.")
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x
        
        
    def mu(self, state):
        """
        Get the deterministic policy.

        """
        
        return torch.clamp(self.forward(state), -1, 1)
        
        
</code></pre>


During training, the agent explores the action space by introducing random exploration noise in its actions. This is performed by a noise model known as **Ornstein-Uhlenbeck action noise**, or more commonly, OU noise. The noise is parameterized by its mean, mean attraction constant and variance. The variance can be decayed to facilitate high exploration towards the beginning of training and exploitation later on.

<pre><code>
class OUNoise:
    
    def __init__(self, size, mean=0, mac=0.15, var=0.1, varmin=0.01, decay=1e-6, seed=0):
        """
        Initialize Ornstein-Uhlenbech action noise.

        Parameters
        ----------
        size : list or numpy array
            Dimensions of the noise [a,b] where a is the batch size and b is the number of actions
        mean : number, optional
            Mean of the OU noise. The default is 0.
        mac : number, optional
            Mena attraction constant. The default is 0.15.
        var : number, optional
            Variance. The default is 0.1.
        varmin : TYPE, optional
            Minimum variance. The default is 0.01.
        decay : number, optional
            Decay rate of variance. The default is 1e-6.
        seed : number, optional
            Seed. The default is 0.

        """
        np.random.seed(seed)
        self.mean = mean * np.ones(size)
        self.mac = mac
        self.var = var
        self.varmin = varmin
        self.decay = decay
        self.x = np.zeros(size) #0.25 * np.random.rand(20,4)
        self.xprev = self.x
        self.step_count = 0
        
    def step(self):
        """
        Step the OU noise model by computing the noise and decaying variance.

        Returns
        -------
        noise : numpy array
            OU action noise.

        """
        r = self.x.shape[0]
        c = self.x.shape[1]
        self.x = self.xprev + self.mac * (self.mean - self.xprev) + self.var * np.random.randn(r,c)
        self.xprev = self.x
        dvar = self.var * (1-self.decay)
        self.var = np.maximum(dvar, self.varmin)
        self.step_count += 1
        return self.x

</code></pre>