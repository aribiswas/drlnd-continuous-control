import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class StochasticActor(nn.Module):

    def __init__(self, num_obs, num_act, seed):
        
        torch.manual_seed(seed)

        super(StochasticActor, self).__init__()

        self.num_obs = num_obs
        self.num_act = num_act

        # layers
        self.fc1 = nn.Linear(num_obs, 256)
        self.fc2 = nn.Linear(256, 128)
        self.meanfc1 = nn.Linear(128, 64)
        self.meanfc2 = nn.Linear(64, num_act)
        #self.stdfc1 = nn.Linear(128, 64)
        #self.stdfc2 = nn.Linear(64, num_act)
        self.tanh = nn.Tanh()
        #self.logsoftmax = nn.LogSoftmax()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        meanx = F.relu(self.meanfc1(x))
        meanx = self.tanh(self.meanfc2(meanx))
        
        #stdx = F.relu(self.stdfc1(x))
        #stdx = self.logsoftmax(self.stdfc2(stdx))
        stdx = 0.5 * torch.ones(self.num_act).float().to(self.device)

        return meanx, torch.exp(stdx)
    
    
    def pi(self, state, action=None):
        
        # convert to torch
        if isinstance(state, numpy.ndarray):
            x = torch.from_numpy(state).float().to(self.device)
        elif isinstance(state, torch.Tensor):
            x = state
        else:
            raise TypeError("State input must be a numpy array or torch Tensor.")
            
        # obtain mean and std from network
        mean, std = self.forward(x)

        # create Normal distribution
        dist = Normal(mean, std)
        
        # get the entropy
        entropy = dist.entropy().mean()
        
        # optional: compute log probabilities
        if action is None:
            logp = None
        else:
            # convert to torch
            if isinstance(action, numpy.ndarray):
                act = torch.from_numpy(action).float().to(self.device)
            elif isinstance(state, torch.Tensor):
                act = action
            else:
                raise TypeError("Action input must be a numpy array or torch Tensor.")
            logp = dist.log_prob(act)
        
        return dist, logp, entropy


    def get_action(self, state):
        
        with torch.no_grad():
            
            # get the policy
            dist, _, _ = self.pi(state)
    
            # sample action
            action = dist.sample()
            action = torch.clamp(action, -1, 1)  # limit actions to [-1,1]

        return action.numpy()
    

class Critic(nn.Module):

    def __init__(self, num_obs, seed):
        
        torch.manual_seed(seed)

        super(Critic, self).__init__()

        self.num_obs = num_obs

        # layers
        self.fc1 = nn.Linear(num_obs, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        

    def forward(self, state):
        
        # convert to torch
        if isinstance(state, numpy.ndarray):
            x = torch.from_numpy(state).float().to(self.device)
        elif isinstance(state, torch.Tensor):
            x = state
        else:
            raise TypeError("Input must be a numpy array or torch Tensor.")

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)   

        return x


    def get_value(self, state):
        
        return self.forward(state)

