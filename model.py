import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class StochasticActor(nn.Module):

    def __init__(self, num_obs, num_act, seed=0):
        
        torch.manual_seed(seed)

        super(StochasticActor, self).__init__()

        self.num_obs = num_obs
        self.num_act = num_act

        # layers 
        self.fc1 = nn.Linear(num_obs, 256)
        self.fc2 = nn.Linear(256, 128)
        self.bn1 = nn.BatchNorm1d(256)
        self.meanfc1 = nn.Linear(128, 64)
        self.meanfc2 = nn.Linear(64, num_act)
        self.tanh = nn.Tanh()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        meanx = F.relu(self.meanfc1(x))
        meanx = self.tanh(self.meanfc2(meanx))
        
        # constant logstdx
        # TODO: output varying logstdx from network
        logstdx = 0.5 * torch.ones(self.num_act).float().to(self.device)

        return meanx, logstdx
    
    
    def pi(self, state, action=None):
        
        # convert to torch
        if isinstance(state, numpy.ndarray):
            x = torch.from_numpy(state).float().to(self.device)
        elif isinstance(state, torch.Tensor):
            x = state
        else:
            raise TypeError("State input must be a numpy array or torch Tensor.")
            
        # obtain mean and std from network
        mean, logstd = self.forward(x)

        # create Normal distribution
        dist = Normal(mean, torch.exp(logstd))
        
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
            
            # take the log prob of all actions and sum them
            logp = torch.sum(dist.log_prob(act), dim=-1)
        
        return dist, logp, entropy
    

class Critic(nn.Module):

    def __init__(self, num_obs, seed=0):
        
        torch.manual_seed(seed)

        super(Critic, self).__init__()

        self.num_obs = num_obs

        # layers
        self.fc1 = nn.Linear(num_obs,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,1)

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
    



# ======== For DDPG Agent =========


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
        
        return torch.clamp(self.forward(state), -1, 1)
    

class QCritic(nn.Module):

    def __init__(self, num_obs, num_act, seed=0):
        
        torch.manual_seed(seed)

        super(QCritic, self).__init__()

        self.num_obs = num_obs

        # ------ layers ------
        
        # state path
        self.sfc1 = nn.Linear(num_obs,64)
        self.sfc2 = nn.Linear(64,64)
        
        # action path
        self.afc1 = nn.Linear(num_act,64)
        self.afc2 = nn.Linear(64,64)
        
        # common path
        self.cfc1 = nn.Linear(64*2,64)
        self.cfc2 = nn.Linear(64,1)

        
    def forward(self, state, action):
        
        # convert to torch
        if isinstance(state, numpy.ndarray):
            s = torch.from_numpy(state).float().to(self.device)
        elif isinstance(state, torch.Tensor):
            s = state
        else:
            raise TypeError("Input must be a numpy array or torch Tensor.")
            
        if isinstance(action, numpy.ndarray):
            a = torch.from_numpy(action).float().to(self.device)
        elif isinstance(action, torch.Tensor):
            a = action
        else:
            raise TypeError("Input must be a numpy array or torch Tensor.")

        # state path
        xs = F.relu(self.sfc1(s))
        xs = F.relu(self.sfc2(xs))
        
        # action path
        xa = F.relu(self.afc1(a))
        xa = F.relu(self.afc2(xa))
        
        # common path
        xc = torch.cat((xs,xa), dim=1)
        xc = F.relu(self.cfc1(xc))
        xc = self.cfc2(xc)

        return xc


    def Q(self, state, action):
        
        return self.forward(state, action)

