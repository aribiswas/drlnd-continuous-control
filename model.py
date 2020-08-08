import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

class StochasticPolicy(nn.Module):

    def __init__(self, num_obs, num_act, seed):
        """
        Neural network for a stochastic continuous action policy.

        Parameters
        ----------
        num_obs: number
            Dimension of observations.
        num_act: number
            Dimension of actions.
        seed: number
            random seed.

        Returns
        -------
        None.

        """
        torch.manual_seed(seed)

        super(StochasticPolicy, self).__init__()

        self.num_obs = num_obs
        self.num_act = num_act

        # layers
        self.fc1 = nn.Linear(num_obs, 256)
        self.fc2 = nn.Linear(256, 128)
        self.meanfc1 = nn.Linear(128, 64)
        self.meanfc2 = nn.Linear(64, 4)
        self.stdfc1 = nn.Linear(128, 64)
        self.stdfc2 = nn.Linear(64, num_act)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters
        ----------
        x : numpy.array
            Input of the network.

        Returns
        -------
        meanx: torch.Tensor
            Mean of actions.
        stdx: torch.Tensor
            Std dev of actions.

        """

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        meanx = F.relu(self.meanfc1(x))
        meanx = self.tanh(self.meanfc2(meanx))
        stdx = F.relu(self.stdfc1(x))
        stdx = self.softmax(self.stdfc2(stdx))

        return meanx, stdx


    def get_action(self, state):
        """
        Sample action from policy, given a state.

        Parameters
        ----------
        state : numpy array
            State of the environment.

        Returns
        -------
        action : number
            Move right (3) or left(4).
        log_prob : torch.Tensor
            Log probability of action.

        """
        
        # convert to torch
        x = torch.from_numpy(state).float().to(self.device)

        # obtain mean and std from network
        mean, std = self.forward(x)

        # create Gaussian distribution
        cov_matrix = torch.diag(std)
        dist = MultivariateNormal(mean, cov_matrix)

        # sample action
        action = dist.sample()
        action = torch.clamp(action, -1, 1).detach().numpy()  # limit actions to [-1,1]

        # obtain log probabilities
        logp = dist.log_prob(action).detach()

        return action, logp
    



class ValueNetwork(nn.Module):

    def __init__(self, num_obs, seed):
        """
        Neural network for approximating a value function V(s).

        Parameters
        ----------
        num_obs: number
            Dimension of observations.
        seed: number
            Random seed.

        Returns
        -------
        None.

        """
        torch.manual_seed(seed)

        super(ValueNetwork, self).__init__()

        self.num_obs = num_obs

        # layers
        self.fc1 = nn.Linear(num_obs, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters
        ----------
        x : numpy.array
            Input of the network.

        """

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)   

        return x


    def get_value(self, state):
        """
        Get the state value V(s), given the state.

        Parameters
        ----------
        state : numpy array
            State of the environment.

        Returns
        -------
        torch.Tensor
            State value V(s) of the observations.

        """
        
        return self(state)

