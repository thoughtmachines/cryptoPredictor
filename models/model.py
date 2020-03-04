import torch
from torch import nn


class SeqRegressor(nn.Module):

    def __init__(self):
        super(SeqRegressor,self).__init__()

        self.lstm = nn.LSTM(23,20,1)

        self.lin = nn.Linear(140,50)
        self.lin2 = nn.Linear(50,10)
        self.lin3 = nn.Linear(10,5)
        self.lin4 = nn.Linear(5,1)
        self.relu = nn.ReLU()

        hidden_state = torch.ones(1,1, 20)
        cell_state = torch.ones(1,1, 20)
        self.hidden = (hidden_state,cell_state)

    def forward(self,x):

        """
        hidden_state = torch.randn(no_stack, batch_size, hidden_dim)
        cell_state = torch.randn(no_stack, batch_size, hidden_dim)
        """
        out,hidden = self.lstm(x,self.hidden)
        out  = out.flatten().view(1,140)
        x = self.relu(out)
        x = self.relu(self.lin(x))
        x = self.relu(self.lin2(x))
        x = self.relu(self.lin3(x))
        x = self.lin4(x)
        return x


class MLPRegressor(nn.Module):

    def __init__(self,stochastic=False,x=None):
        super(MLPRegressor,self).__init__()
        self.stochastic = stochastic
        self.relu = nn.ReLU()

        self.layer1 = nn.Linear(161,130)
        self.layer2 = nn.Linear(130,100)
        self.layer3 = nn.Linear(100,50)
        self.layer4 = nn.Linear(50,25)
        self.layer5 = nn.Linear(25,10)
        self.layer6 = nn.Linear(10,1)

    

    def eval(self,x):
        x = x.flatten().view(1,161)
        memory = []

        x = self.relu(self.layer1(x))
        memory.append(x)

        x = self.relu(self.layer2(x))
        memory.append(x)

        x = self.relu(self.layer3(x))
        memory.append(x)

        x = self.relu(self.layer4(x))
        memory.append(x)

        x = self.relu(self.layer5(x))
        memory.append(x)

        self.memory = memory
    
    def forward(self,x):
        x = x.flatten().view(1,161)

        x = self.relu(self.layer1(x))
        if self.stochastic:
            x+= (x- self.memory[0])* torch.rand(x.shape) * 0.1
            self.memory[0] = x.clone()

        x = self.relu(self.layer2(x))
        if self.stochastic:
            x+= (x- self.memory[1])* torch.rand(x.shape)* 0.1
            self.memory[1] = x.clone()

        x = self.relu(self.layer3(x))
        if self.stochastic:
            x+= (x- self.memory[2])* torch.rand(x.shape)* 0.1
            self.memory[2] = x.clone()

        x = self.relu(self.layer4(x))
        if self.stochastic:
            x+= (x- self.memory[3])* torch.rand(x.shape)* 0.1
            self.memory[3] = x.clone()

        x = self.relu(self.layer5(x))
        if self.stochastic:
            x+= (x- self.memory[4])* torch.rand(x.shape)* 1
            self.memory[4] = x.clone()

        x = self.relu(self.layer6(x))
        return x