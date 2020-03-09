import torch
from torch import nn


class SeqRegressor(nn.Module):

    def __init__(self,hidden_size=23):
        super(SeqRegressor,self).__init__()

        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(23,hidden_size,1)

        self.layer1 = nn.Linear(hidden_size,20)
        self.layer2 = nn.Linear(20,15)
        self.layer3 = nn.Linear(15,7)
        self.layer4 = nn.Linear(7,1)
        self.relu = nn.ReLU()

        hidden_state = torch.ones(1,1, hidden_size)
        cell_state = torch.ones(1,1, hidden_size)
        self.hidden = (hidden_state,cell_state)

    def forward(self,x):

        """
        hidden_state = torch.randn(no_stack, batch_size, hidden_dim)
        cell_state = torch.randn(no_stack, batch_size, hidden_dim)
        """
        
        out,_ = self.lstm(x,self.hidden)
        
        x = self.relu(out[:,-1,:][-1])
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.relu(self.layer4(x))
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

        x = self.relu(self.layer6(x))
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
            x+= (x- self.memory[4])* torch.rand(x.shape)* 0.1
            self.memory[4] = x.clone()

        x = self.relu(self.layer6(x))
        
        return x