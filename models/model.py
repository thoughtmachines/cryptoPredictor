import torch
from torch import nn

DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = "cuda:0"
    print("Using CUDA backend")

class SeqRegressor(nn.Module):

    def __init__(self):
        super(SeqRegressor,self).__init__()

        self.lstm = nn.LSTM(23,20,1)
        #nn.LSTM(input_size,hidden_size,num_layers)
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

    def __init__(self):
        super(MLPRegressor,self).__init__()

        self.relu = nn.ReLU()

        self.layer1 = nn.Linear(161,100)
        self.layer1_bn = nn.BatchNorm1d(100)

        self.layer2 = nn.Linear(100,50)
        self.layer2_bn = nn.BatchNorm1d(50)

        self.layer3 = nn.Linear(50,25)
        self.layer3_bn = nn.BatchNorm1d(25)

        
        self.layer4 = nn.Linear(25,10)
        self.layer4_bn = nn.BatchNorm1d(10)
        

        
        self.layer5 = nn.Linear(10,1)
    
    def forward(self,x):
        x = x.to(DEVICE)
        x = x.flatten().view(1,161)
        x = self.relu(self.layer1(x))
        #x = self.layer1_bn(x)
        x = self.relu(self.layer2(x))
        #x = self.layer2_bn(x)
        x = self.relu(self.layer3(x))
        #x = self.layer3_bn(x)
        x = self.relu(self.layer4(x))
        x = self.relu(self.layer5(x))
        return x



