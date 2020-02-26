import torch
from torch import nn


class SeqRegressor(nn.Module):

    def __init__(self):
        super(SeqRegressor,self).__init__()

class CNNRegressor(nn.Module):

    def __init__(self):
        super(CNNRegressor,self).__init__()

        self.relu = nn.ReLU()
        self.layer1 = nn.Conv2d(1,10,(3,23))
        self.layer1_bn = nn.BatchNorm2d(10)

        self.layer2 = nn.Conv2d(10,5,(3,1))
        self.layer2_bn = nn.BatchNorm2d(5)

        self.layer3 = nn.Conv2d(5,1,(3,1))

    def forward(self,x):
        x = self.relu(self.layer1(x))
        x = self.layer1_bn(x)
        x = self.relu(self.layer2(x))
        x = self.layer2_bn(x)
        x = self.relu(self.layer3(x))
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



