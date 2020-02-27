import pandas as pd
import numpy as np
import torch


DEVICE = torch.device("cpu")

class cryptoData(object):

    def __init__(self,currency,test = False):
        data = pd.read_csv("data/data/"+currency+"Final.csv")

        data = data.drop(["Unnamed: 0",
                        "Unnamed: 0_x",
                        "timestamp",
                        "datetime",
                        "index",
                        "top100cap-"+currency,
                        "mediantransactionvalue-"+currency,
                        "Unnamed: 0_y"
                        ], axis=1)
        self.test = test

        if test:
            data = data.loc[750:850,:]
        else:
            data = data.loc[50:749,:]

        self.data = torch.Tensor(data.to_numpy()).to(DEVICE)

        self.data_clone = self.data.clone()
        self.data/=self.data.mean(0,keepdim=True)[0]
        self.data[:,20] = self.data_clone[:,20]


    def __getitem__(self,key):
        seven_day_data = self.data[key:key+7,:]
        target = self.data_clone[key+2,20]
        return seven_day_data, target

    
    def __len__(self):
        if self.test:
            return 90
        return 680