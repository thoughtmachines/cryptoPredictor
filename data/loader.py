import pandas as pd
import numpy as np
import torch


class cryptoData(object):

    def __init__(self,currency,test = False,DEVICE = torch.device("cpu")):
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

        if currency == "btc":
            var = "bitcoin-price"
        elif currency == "eth":
            var = "ethereum-price"
        else:
            var = "litecoin-price"

        data = data.iloc[:931]
        train_data = data.iloc[50:700]
        test_data = data.iloc[700:]

        price = np.asarray(data[var])
        mean = train_data.mean()

        train_data = train_data/mean
        test_data = test_data/mean
        
        train_data = torch.Tensor(train_data.to_numpy())
        test_data = torch.Tensor(test_data.to_numpy())

        train_data[:,5] = torch.Tensor(price[50:700])
        test_data[:,5] = torch.Tensor(price[700:])

        xtrain = []
        ytrain = []

        for i in range(len(train_data)-7):
            xtrain.append(np.asarray(train_data[i:i+7,:]))
            ytrain.append(np.asarray(train_data[i+7][5]))

        self.xtrain = torch.Tensor(np.asarray(xtrain)).to(DEVICE)
        self.ytrain = torch.Tensor(np.asarray(ytrain)).to(DEVICE)

        xtest = []
        ytest = []
        for i in range(len(test_data)-7):
            xtest.append(np.asarray(test_data[i:i+7,:]))
            ytest.append(np.asarray(test_data[i+7][5]))

        self.xtest = torch.Tensor(np.asarray(xtest)).to(DEVICE)
        self.ytest = torch.Tensor(np.asarray(ytest)).to(DEVICE)


    def __getitem__(self,key):
        if not self.test:
            seven_day_data = self.xtrain[key]
            target = self.ytrain[key]
        else:
            seven_day_data = self.xtest[key]
            target  = self.ytest[key]
        return seven_day_data, target

    
    def __len__(self):
        if self.test:
            return 223
        return 643