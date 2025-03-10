import torch
from torch import nn

class Encoder(nn.Module):

    def __init__(self, config):
        super(Encoder,self).__init__()
        self.linear1 = nn.Linear(config["in_feature"], 64)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(64, 64)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(64, 64)
        self.relu3 = nn.ReLU()
  
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        return x

class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.encoder = Encoder(config=config)
        self.classifier = nn.Linear(64, config.n_class)
    
    def forward(self, x):
        feature = self.encoder(x)
        logits = self.classifier(feature)
        return {"feature": feature, "logits": logits}