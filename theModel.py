from dataclasses import dataclass
import torch.nn as nn

@dataclass
class ModelConfig:
    nInputFeatures: int = None
    nOutputFeatures: int = None

def initModelConfig(dataSet):
    config = ModelConfig()
    input, output = dataSet.__getitem__(0)
    config.nInputFeatures = len(input)
    config.nOutputFeatures = len(output)
    return config

class MyModel(nn.Module):
    def __init__(self, config, pDevice):
        super().__init__()
        self.layer0 = nn.Linear(in_features = config.nInputFeatures, out_features=config.nInputFeatures * 2, device = pDevice)
        self.function = nn.ReLU()
        self.layer1 = nn.Linear(in_features = config.nInputFeatures * 2, out_features = config.nInputFeatures * 2, device = pDevice)
        self.layer2 = nn.Linear(in_features = config.nInputFeatures * 2, out_features = config.nInputFeatures, device = pDevice)
        self.layer3 = nn.Linear(in_features = config.nInputFeatures, out_features = config.nOutputFeatures, device = pDevice)
        self.to(pDevice)

    def forward(self, x):
        y = self.layer0(x)
        y = self.function(y)
        y = self.layer1(y)
        y = self.function(y)
        y = self.layer2(y)
        y = self.function(y)
        y = self.layer3(y)
        return y
