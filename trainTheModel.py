import os
import sys
import time
import math
import argparse
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from barsDataSet import createBarDataSets

trainingSet, testingSet = createBarDataSets()

#global constants
BLOCK_SIZE = 16

class MyModel(nn.Module):
    def __init__(self):
        self.mlp = nn.Sequential(
            nn.Linear(BLOCK_SIZE, BLOCK_SIZE),
            nn.Tanh(),
            nn.Linear(BLOCK_SIZE, BLOCK_SIZE)
        )

    def get_block_size(self):
        return BLOCK_SIZE
        
