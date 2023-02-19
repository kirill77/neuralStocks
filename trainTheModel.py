import os
import sys
import time
import math
import argparse
from typing import List
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from barsDataSet import createBarDataSets
from theModel import initModelConfig, MyModel

#global constants
BATCH_SIZE = 32
LEARNING_RATE = 0.1
WEIGHT_DECAY = 0.001

trainingSet, testingSet = createBarDataSets()

batchLoader = DataLoader(dataset = trainingSet, batch_size = BATCH_SIZE, shuffle = True)

modelConfig = initModelConfig(trainingSet)
model = MyModel(modelConfig)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.99), eps=1e-8)

lossFunction = nn.MSELoss()

# Define the training loop
def train(model, batchLoader, lossFunction, optimizer):
    # Set the model to train mode
    model.train()

    # Initialize the total loss and total number of batches
    total_loss = 0
    num_batches = 0    

    # Loop over the training data
    for data, target in batchLoader:
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(data)

        # Calculate the loss
        loss = lossFunction(output, target)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update the total loss and total number of batches
        total_loss += loss.item()
        num_batches += 1

    # Print the average loss for the epoch
    print('Average loss:', total_loss / num_batches)

# Example usage of the training loop
for epoch in range(100):
    train(model, batchLoader, lossFunction, optimizer)