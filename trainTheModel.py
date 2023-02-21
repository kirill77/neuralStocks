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
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from barsDataSet import createBarDataSets
from theModel import initModelConfig, MyModel

#global constants
BATCH_SIZE = 512
LEARNING_RATE = 0.1
WEIGHT_DECAY = 0.001
IS_CUDA = True

# Define the training loop
def train(model, batchLoader, lossFunction, optimizer, iEpoch):

    # Set the model to train mode
    model.train()

    # Initialize the total loss and total number of batches
    total_loss = 0
    num_batches = 0    

    t0 = time.time()

    # Loop over the training data
    for data, target in batchLoader:
        if (IS_CUDA):
            data = data.to("cuda")
            target = target.to("cuda")
        
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

    # Print stats about this epoch
    if (IS_CUDA):
        torch.cuda.synchronize()
    t1 = time.time()
    avgLoss = total_loss / num_batches
    print(f"Epoch {iEpoch}, nBatches: {num_batches}, avg.loss: {avgLoss}, time: {(t1-t0)*1000:.2f}ms, "\
          f"lr: {optimizer.state_dict()['param_groups'][0]['lr']}")
    return avgLoss

def main():
    writer = SummaryWriter('logs')

    trainingSet, testingSet = createBarDataSets()

    print(f"number of training examples: {len(trainingSet)}")
    print(f"number of testing examples: {len(testingSet)}")

    print("creating data loader...")
    batchLoader = DataLoader(dataset = trainingSet, batch_size = BATCH_SIZE, shuffle = True)

    modelConfig = initModelConfig(trainingSet)
    model = MyModel(modelConfig)
    if (IS_CUDA):
        model.to("cuda") # run on the GPU

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.99), eps=1e-8)

    scheduler = StepLR(optimizer, step_size=2, gamma=0.5)

    lossFunction = nn.MSELoss()
    if (IS_CUDA):
        lossFunction.to("cuda")

    # Example usage of the training loop
    print("starting training...")
    for iEpoch in range(100):
        avgLoss = train(model, batchLoader, lossFunction, optimizer, iEpoch)
        scheduler.step()

        # Log the loss to TensorBoard
        writer.add_scalar('training_loss', avgLoss, iEpoch)

    writer.close()

if __name__ == "__main__":
    main()