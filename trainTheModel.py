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

# Global constants
BATCH_SIZE = 512
LEARNING_RATE = 0.1
WEIGHT_DECAY = 0.001
SCHEDULER_STEPS = 2
SCHEDULER_GAMMA = 0.75
IS_CUDA = True

# Create device-dependent variables
if (IS_CUDA):
    startEvent = torch.cuda.Event(enable_timing=True)
    endEvent = torch.cuda.Event(enable_timing=True)
    pDevice = torch.device("cuda")
else: pDevice = torch.device("cpu")
# For some reason this works much faster on the CPU - need to figure out
pDataSetDevice = torch.device("cpu")

# Define the training loop
def trainAnEpoch(model, batchLoader, lossFunction, optimizer, iEpoch):

    # Chat GPT advices to call this once in the beginning of each epoch
    model.train()

    # Initialize the total loss and total number of batches
    total_loss = 0
    num_batches = 0    

    cpuStart = time.time()
    totalGPUTime = 0

    # Loop over the training data
    for data, target in batchLoader:
        if (IS_CUDA):
            # Record the start time
            startEvent.record()
        if (data.device.type != pDevice.type):
            data = data.to(pDevice)
            target = target.to(pDevice)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(data)

        # Calculate the loss
        loss = lossFunction(output, target)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        if (IS_CUDA):
            # Record the end time
            endEvent.record()
            torch.cuda.synchronize()
            totalGPUTime += startEvent.elapsed_time(endEvent)

        # Update the total loss and the total number of batches
        total_loss += loss.item()
        num_batches += 1

    # Print stats about this epoch
    cpuEnd = time.time()
    if (iEpoch == 0):
        print(f"last batch input/output: {data.shape}/{target.shape}, nBatches: {num_batches}")
    avgLoss = total_loss / num_batches
    print(f"Epoch {iEpoch}, avg.loss: {avgLoss:.4f}, CPUTime: {(cpuEnd-cpuStart)*1000:.2f}ms, GPUTime: {totalGPUTime:.2f}ms, "\
          f"lr: {optimizer.state_dict()['param_groups'][0]['lr']:.6f}")
    return avgLoss

def main():
    writer = SummaryWriter('logs')

    trainingSet, testingSet = createBarDataSets(pDataSetDevice)

    print(f"number of training examples: {len(trainingSet)}")
    print(f"number of testing examples: {len(testingSet)}")

    print("creating data loader...")
    batchLoader = DataLoader(dataset = trainingSet, batch_size = BATCH_SIZE, shuffle = True)

    modelConfig = initModelConfig(trainingSet)
    model = MyModel(modelConfig, pDevice)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.99), eps=1e-8)

    scheduler = StepLR(optimizer, step_size=SCHEDULER_STEPS, gamma=SCHEDULER_GAMMA)

    lossFunction = nn.MSELoss()
    lossFunction.to(pDevice)

    print("starting training...")
    for iEpoch in range(100):
        # Train an epoch
        avgLoss = trainAnEpoch(model, batchLoader, lossFunction, optimizer, iEpoch)
        # Adjust learning rate
        scheduler.step()

        # Log the loss to TensorBoard
        writer.add_scalar('training_loss', avgLoss, iEpoch)

    writer.close()

if __name__ == "__main__":
    main()