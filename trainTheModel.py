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
def trainAnEpoch(model, trainingBatches, testingBatches, lossFunction, optimizer, iEpoch):

    # Chat GPT advices to call this once in the beginning of each epoch
    model.train()

    # Initialize the total loss and the total number of batches
    totalTrainingLoss = 0
    numTrainingBatches = 0
    totalTestingLoss = 0
    numTestingBatches = 0

    cpuStart = time.time()
    totalGPUTime = 0

    # Loop over the training data
    trainIter = iter(trainingBatches)
    # Assume that at least one batch is available
    data, target = next(trainIter)

    while True:
        if (IS_CUDA):
            # Record the GPU start time
            startEvent.record()

        # Copy tensors to the training device
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
            # Record the GPU end time
            endEvent.record()

        # try to get the next batch while GPU is still busy with the previous batch
        try:
            nextData, nextTarget = next(trainIter)
        except StopIteration: nextData = None

        if (IS_CUDA):
            torch.cuda.synchronize()
            totalGPUTime += startEvent.elapsed_time(endEvent)

        # Update the total loss and the total number of batches
        fLoss = loss.item();
        totalTrainingLoss += fLoss
        numTrainingBatches += 1

        if nextData is None:
            break # no more batches
        data = nextData
        target = nextTarget

    # Compute error on testing batches
    with torch.no_grad():
        for data, target in testingBatches:

            # Copy tensors to the training device
            if (data.device.type != pDevice.type):
                data = data.to(pDevice)
                target = target.to(pDevice)

            # Forward pass
            output = model(data)
            # Calculate the loss
            loss = lossFunction(output, target)

            # Update the total loss and the total number of batches
            fLoss = loss.item();
            totalTestingLoss += fLoss
            numTestingBatches += 1

    # Print stats about this epoch
    cpuEnd = time.time()
    if (iEpoch == 0):
        print(f"last batch input/output: {data.shape}/{target.shape}, nBatches: {numTrainingBatches}")
    avgTrainingLoss = totalTrainingLoss / numTrainingBatches
    avgTestingLoss = totalTestingLoss / numTestingBatches
    print(f"Epoch {iEpoch}, trainingLoss: {avgTrainingLoss:.4f}, testingLoss: {avgTestingLoss:.4f}, "
          f"CPUTime: {(cpuEnd-cpuStart)*1000:.2f}ms, GPUTime: {totalGPUTime:.2f}ms, "\
          f"lr: {optimizer.state_dict()['param_groups'][0]['lr']:.6f}")
    return avgTrainingLoss

def main():
    writer = SummaryWriter('logs')

    trainingSet, testingSet = createBarDataSets(pDataSetDevice)

    print(f"number of training examples: {len(trainingSet)}")
    print(f"number of testing examples: {len(testingSet)}")

    print("creating data loader...")
    trainingBatches = DataLoader(dataset = trainingSet, batch_size = BATCH_SIZE, shuffle = True)
    testingBatches = DataLoader(dataset = testingSet, batch_size = BATCH_SIZE, shuffle = False)

    modelConfig = initModelConfig(trainingSet)
    model = MyModel(modelConfig, pDevice)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.99), eps=1e-8)

    scheduler = StepLR(optimizer, step_size=SCHEDULER_STEPS, gamma=SCHEDULER_GAMMA)

    lossFunction = nn.CrossEntropyLoss()
    lossFunction.to(pDevice)

    print("starting training...")
    for iEpoch in range(100):
        # Train an epoch
        avgTrainingLoss = trainAnEpoch(model, trainingBatches, testingBatches, lossFunction, optimizer, iEpoch)
        # Adjust learning rate
        scheduler.step()

        # Log the loss to TensorBoard
        writer.add_scalar('trainingLoss', avgTrainingLoss, iEpoch)

    writer.close()

if __name__ == "__main__":
    main()