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
LEARNING_RATE = 1
WEIGHT_DECAY = 0.001
IS_CUDA = True
N_FAILED_LEARNING_STEPS = 10

# Create device-dependent variables
if (IS_CUDA):
    startEvent = torch.cuda.Event(enable_timing=True)
    endEvent = torch.cuda.Event(enable_timing=True)
    pDevice = torch.device("cuda")
else: pDevice = torch.device("cpu")
# For some reason this works much faster on the CPU - need to figure out
pDataSetDevice = torch.device("cpu")

# Define the training loop
def trainAnEpoch(model, trainingBatches, testingBatches, lossFunction, iEpoch, optimizer = None):

    # If optimizer is not there - means we're not going to do training
    if optimizer:
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)

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

    for data, target in trainingBatches:
        if (IS_CUDA):
            # Record the GPU start time
            startEvent.record()

        # Copy tensors to the training device
        if (data.device.type != pDevice.type):
            data = data.to(pDevice)
            target = target.to(pDevice)

        # Zero the gradients
        if optimizer:
            optimizer.zero_grad()

        # Forward pass
        output = model(data)
        # Calculate the loss
        loss = lossFunction(output, target)

        if optimizer:
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        if (IS_CUDA):
            # Record the GPU end time
            endEvent.record()
            torch.cuda.synchronize()
            totalGPUTime += startEvent.elapsed_time(endEvent)

        # Update the total loss and the total number of batches
        fLoss = loss.item();
        totalTrainingLoss += fLoss
        numTrainingBatches += 1

    # Compute error on testing batches - light weight pass without gradients
    model.eval()
    torch.set_grad_enabled(False)
    for data, target in testingBatches:
        if (IS_CUDA):
            # Record the GPU start time
            startEvent.record()

        # Copy tensors to the training device
        if (data.device.type != pDevice.type):
            data = data.to(pDevice)
            target = target.to(pDevice)

        # Forward pass
        output = model(data)
        # Calculate the loss
        loss = lossFunction(output, target)

        if (IS_CUDA):
            # Record the GPU end time
            endEvent.record()
            torch.cuda.synchronize()
            totalGPUTime += startEvent.elapsed_time(endEvent)

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
    lr = 0.
    if optimizer:
        lr = optimizer.state_dict()['param_groups'][0]['lr']
    print(f"Epoch {iEpoch}, trainingLoss: {avgTrainingLoss:.4f}, testingLoss: {avgTestingLoss:.4f}, "
          f"CPUTime: {(cpuEnd-cpuStart)*1000:.2f}ms, GPUTime: {totalGPUTime:.2f}ms, "\
          f"lr: {lr:.6f}")
    return avgTestingLoss

def main():
    writer = SummaryWriter('logs')

    trainingSet, testingSet = createBarDataSets(pDataSetDevice)

    print(f"number of training examples: {len(trainingSet)}")
    print(f"number of testing examples: {len(testingSet)}")

    print("creating data loader...")
    trainingBatches = DataLoader(dataset = trainingSet, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4, persistent_workers = True)
    testingBatches = DataLoader(dataset = testingSet, batch_size = BATCH_SIZE, shuffle = False, num_workers = 4, persistent_workers = True)

    modelConfig = initModelConfig(trainingSet)
    model = MyModel(modelConfig, pDevice)
    model.loadFromDir('.')

    lossFunction = nn.CrossEntropyLoss()
    lossFunction.to(pDevice)

    print("calculating the initial loss...")
    bestSavedTestingLoss = trainAnEpoch(model, trainingBatches, testingBatches, lossFunction, iEpoch=-1)

    global LEARNING_RATE
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    print("starting training...")
    bestTestingLoss = None
    nFailedLearningSteps = 0
    iEpoch = 0
    while True:
        # Train an epoch
        avgTestingLoss = trainAnEpoch(model, trainingBatches, testingBatches, lossFunction, iEpoch, optimizer)

        # If we've failed to improve avgTesting loss some number of times - decrease the learning rate
        if (bestTestingLoss is None or avgTestingLoss < bestTestingLoss):
            nFailedLearningSteps = 0
            bestTestingLoss = avgTestingLoss
            if (bestTestingLoss < bestSavedTestingLoss):
                print("    ** new all-time-best model (saving)...")
                bestSavedTestingLoss = bestTestingLoss
                torch.save(model.state_dict(), 'bestModel.pth');
            else:
                print("    ** new local-best model (not saving)")
        else:
            nFailedLearningSteps += 1
            if (nFailedLearningSteps >= N_FAILED_LEARNING_STEPS):
                print("    ** decreasing learning rate")
                nFailedLearningSteps = 0
                model.loadFromDir('.') # load the all-time-best model
                LEARNING_RATE /= 2
                for param_group in optimizer.param_groups:
                    param_group['lr'] = LEARNING_RATE
            elif (math.isnan(avgTestingLoss)):
                model.loadFromDir('.') # load the all-time-best model

        # Log the loss to TensorBoard
        writer.add_scalar('trainingLoss', avgTestingLoss, iEpoch)

        iEpoch += 1

    writer.close()

if __name__ == "__main__":
    main()