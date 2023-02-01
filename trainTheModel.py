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

import csv
from datetime import datetime

import functools

class Bar:
    def __init__(self, iTicker, csvRow):
        self.iTicker = -1;
        # try to read data from the row
        if (len(csvRow) < 6):
            return
        try:
            substring = csvRow[0][0:19]
            self.startTime = datetime.strptime(substring, "%Y-%m-%d %H:%M:%S")
            self.fOpen = float(csvRow[1])
            self.fHigh = float(csvRow[2])
            self.fLow = float(csvRow[3])
            self.fClose = float(csvRow[4])
            self.fVolume = float(csvRow[5])
        except: return
        self.iTicker = iTicker

def readTickersAndBars(nMaxTickers = -1, nMaxBarsPerTicket = -1):
    # load list of tickers
    with open("tickers.txt", 'r') as f:
        data = f.read()
    allTickers = data.splitlines()
    allTickers = [w.strip() for w in allTickers] # get rid of any leading or trailing white space
    allTickers = [w for w in allTickers if w] # get rid of any empty strings
    if (nMaxTickers != -1):
        allTickers = allTickers[0:nMaxTickers]
    print(f"number of tickers in the dataset: {len(allTickers)}")

    allBars = []

    # for each ticker - load its bars from the file
    for iTicker, sTicker in enumerate(allTickers):
        fileName = f"./history/{sTicker}_1h.csv"
        print(f"loading bars for {sTicker}")
        nBarsThisTicker = 0;
        with open (fileName) as csvFile:
            csvRows = csv.reader(csvFile)
            for csvRow in csvRows:
                bar = Bar(iTicker, csvRow)
                if (bar.iTicker < 0):
                    continue
                allBars.append(bar)
                nBarsThisTicker += 1
                if nMaxBarsPerTicket > 0 and nBarsThisTicker >= nMaxBarsPerTicket:
                    break
    print(f"loaded {len(allBars)} bars")

    return allTickers, allBars

class BarsPerTime:

    def __init__(self, startTime, pBars):
        self.startTime = startTime
        self.pBars = pBars

def compareBarsPerTime(a, b):
    if a.startTime > b.startTime:
        return -1
    elif a.startTime == b.startTime:
        return 0
    else:
        return 1

class BarsDataSet(Dataset):

    def __init__(self, allTickers, allBars):
        self.allTickers = allTickers

        allStartTimes = {}
        for bar in allBars:
            # try/except will create an array if it doesn't exist yet
            try:
                allStartTimes[bar.startTime].append(bar)
            except: allStartTimes[bar.startTime] = [bar]

        # repackage the dict as an array
        self.allStartTimes = []
        for startTime in allStartTimes:
            self.allStartTimes.append(BarsPerTime(startTime, allStartTimes[startTime]))
        # sort times in ascending order
        self.allStartTimes.sort(key=functools.cmp_to_key(compareBarsPerTime))

        self.nPastBars = 3 # how many bars we use for prediction
        self.nFutureBars = 1 # how many bars ahead we're trying to predict
        self.nTotalExamples = len(self.allStartTimes) - self.nPastBars - self.nFutureBars + 1
        self.nTrainingExamples = int(self.nTotalExamples * 9 / 10)
        # on each timestep each ticker is a separate example
        self.nTotalExamples *= len(allTickers)
        self.nTrainingExamples *= len(allTickers)

    def __len__(self):
        return self.nTrainingExamples

    def __getitem__(self, idx):
        # as input we provide:
        # for each ticker: open for the first bar
        # for each ticker and for each past bar: min, max, close, volume
        nTickers = len(self.allTickers)
        x = torch.zeros(nTickers + nTickers * self.nPastBars * 4)
        for iStartTime in range(idx + self.nPastBars - 1, -1, -1):
            for bar in self.allStartTimes:
                bar = bar
        y = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        x[1:1+len(ix)] = ix
        y[:len(ix)] = ix
        y[len(ix)+1:] = -1 # index -1 will mask the loss at the inactive locations
        return x, y


nMaxTickers = 10
nMaxBarsPerTicket = 100
allTickers, allBars = readTickersAndBars(10, 100)
dataSet = BarsDataSet(allTickers, allBars)
dataSet = dataSet