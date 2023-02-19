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

from bar import Bar
from barsDataSet import BarsDataSet

# global constants
N_MAX_TICKERS = 10
N_MAX_BARS_PER_TICKER = 100

def readTickersAndBars():
    # load list of tickers
    with open("tickers.txt", 'r') as f:
        data = f.read()
    allTickers = data.splitlines()
    allTickers = [w.strip() for w in allTickers] # get rid of any leading or trailing white space
    allTickers = [w for w in allTickers if w] # get rid of any empty strings
    if (N_MAX_TICKERS != -1):
        allTickers = allTickers[0:N_MAX_TICKERS]
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
                if N_MAX_BARS_PER_TICKER > 0 and nBarsThisTicker >= N_MAX_BARS_PER_TICKER:
                    break
    print(f"loaded {len(allBars)} bars")

    return allTickers, allBars

allTickers, allBars = readTickersAndBars()
dataSet = BarsDataSet(allTickers, allBars)

# check that the number of items is correct
firstItem = dataSet.__getitem__(0) # first item must exist
lastItem = dataSet.__getitem__(dataSet.nTotalExamples - 1) # last item must exist
# one after last item must NOT exist
errorOccurred = False
try:
    invalidItem = dataSet.__getitem__(dataSet.nTotalExamples)
except: errorOccurred = True
assert(errorOccurred)
