# !/usr/bin/env python3
import numpy as np
import math
import time

import random

from math import pi

import sys
import os
from Agent import Agent

agent=Agent()
agent.reset()

TT=10000

for t in np.arange(TT):
    agent.episode(tt)
    agent.reset()
    if t%50==0:
        print(t)
        agent.rl.saveNets()

## Uncomment to test trained networks:
# agent.reset()
# for t in np.arange(TT):
#     agent.act()