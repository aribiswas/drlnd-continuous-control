# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
from utils import OUNoise
import numpy as np

noise_model = OUNoise(size=[1,1], mean=0, mac=0.2, var=0.05, varmin=0.001, decay=5e-6, seed=0)
log = []

for i in range(500*1001):
    noise = noise_model.step()[0][0]
    if i%1001==0:
        log.append(noise)
    
# plot score history
plt.ion()
fig, ax1 = plt.subplots(1,1, figsize=(4,4), dpi=200)
ax1.set_title("OU Noise")
ax1.set_xlabel("Steps")
ax1.set_ylabel("Noise")
ax1.plot(log)
