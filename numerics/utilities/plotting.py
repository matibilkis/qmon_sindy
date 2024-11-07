import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.getcwd())
from numerics.utilities.misc import *
import numpy as np

x = load_data(itraj=10)

plt.figure(figsize=(5,2))
ax=plt.subplot(131)
ax.plot(x[:,0])
ax=plt.subplot(132)
ax.plot(x[:,1])
ax=plt.subplot(133)
ax.plot(x[:,0],x[:,1])

plt.show()
