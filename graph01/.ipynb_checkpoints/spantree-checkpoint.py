#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import scipy
import pandas
import numpy as np


# In[ ]:


x = np.linspace(0, 10, 100)
y = x + np.random.randn(100)

plt.plot(x, y, label="test")

plt.legend()

plt.show()

