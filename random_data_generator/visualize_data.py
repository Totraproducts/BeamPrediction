# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 09:47:39 2019

@author: Rahul Sharma
"""

import pandas as pd
from matplotlib import pyplot as plt

dataframe = pd.read_csv("export_dataframe.csv")
x=dataframe["x_coordinate"][0:3]
y=dataframe["y_coordinate"][0:3]
plt.plot(x,y)