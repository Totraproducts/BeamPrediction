# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 09:31:28 2019

@author: Rahul Sharma
"""

import numpy as np
import pandas as pd
from numpy.random import randint
import random
import matplotlib.pyplot as plt

grid_size = 8;
no_of_dataset = 500;
graph_beam_number = 5
beam_angle = 30

df = pd.DataFrame(columns=['x_coordinate', 'y_coordinate', 'Azimuth', 'Elevation', 'Beam Strength(dB)','Beam Number'])

azimuthRad_list=[]
radius_list = []

fig = plt.figure()

ax = plt.subplot(111, projection='polar')

for i in range(no_of_dataset):
    azimuthAng = random.uniform(0,361)
    
    coor_list = list(randint(grid_size, size=2))
    df.loc[i] = coor_list + [azimuthAng] + [random.uniform(0,90)] + [randint(-30,-19)] + [int(azimuthAng/beam_angle)]
    
    if int(azimuthAng/beam_angle)==graph_beam_number:
        azimuthRad_list.append(int(azimuthAng)*np.pi/180)
        radius_list.append((coor_list[0]**2 + coor_list[1]**2)**(0.5))
        
export_csv = df.to_csv (r'export_dataframe.csv', index = None, header=True)
ax.scatter(azimuthRad_list, radius_list)
plt.title('Scatter Polar Plot of Azimuth angle vs ue-distance')
plt.show()
fig.savefig('Beam_scatter-plot.jpg')
