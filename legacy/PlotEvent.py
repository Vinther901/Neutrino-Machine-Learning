import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

seq = pd.read_csv('data/sequential.csv')

pos = seq[['dom_x','dom_y','dom_z']].drop_duplicates()

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(pos.iloc[:,0],pos.iloc[:,1],pos.iloc[:,2],alpha=0.05)

fig.show()

events = seq['event_no'].unique()

def plot(index):
    ax.cla()
    ax.scatter(pos.iloc[:,0],pos.iloc[:,1],pos.iloc[:,2],alpha=0.05)
    pos_event = seq.loc[seq['event_no'] == events[index]][['dom_x','dom_y','dom_z','dom_charge']]
    event = ax.scatter3D(pos_event['dom_x'],pos_event['dom_y'],pos_event['dom_z'],c=pos_event['dom_charge'],cmap='plasma')
    fig.canvas.draw()