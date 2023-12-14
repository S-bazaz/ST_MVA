# -*- coding: utf-8 -*-
##############
#  Packages  #
##############

import os
import sys
import pywt
from pathlib import Path
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt

pio.renderers.default = "plotly_mimetype+notebook"
from typing import Dict, Optional, Union, BinaryIO, Tuple, List


##################
#      Imports   #
##################
root_path = Path(os.path.abspath(__file__)).parents[1]
sys.path.insert(0, str(root_path))



###################
#   ploting st    #
###################


def plot_all_st(X, clustering=None, title="<b>Signals</b>"):
    """
    Plot multiple signals in a single interactive Plotly figure.

    Args:
        X (list of arrays): A list of signal arrays to be plotted.
        clustering (list or None, optional): A list of cluster assignments for each signal. 
            If provided, signals will be color-coded by cluster. Default is None.
        title (str, optional): The title of the plot. Default is "<b>Signals</b>".

    Returns:
        None: Displays an interactive Plotly figure with the plotted signals.
    """
    
    fig = go.Figure(
        layout=go.Layout(
            height=600, 
            width=800, 
            template = "plotly_dark", 
            title = title
    ))
        
    if clustering:
        pal = ["palegreen", "darkred"]
    else:
        pal = sns.color_palette("Spectral", len(X)).as_hex()

    for i in range(len(X)):
        if clustering:
            color = pal[clustering[i]]
        else:
            color = pal[i]
            
        fig.add_trace(go.Scatter(y=X[i], 
                                 mode="lines", 
                                 line=dict(
                                     width=2,
                                     color=color,
                                 ),
                                 opacity = 0.6
                                ))
    fig.show()

###################
#   ploting st    #
###################

def plot_signal(vec, title = "signal"):
    fig = px.line(vec, template = "plotly_dark", title = title)
    fig.show()

def add_fig(fig, signal, color, name):
    fig.add_trace(go.Scatter(y=signal, 
                 mode="lines", 
                 line=dict(
                     width=2,
                     color=color,
                 ),
                 opacity = 0.6,
                 name=name
                )
             )
    
def my_pal(n):
    return sns.color_palette("Spectral", n).as_hex()


def plot_estim(xb, true_x, title = "estimate xbar"):
    fig = go.Figure(
        layout=go.Layout(
            height=600, 
            width=800, 
            template = "plotly_dark", 
            title = title
    ))
    pal = my_pal(4)

    add_fig(fig, xb, "darkorchid" ,f"Estimation")
    add_fig(fig, true_x, "palegreen" ,f"Signal")
    fig.show()




def plot_scalogram(sig, scales, 
                 waveletname = 'cmor', 
                 levels = np.linspace(1e-1, 3, 40),
                 title = 'Scalogram of signal', 
                 fs = 500,
                 cmap = plt.cm.plasma, 
                 ):
    plt.clf()
    dt = 1/fs
    n = sig.shape[0]
    time = np.arange(0, n)/fs
    [coefficients, frequencies] = pywt.cwt(sig, scales, waveletname, dt)
    power = (abs(coefficients)) ** 2
    period = 1. / frequencies

    
    contourlevels = np.log2(levels)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.contourf(time, np.log2(period), np.log2(power), contourlevels, extend='both',cmap=cmap)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("Log period (s)", fontsize=8)
    
    ax.set_xlabel("Time (s)", fontsize=8)
    
    yticks = 2**np.arange(np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max())))

    ax.set_yticks(np.log2(yticks))
    ax.set_yticklabels(yticks)

    ax.invert_yaxis()
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], -1)
    plt.show()