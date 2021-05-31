import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

colors = []
for l in np.linspace(1, 0, 100):
    colors.append((30./255, 136./255, 229./255,l))
for l in np.linspace(0, 1, 100):
    colors.append((255./255, 13./255, 87./255,l))
red_transparent_blue = LinearSegmentedColormap.from_list("red_transparent_blue", colors)

