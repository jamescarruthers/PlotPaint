import plotpaint
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from shapely import LineString
import subprocess 
import numpy as np
from easing import *

painting = plotpaint.Plotpaint()

# generate some coordinates
circle = painting.circle((200,200), 150, 128)
# add the circle to the painting, using all the default settings
painting.addStroke(circle)

# change the defaults (see the class init) and add a line
painting.runInEase = painting.ease.circOut
painting.runOutEase = painting.ease.circIn
painting.runIn = 50
painting.runOut = 50
painting.addStroke([(100, 300), (300,100)])

# reset the defaults
painting.reset()

# a slightly advanced option is to process the line whilst adding the stroke
def lineProc(d, x, y, z):
    z += math.sin(d/4) * 0.5
    return [x,y,z]
painting.processLine = lineProc
painting.addStroke([(100,100), (300,300)])

subprocess.run("pbcopy", text=True, input=painting.output("painting.gcode"))

painting.viz([painting.strokes[0], painting.strokes[1], painting.strokes[2]])


