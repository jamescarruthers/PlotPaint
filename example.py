import plotpaint
import math
import subprocess 
import numpy as np

painting = plotpaint.Plotpaint()

painting.pathOffset = 0.25
painting.addStrokeNP(painting.poly((25,25), 50, 90, 6))
painting.addStrokeNP(painting.poly((painting.pageWidth-25,painting.pageHeight-25), 10, 0, 36))

# generate some coordinates
circle = painting.poly((150,150), 100, 0, 36)
# add the circle to the painting, using default settings
painting.addStrokeNP(circle)
# the addStroke function returns the stroke in the strokes array

circle = painting.poly((150,150), 75, 180, 128)
circle.reverse()
painting.addStrokeNP(circle)

# change the defaults (see the class init) and add a line
painting.runInEase = painting.ease.circOut
painting.runOutEase = painting.ease.circIn
painting.runIn = 50
painting.runOut = 50

sine = []
for x in range(150,200):
        sine.append([100 + math.sin(x/math.pi) * 20, 100+x])
painting.addStrokeNP(sine)

sine = []
for x in range(150,200):
        sine.append([x + 100, 100+math.sin(x/math.pi) * 20])
painting.addStrokeNP(sine)

# reset to defaults
painting.reset()

# a slightly advanced option is to process the line whilst adding the stroke
# create a function and then assign that function to one of the process variables
def lineProc(points):
        
        # Calculate cumulative distance along the path
        distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
        distance = np.insert(np.cumsum(distances), 0, 0)

        sine_wave = 1 * np.sin(distance)
        
        # x
        # points[:, 0] += sine_wave
        # y
        # points[:, 1] += sine_wave
        # z
        points[:, 2] += sine_wave

        return points

painting.processLine = lineProc
painting.processRunin = lineProc
painting.addStrokeNP([(100,100), (300,300)])

# reset to defaults
painting.reset()

painting.addStroke([(100,300), (300,100)])

# the viz function allows you to preview one or more strokes in 3D
painting.viz(painting.strokes, True)

# copy gcode to clipboard
subprocess.run("pbcopy", text=True, input=painting.output("painting.gcode"))



