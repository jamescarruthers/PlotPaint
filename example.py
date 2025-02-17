import plotpaint
import math
import subprocess 
import numpy as np
import plotpaint.npline
from plotpaint.draw import *

painting = plotpaint.Plotpaint()

# painting.pathOffset = 0.25
# painting.addStroke(poly((100,100), 50, 90, 6))
# painting.addStroke(poly((painting.pageWidth-25,painting.pageHeight-25), 10, 0, 36))

# pentagon = poly((200,200), 50, 90, 5)
# print(plotpaint.npline.polygon_area(pentagon))
# painting.addStroke(pentagon)
# print("concfill")
# newpentagon = plotpaint.npline.offset(pentagon, -20)
# newpentagon = concfill(newpentagon, 5)

# painting.addStroke(newpentagon)

# painting.addStroke(line((100,100), 100, 45))

# # generate some coordinates
# circle = poly((150,150), 100, 0, 36)
# # add the circle to the painting, using default settings
# painting.addStroke(circle)
# # the addStroke function returns the stroke in the strokes array

# circle = poly((150,150), 75, 180, 128)
# circle.reverse()
# painting.addStroke(circle)

# # change the defaults (see the class init) and add a line
# painting.runInEase = painting.ease.circOut
# painting.runOutEase = painting.ease.circIn
# painting.runIn = 0
# painting.runOut = 0

# sine = []
# for x in range(150,200):
#         sine.append([100 + math.sin(x/math.pi) * 20, 100+x])
# painting.addStroke(sine)

# sine = []
# for x in range(150,200):
#         sine.append([x + 100, 100+math.sin(x/math.pi) * 20])
# painting.addStroke(sine)

# # reset to defaults
# painting.reset()

# # a slightly advanced option is to process the line whilst adding the stroke
# # create a function and then assign that function to one of the process variables
# def lineProc(points):
        
#         # Calculate cumulative distance along the path
#         distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
#         distance = np.insert(np.cumsum(distances), 0, 0)

#         sine_wave = 1 * np.sin(distance)
        
#         # x
#         # points[:, 0] += sine_wave
#         # y
#         # points[:, 1] += sine_wave
#         # z
#         points[:, 2] += sine_wave

#         return points

# painting.processLine = lineProc
# painting.processRunin = lineProc
# painting.addStroke([(100,100), (300,300)])

# # reset to defaults
# painting.reset()

# painting.addStroke([(100,300), (300,100)])
painting.runIn = 0
painting.runOut = 0

circle = poly((200,200), 100, 0, 8)
paths = plotpaint.npline.split_path(circle, 500)
for path in paths:
        painting.addStroke(path)

painting.runIn = 0
painting.runOut = 0

points = []
for path in paths:
        painting.addStroke((poly(path[0], 10, 0, 4)))
        

        

print(len(paths))

# the viz function allows you to preview one or more strokes in 3D
painting.viz(painting.strokes, True)

# copy gcode to clipboard
subprocess.run("pbcopy", text=True, input=painting.output("painting.gcode"))



