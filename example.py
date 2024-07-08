import plotpaint
import math
import subprocess 

painting = plotpaint.Plotpaint()

# generate some coordinates
circle = painting.circle((200,200), 150, 128)
# add the circle to the painting, using default settings
painting.addStroke(circle)
# the addStroke function returns the stroke in the strokes array

# change the defaults (see the class init) and add a line
painting.runInEase = painting.ease.circOut
painting.runOutEase = painting.ease.circIn
painting.runIn = 50
painting.runOut = 50
painting.addStroke([(100, 300), (300,100)])

# reset to defaults
painting.reset()

# a slightly advanced option is to process the line whilst adding the stroke
# create a function and then assign that function to one of the process variables
def lineProc(d, x, y, z):
    z += math.sin(d/4) * 0.5
    return [x,y,z]
painting.processLine = lineProc
painting.addStroke([(100,100), (300,300)])

# copy gcode to clipboard
# subprocess.run("pbcopy", text=True, input=painting.output("painting.gcode"))

# the viz function allows you to preview one or more strokes in 3D
painting.viz([painting.strokes[0], painting.strokes[1], painting.strokes[2]])


