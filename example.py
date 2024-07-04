import plotpaint
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from shapely import LineString
import subprocess 
import numpy as np

setup = plotpaint.Plotpaint()
painting = plotpaint.Plotpaint()

# for x in range(10, 190, 10):
#     for y in range(10, 190, 10):
#         setup.dab(x,y)
#         painting.dab(x, y, random.randint(0, 360), random.randint(0,10), painting.Ease(0, 2, plotpaint.ease_in_cubic, 0, 2, plotpaint.ease_out_cubic))

# setup.output("setup.gcode")
# painting.output("painting.gcode")

# line = [(1,0), (2.5,1), (10,0)]
line = plotpaint.circle((200,200), 150, 128)
# line = [(0,0), (0,10)]
# # line = [(0,0), (10,0), (10,10), (0,10), (0,0)]
# # line.reverse()

# line = []
# for i in np.arange(0,30, 0.1):
#     line.append((math.sin(i) * 2, i))

cropped_line = line
# cropped_line = plotpaint.extend_line(cropped_line, 20)
cropped_line = plotpaint.process_line(cropped_line, 5, 50, 50)
# cropped_line = line
subprocess.run("pbcopy", text=True, input=painting.gcodePoints(cropped_line))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x, y, z = zip(*cropped_line)
ax.plot(x, y, z, color='blue')
ax.set_title("Path Preview")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
ax.set_zlim(bottom=0)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()