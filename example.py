import plotpaint
import random
import matplotlib.pyplot as plt
import math
from shapely import LineString
import subprocess 

setup = plotpaint.Plotpaint()
painting = plotpaint.Plotpaint()

for x in range(10, 190, 10):
    for y in range(10, 190, 10):
        setup.dab(x,y)
        painting.dab(x, y, random.randint(0, 360), random.randint(0,10), painting.Ease(0, 2, plotpaint.ease_in_cubic, 0, 2, plotpaint.ease_out_cubic))

# setup.output("setup.gcode")
# painting.output("painting.gcode")

line = [(1,0), (2.5,1), (10,0)]
line = plotpaint.circle((0,0), 50, 8)
# line = [(0,0), (0,10)]
# line = [(0,0), (10,0), (10,10), (0,10), (0,0)]
# line.reverse()

# line = []
# for i in range(0,10):
#     line.append((math.sin(i) * 2, i))

cropped_line = line
# cropped_line = plotpaint.extend_line(cropped_line, 20)
cropped_line = plotpaint.extend_line(cropped_line, 20)
# cropped_line = line
cropped_z = [(x, y, 0) for x, y in cropped_line]
subprocess.run("pbcopy", text=True, input=painting.gcodePoints(cropped_z))

line = LineString(line)
cropped_line = LineString(cropped_line)



fig, ax = plt.subplots()
x, y = line.xy
ax.plot(x, y, label='Original LineString', color='blue')
x, y = cropped_line.xy
ax.plot(x, y, label='Cropped LineString', color='red', linestyle='--')
ax.legend()
ax.set_title("Original and Cropped LineString")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()