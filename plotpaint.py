from dataclasses import dataclass
from typing import Callable
import math
import numpy as np

import shapely

from easing import *
from util import *


class Plotpaint:
    
    @dataclass
    class Dab():
        x: float
        y: float
        angle: float = 0
        distance: float = 0

    @dataclass
    class Ease():
        runin: float = 0
        easeindist: float = 0
        easein: Callable[[], None] = linear
        runout: float = 0
        easeoutdist: float = 0
        easeout: Callable[[], None] = linear

    
    def __init__ (self):
        self.strokes = []
        self.feed = 5000
        self.zUp = 20
        self.zDown = 5
        
    def dab(self, x, y, angle=0, distance=0, ease=None):
        self.strokes.append([self.Dab(x, y, angle, distance), ease])        
            
        
    def output(self, filename):

        output = "G17 G21 G90 G54 M3\n"

        for stroke in self.strokes:
            if isinstance(stroke[0], self.Dab):
                output += self.processDab(stroke)
                
        f = open(filename, "w+")
        f.write(output)
        f.close()
        
    def processDab(self, dab):
        
        output = ""
        
        if (dab[0].distance == 0):
            points = []
            points.append([dab[0].x, dab[0].y, 0])
            output += self.gcodePoints(points)
        else:
            angle = math.radians(dab[0].angle) - (math.pi / 2)
            
            x0 = dab[0].x + math.cos(angle) * (dab[0].distance / 2)
            y0 = dab[0].y + math.sin(angle) * (dab[0].distance / 2)
            x1 = dab[0].x + math.cos(angle + math.pi) * (dab[0].distance / 2)
            y1 = dab[0].y + math.sin(angle + math.pi) * (dab[0].distance / 2)
            
            points = []
            points.append([x0, y0, 0])
            points.append([x1, y1, 0])
            
            if isinstance(dab[1], self.Ease):
                
                start = interpolate_points(points[0], point_at_distance(points[0], points[-1], dab[1].easeindist), dab[1].easein)
                end = interpolate_points(point_at_distance(points[-1], points[0], dab[1].easeoutdist), points[-1], dab[1].easeout)

                points = list(start) + points[1:]
                points = points[:len(points)-1] + list(end)

            output += self.gcodePoints(points)

        return output

    def gcodePoints(self, points):
        
        output = ""
        
        output += f"G00 X{points[0][0]:.2f} Y{points[0][1]:.2f} Z{self.zUp}\n"
        # output += "G01 Z0\n"
        for point in points:
            output += f"G01 X{point[0]:.2f} Y{point[1]:.2f} Z{point[2]:.2f} F{self.feed}\n"
        output += f"G00 Z{self.zUp}\n"
        
        return output


        
