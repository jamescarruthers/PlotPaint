from shapely import LineString
from shapely.affinity import scale
import numpy as np
from easing import Easing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

class Plotpaint:

    def __init__ (self):
        self.strokes = []
        self.ease = Easing()
        self.reset()
        
    def reset(self):
        # Do not reset self.strokes
        self.feed = 5000
        self.zUp = 20
        self.zDown = 5
        self.runInEase = self.ease.backOut
        self.runOutEase = self.ease.backIn
        self.runIn = 20
        self.runOut = 20
        self.runInAdj = 0
        self.runOutAdj = 0
        self.xFact = -1
        self.yFact = -1
        self.processRunin = None
        self.processLine = None
        self.processRunout = None

    def addStroke(self, line):
        
        inv = 0
        
        # if it's polygon then we probably want to mirror the run-in and run-out the opposite way
        if LineString(line).is_closed:
            inv = -1
        else:
            inv = 1

        newLine = line

        inLine = []
        if self.runIn:
            inLine = self.cropPath(line, self.runIn)
            inLine = scale(LineString(inLine), yfact = self.yFact * inv, xfact = self.xFact * inv, origin = inLine[0])
            inLine = list(inLine.coords)
            inLine.reverse()
            newLine = inLine[:-1] + newLine

        outLine = []
        if self.runOut:
            outLine = self.cropPath(line, -self.runOut)
            outLine = scale(LineString(outLine), yfact = self.yFact * inv, xfact = self.xFact * inv, origin = outLine[0])
            outLine = list(outLine.coords)
            newLine = newLine + outLine[1:]

        line2D = LineString(newLine)

        runinDist = self.runIn - self.runInAdj
        runoutLen = self.runOut - self.runOutAdj
        runoutDist = line2D.length - runoutLen

        line3D = []

        for d in np.arange(0, line2D.length, 0.1):
            
            if (d <= runinDist):
                xy = line2D.interpolate(d)
                z = self.runInEase(d, self.zUp, self.zDown - self.zUp, runinDist)
                
                if self.processRunin != None:
                    x, y, z = self.processRunin(d, xy.x, xy.y, z)
                    line3D.append((x, y, z))
                else:
                    line3D.append((xy.x, xy.y, z))
                
            if (d > runinDist and d < runoutDist):
                xy = line2D.interpolate(d)
                z = self.zDown
                if self.processLine != None:
                    x, y, z = self.processLine(d, xy.x, xy.y, z)
                    line3D.append((x, y, z))
                else:
                    line3D.append((xy.x, xy.y, z))
                
            if (d >= runoutDist):
                xy = line2D.interpolate(d)
                z = self.runOutEase((d - runoutDist), self.zDown, self.zUp - self.zDown, runoutLen)
                if self.processRunout != None:
                    x, y, z = self.processRunout(self, d, xy.x, xy.y, z)
                    line3D.append((x, y, z))
                else:
                    line3D.append((xy.x, xy.y, z))


        self.strokes.append(self.simplify(line3D, 0.01))

    def cropPath(self, line, distance):
            
        if distance < 0:
            distance *= -1
            line.reverse()
            
        line = LineString(line)
        
        if distance >= line.length:
            return line
        
        coords = list(line.coords)
        cumulative_length = 0.0
        cropped_coords = []
        
        cropped_coords.append(coords[0])

        for pair in zip(coords[:-1], coords[1:]):
            
            segment_length = LineString(pair).length
            
            if cumulative_length + segment_length >= distance:
                cropped_coords.append(line.interpolate(distance).coords[0])
                break
            else:
                cropped_coords.append(pair[1])
                cumulative_length += segment_length

        return cropped_coords

    def simplify(self, points, epsilon):
        """
        Simplify a 3D line using the Ramer-Douglas-Peucker algorithm.

        :param points: A numpy array of shape (n, 3) representing the 3D points.
        :param epsilon: The maximum distance of a point to the line for it to be kept.
        :return: A numpy array of the simplified points.
        """
        if len(points) < 3:
            return points

        # Find the point with the maximum distance
        start, end = np.array(points[0]), np.array(points[-1])
        dmax = 0
        index = -1
        for i in range(1, len(points) - 1):
            d = self.point_line_distance(points[i], start, end)
            if d > dmax:
                index = i
                dmax = d

        # If max distance is greater than epsilon, recursively simplify
        if dmax > epsilon:
            # Recursive call
            left = self.simplify(points[:index + 1], epsilon)
            right = self.simplify(points[index:], epsilon)
            # Combine results
            return np.vstack((left[:-1], right))
        else:
            return np.array([start, end])

    def point_line_distance(self, point, start, end):
        point = np.array(point)
        start = np.array(start)
        end = np.array(end)
        if np.all(start == end):
            return np.linalg.norm(point - start)
        else:
            return np.linalg.norm(np.cross(end - start, start - point)) / np.linalg.norm(end - start)

    def viz(self, lines):
        
        if isinstance(lines[0][0], (int, float)):
            lines = [lines]
        
        fig = plt.figure()
        fig.set_figheight(10)
        fig.set_figwidth(14)
        
        ax = fig.add_subplot(111, projection='3d')
        for coords in lines:
            x, y, z = zip(*coords)
            ax.plot(x, y, z)
            
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.grid(True)
        ax.set_zlim(bottom=0)
        ax.set_aspect('equal', adjustable='box')
        plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        plt.show()
        
        
    def output(self, filename):

        output = "G17 G21 G90 G54 M3\n"

        for stroke in self.strokes:
            output += self.gcodePoints(stroke)                
        f = open(filename, "w+")
        f.write(output)
        f.close()
        
        return output


    def gcodePoints(self, points):
        
        output = ""
        
        output += f"G00 X{points[0][0]:.2f} Y{points[0][1]:.2f} Z{self.zUp}\n"
        # output += "G01 Z0\n"
        for point in points:
            output += f"G01 X{point[0]:.2f} Y{point[1]:.2f} Z{point[2]:.2f} F{self.feed}\n"
        output += f"G00 Z{self.zUp}\n"
        
        return output


    def circle(self, center: tuple, radius: float, num_points: int) -> list:

        circle_coords = []
        angle_step = 2 * math.pi / num_points
        
        for i in range(num_points):
            angle = i * angle_step + math.pi/2
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            circle_coords.append((x, y))
        
        circle_coords.append(circle_coords[0])
        
        return circle_coords