from plotpaint.easing import Easing

import math
import traceback
import re
import time

from shapely import LineString
from shapely.affinity import scale, rotate

from simplify5d import simplify

import numpy as np
import matplotlib.pyplot as plt

class Plotpaint:

    def __init__ (self):
        
        # strokes stores all the strokes ready to be output as gcode
        self.strokes = []
        self.ease = Easing()
        self.reset()
        
    def reset(self):
                
        print("Reset to defaults")
        
        # default page size is A3
        self.pageWidth = 420
        self.pageHeight = 297
        
        # runIn distance extends the stroke to allow the brush to have some run up
        self.runIn = 20
        # runInEase sets the curve of the Z as it moves down
        self.runInEase = self.ease.backOut
        # runInAdj adjusts where zDown is according to the stoke start point
        self.runInAdj = 0
        
        self.runOut = 20
        self.runOutEase = self.ease.backIn
        self.runOutAdj = 0
        
        self.pathOffset = 0
        
        # if path generation takes too long then you can make these values larger
        # how finely to interpolate a path (in mm)
        self.interpolate = 0.05
        # how much to simplify paths (tolerance in mm)
        self.simplifyVal = 0.02
        
        # assign a function to process the run-in, line or run-out
        # will be more useful once previous coordinates are available
        self.processRunin = None
        self.processLine = None
        self.processRunout = None
        
        # machine variables
        self.zUp = 20 #mm
        self.zDown = 5 #mm
        self.feed = 5000
        self.gcodeHeader = "G17 G21 G90 G54 M3\n"


    def addStrokeNP(self, line):
        
        print("Interpolating... ", end="")
        start_time = time.time()
        
        line = np.array(line)
        x_points = line[:, 0]
        y_points = line[:, 1]

        # Calculate cumulative distance along the path
        distances = np.sqrt(np.diff(x_points)**2 + np.diff(y_points)**2)
        cumulative_distances = np.insert(np.cumsum(distances), 0, 0)

        # Calculate the number of interpolation points
        num_points = int(cumulative_distances[-1] / self.interpolate)

        # Ensure original points are part of the interpolated distances
        interp_distances = np.linspace(0, cumulative_distances[-1], num_points - len(line) + 1)
        interp_distances = np.unique(np.concatenate((cumulative_distances, interp_distances)))

        # Interpolate the x and y coordinates separately
        x = np.interp(interp_distances, cumulative_distances, x_points)
        y = np.interp(interp_distances, cumulative_distances, y_points)
        xy = np.column_stack((x,y))
        
        closed = np.array_equal(line[0], line[-1])

        if closed:
            
            # roll the array if the offset is set
            if self.pathOffset:
                num_rows = xy.shape[0]
                shift_amount = int(self.pathOffset * num_rows)
                xy = np.roll(xy, shift_amount, axis=0)

            # run in is the end part of the path, and run out is the first part of the path
            runIn = xy[-int(self.runOut / self.interpolate):-1]
            runOut = xy[1:int(self.runIn / self.interpolate)]
        else:
            runIn = xy[1:int(self.runIn / self.interpolate)]
            runOut = xy[-int(self.runOut / self.interpolate):-1]
        
            runIn = self.rotate_path(runIn, 180, xy[0])
            runIn = runIn[::-1]
        
            runOut = self.rotate_path(runOut, -180, xy[-1])
            runOut = runOut[::-1]

        
        z = np.linspace(0, 1, len(runIn))
        z = self.ease.circOutNP(z)
        z = self.zUp + (self.zDown - self.zUp) * z
        runIn = np.column_stack((runIn, z))
        
        if self.processRunin:
            runIn = self.processRunin(runIn)

        xy = xy if self.processLine or self.pathOffset else line

        z = np.full(len(xy), self.zDown)
        xy = np.column_stack((xy, z))
        
        if self.processLine:
            xy = self.processLine(xy)

        z = np.linspace(0, 1, len(runOut))
        z = self.ease.circInNP(z)
        z = self.zDown + (self.zUp - self.zDown) * z
        runOut = np.column_stack((runOut, z))

        if self.processRunout:
            runOut = self.processRunin(runOut)

        xyz = np.concatenate((runIn,xy,runOut), axis=0)

        end_time = time.time()
        print(str(int((end_time - start_time) * 1000)) + " ms")

        print("Simplifying... ", end="")
        start_time = time.time()
        
        # simpleLine = xyz
        
        simpleLine = self.remove_duplicate_segments(xyz)
        simpleLine = simplify(simpleLine, self.simplifyVal, True)

        # simpleLine = np.array(simpleLine)

        # simpleLine[:, 1] = self.pageHeight - simpleLine[:, 1]
        
        self.strokes.append(simpleLine)
        end_time = time.time()
        print(str(int((end_time - start_time) * 1000)) + " ms")


        return self.strokes[-1]


    def addStroke(self, line):
        
        stack = traceback.extract_stack()
        name = re.findall(r'\((.*?)\)', stack[-2][3])[0]
        name = re.sub(r'[\[\(]', '', name)
        print("Adding stroke: " + name)
        
        # Set xfact and yfact based on whether the line is closed
        xfact, yfact = (-1, 1) if LineString(line).is_closed else (1, 1)

        newLine = line
        
        # Process the run-in
        if self.runIn:
            inLine = self.cropPath(line, self.runIn)
            inLine = scale(rotate(LineString(inLine), 180, inLine[0]), yfact=yfact, xfact=xfact, origin=inLine[0])
            inLine = list(inLine.coords)[::-1]
            newLine = inLine[:-1] + newLine

        # Process the run-out
        if self.runOut:
            outLine = self.cropPath(line, -self.runOut)
            outLine = scale(rotate(LineString(outLine), 180, outLine[0]), yfact=yfact, xfact=xfact, origin=outLine[0])
            outLine = list(outLine.coords)
            newLine = newLine + outLine[1:]

        line2D = LineString(newLine)

        runinDist = self.runIn - self.runInAdj
        runoutLen = self.runOut - self.runOutAdj
        runoutDist = line2D.length - runoutLen

        line3D = []

        print("Interpolating... ", end="")
        start_time = time.time()

        for d in np.arange(0, line2D.length, self.interpolate):
            xy = line2D.interpolate(d)
            if d <= runinDist:
                z = self.runInEase(d, self.zUp, self.zDown - self.zUp, runinDist)
                if self.processRunin:
                    x, y, z = self.processRunin(d, xy.x, xy.y, z)
                line3D.append((xy.x, xy.y, z))
            elif d < runoutDist:
                z = self.zDown
                if self.processLine:
                    x, y, z = self.processLine(d, xy.x, xy.y, z)
                line3D.append((xy.x, xy.y, z))
            else:
                z = self.runOutEase(d - runoutDist, self.zDown, self.zUp - self.zDown, runoutLen)
                if self.processRunout:
                    x, y, z = self.processRunout(d, xy.x, xy.y, z)
                line3D.append((xy.x, xy.y, z))


        end_time = time.time()
        print(str(int((end_time - start_time) * 1000)) + " ms")

        print("Simplifying... ", end="")
        start_time = time.time()
        
        simpleLine = simplify(line3D, self.simplifyVal, True)
        
        # simpleLine = np.array(simpleLine)
        # simpleLine[:, 1] = self.pageHeight - simpleLine[:, 1]
        
        self.strokes.append(simpleLine)
        end_time = time.time()
        print(str(int((end_time - start_time) * 1000)) + " ms")

        return self.strokes[-1]

    def cropPath(self, line, distance):
            
        if distance < 0:
            distance *= -1
            line.reverse()
            
        line = LineString(line)
        
        if distance >= line.length:
            return line
        
        coords = list(line.coords)
        cumulative_length = 0.0
        cropped_coords = [coords[0]]
        
        for pair in zip(coords[:-1], coords[1:]):
            
            segment_length = LineString(pair).length
            
            if cumulative_length + segment_length >= distance:
                cropped_coords.append(line.interpolate(distance).coords[0])
                break
            cropped_coords.append(pair[1])
            cumulative_length += segment_length

        return cropped_coords

    def viz(self, lines, page=False):
        
        print("Visualising")
        
        if isinstance(lines[0][0], (int, float)):
            lines = [lines]
        
        fig = plt.figure()
        fig.set_figheight(10)
        fig.set_figwidth(14)
        
        ax = fig.add_subplot(111, projection='3d')

        if page:
            pageLines = [[0,0,0],[0,self.pageHeight,0],[self.pageWidth,self.pageHeight,0],[self.pageWidth,0,0],[0,0,0]]
            x, y, z = zip(*pageLines)
            ax.plot(x, y, z)

        for coords in lines:
            # make 3D if 2D
            if len(coords[0]) == 2:
                coords = [(x, y, 0) for x, y in coords]
            x, y, z = zip(*coords)
            ax.plot(x, y, z)
            start_x = x[0]
            start_y = y[0]
            start_z = z[0]
            end_x = x[-1]
            end_y = y[-1]
            end_z = z[-1]
            #ax.quiver(first_x, first_y, first_z, 1, 1, 1, length=2, color='red', arrow_length_ratio=0.5)
            ax.text(start_x, start_y, start_z, "S", color='red')
            ax.text(end_x, end_y, end_z, "E", color='red')



        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.grid(True)
        ax.set_zlim(bottom=0)
        ax.set_aspect('equal', adjustable='box')
        plt.gca().invert_yaxis()
        # plt.gca().set_ylim = self.pageHeight
        # plt.gca().set_xlim = self.pageWidth
        plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        plt.show()
        
        
    def output(self, filename):

        print("Generating gcode")

        output = self.gcodeHeader

        for stroke in self.strokes:
            output += self.gcodePoints(stroke)                
        f = open(filename, "w+")
        f.write(output)
        f.close()
        
        return output


    def gcodePoints(self, points):
        
        output = ""
        warning = False
        
        output += f"G00 X{points[0][0]:.2f} Y{self.pageHeight-points[0][1]:.2f} Z{self.zUp}\n"
        if self.outside_bounds(points[0]):
            warning = True

        for point in points:
            output += f"G01 X{point[0]:.2f} Y{self.pageHeight-point[1]:.2f} Z{point[2]:.2f} F{self.feed}\n"
            if self.outside_bounds(point):
                warning = True
        
        output += f"G00 Z{self.zUp}\n"
        
        if warning:
            print("Plot may be off the page, contains negative coordinates")
            
        return output


    def poly(self, center: tuple, radius: float, startAngle: float, num_points: int) -> list:

        poly_coords = []
        angle_step = 2 * math.pi / num_points
        startAngle = startAngle * (math.pi / 180)
        
        for i in range(num_points):
            angle = i * angle_step + startAngle
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            poly_coords.append((x, y))
        
        poly_coords.append(poly_coords[0])
        
        return poly_coords

    def outside_bounds(self, coord):
        
        if coord[0] < 0 or coord[0] > self.pageWidth or coord[1] < 0 or coord[1] > self.pageHeight or coord[2] < 0:
                return True
        return False

    def normalize_vector(self, v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def remove_duplicate_segments(self, points, tolerance=1e-5):
        if len(points) < 2:
            return points

        # Calculate vectors for each segment
        vectors = np.diff(points, axis=0)

        # Normalize vectors
        normalized_vectors = np.array([self.normalize_vector(v) for v in vectors])

        # Filter out points that create duplicate segments
        filtered_points = [points[0]]
        for i in range(1, len(points)):
            if i == 1 or np.linalg.norm(normalized_vectors[i-1] - normalized_vectors[i-2]) > tolerance:
                filtered_points.append(points[i])

        return np.array(filtered_points)

    def rotate_path(self, points, angle, origin=[0,0]):
        
            angle = np.radians(angle)
            rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
            ])

            translated = points - origin
            rotated = translated.dot(rotation_matrix)
            output = rotated + origin
            
            return output
