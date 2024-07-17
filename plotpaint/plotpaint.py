# from plotpaint.easing import Easing
from plotpaint.npline import npline
from .draw import *

import math
import time

from simplify5d import simplify, deinterpolate
import easywaves

import numpy as np
import matplotlib.pyplot as plt

class Plotpaint:

    def __init__ (self):
        
        # strokes stores all the strokes ready to be output as gcode
        self.strokes = []
        self.ease = easywaves.npCurves
        self.reset()
        
    def reset(self):
                
        print("Reset to defaults")
        
        # default page size is A3
        self.pageWidth = 420
        self.pageHeight = 297
        
        # runIn distance extends the stroke to allow the brush to have some run up
        self.runIn = 20
        # runInEase sets the curve of the Z as it moves down - typically an "out" ease
        self.runInEase = self.ease.backOut
        # runInAdj adjusts where zDown is according to the stoke start point
        self.runInAdj = 0
        
        self.runOut = 20
        self.runOutEase = self.ease.backIn
        self.runOutAdj = 0
        
        self.pathOffset = 0
        
        self.feedEase = self.ease.sineInOut
        
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
        self.feed = 1000
        self.feedMax = 10000
        self.gcodeHeader = "G17 G21 G90 G54 M3\n"


    def addStroke(self, line):
        
        print("Interpolating... ", end="")
        start_time = time.time()

        xy = npline.interpolate(line, self.interpolate)
        
        closed = np.array_equal(line[0], line[-1])

        if closed:
            
            # roll the array if offset is set
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
        
            runIn = npline.rotate(runIn, 180, xy[0])
            runIn = runIn[::-1]
        
            runOut = npline.rotate(runOut, -180, xy[-1])
            runOut = runOut[::-1]

        
        z = np.linspace(0, 1, len(runIn))
        z = self.runInEase(z)
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
        z = self.runOutEase(z)
        z = self.zDown + (self.zUp - self.zDown) * z
        runOut = np.column_stack((runOut, z))

        if self.processRunout:
            runOut = self.processRunin(runOut)


        xyz = xy
        if self.runIn != 0:
            xyz = np.concatenate((runIn, xyz), axis=0)
        if self.runIn != 0:
            xyz = np.concatenate((xyz, runOut), axis=0)


        end_time = time.time()
        print(str(int((end_time - start_time) * 1000)) + " ms")

        print("Simplifying... ")
        start_time = time.time()
        
        # simpleLine = xyz
        
        # simpleLine = self.remove_duplicate_segments(xyz)
        print(f'Simplified from {len(xyz)} to ', end="")
        simpleLine = simplify(xyz, self.simplifyVal, True)
        print(f'{len(simpleLine)} points')
        # simpleLine = np.array(simpleLine)

        # simpleLine[:, 1] = self.pageHeight - simpleLine[:, 1]
        end_time = time.time()
        print(str(int((end_time - start_time) * 1000)) + " ms")
        
        # add the feed speed
        f = np.linspace(0, 2, len(xyz))
        f = self.feedEase(f)
        f = self.feed + (self.feedMax - self.feed) * f
        fLine = np.column_stack((xyz, f))
        matches = np.isin(fLine[:, :3], simpleLine).all(axis=1)
        simpleLine = np.hstack((fLine[matches, :3], fLine[matches, 3][:, np.newaxis]))
        
        self.strokes.append(simpleLine)


        return self.strokes[-1]


    def viz(self, lines, page=False):
        
        print("Visualising")
        
        if isinstance(lines[0][0], (int, float)):
            lines = [lines]
        
        fig = plt.figure()
        fig.set_figwidth(16)
        fig.set_figheight(10)
        
        ax = fig.add_subplot(111, projection='3d')

        if page:
            pageLines = [[0,0,0],[0,self.pageHeight,0],[self.pageWidth,self.pageHeight,0],[self.pageWidth,0,0],[0,0,0]]
            x, y, z = zip(*pageLines)
            ax.plot(x, y, z, clip_on=False)

        for coords in lines:
            # make 3D if 2D
            if len(coords[0]) == 2:
                coords = [(x, y, 0) for x, y in coords]
            x, y, z, f = zip(*coords)
            ax.plot(x, y, z, marker='.', clip_on=False)
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
        # ax.grid(True)
        ax.set_xlim([0, self.pageWidth])
        ax.set_ylim([0, self.pageHeight])
        ax.set_zlim(bottom=0)
        ax.set_aspect('equal')
        ax.set_adjustable('datalim')
        ax.set_clip_box(None)
        ax.view_init(60, -90) 
        ax.margins(0, tight=True)
        plt.gca().invert_yaxis()
        # plt.gca().set_ylim = self.pageHeight
        # plt.gca().set_xlim = self.pageWidth
        plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, wspace=0, hspace=0)
        
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
            output += f"G01 X{point[0]:.2f} Y{self.pageHeight-point[1]:.2f} Z{point[2]:.2f} F{point[3]:.0f}\n"
            if self.outside_bounds(point):
                warning = True
        
        output += f"G00 Z{self.zUp}\n"
        
        if warning:
            print("Plot may be off the page, contains negative coordinates")
            
        return output

    def outside_bounds(self, coord):
        
        if coord[0] < 0 or coord[0] > self.pageWidth or coord[1] < 0 or coord[1] > self.pageHeight or coord[2] < 0:
                return True
        return False
