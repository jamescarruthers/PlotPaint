import numpy as np
from scipy.spatial import distance
from shapely import LineString
from shapely.affinity import scale
import math
from easing import *


def distance_between_points(P1, P2):
    """
    Calculate the Euclidean distance between two points in 3D space.

    Parameters:
    P1 (array-like): Coordinates of the first point (x1, y1, z1).
    P2 (array-like): Coordinates of the second point (x2, y2, z2).

    Returns:
    float: The Euclidean distance between P1 and P2.
    """
    P1 = np.array(P1)
    P2 = np.array(P2)
    distance = np.linalg.norm(P1 - P2)
    return distance

def process_line(line, zHeight=0, runin=20, runout=20):
    
    zRaised = 20
    
    xfact = 1
    yfact = -1
    
    if LineString(line).is_closed:
        xfact = -1
        yfact = 1
        print("possible polygon")
    
    newLine = line
    
    # yfact = 1 to work with polygons - in some way
    
    inLine = []
    if runin:
        inLine = crop_path(line, runin)
        inLine = scale(LineString(inLine), yfact = yfact, xfact = xfact, origin = inLine[0])
        inLine = list(inLine.coords)
        inLine.reverse()
        newLine = inLine[:-1] + newLine

    outLine = []
    if runout:
        outLine = crop_path(line, -runout)
        outLine = scale(LineString(outLine), yfact = yfact, xfact = xfact, origin = outLine[0])
        outLine = list(outLine.coords)
        newLine = newLine + outLine[1:]

    line2D = LineString(newLine)

    line3D = []

    # adjust the "touchdown" point
    runinOver = 0
    runoutOver = 0
    
    realrunin = runin - runinOver
    realrunout = runout - runoutOver

    runoutDist = line2D.length - realrunout
    
    dist = line2D.length - realrunin - realrunout

    for d in np.arange(0, line2D.length, 0.1):
        
        if (d <= realrunin):
            xy = line2D.interpolate(d)
            # ease out
            z = easeOutBack(d, zRaised, zHeight - zRaised, realrunin)
            line3D.append((xy.x, xy.y, z))
        
        # add post-processing for x and y (shake etc)
        # add post-processing for z (shake, slwoly dip etc)
            
        if (d > realrunin and d < runoutDist):
            xy = line2D.interpolate(d)
            z = zHeight
            # add speed based on sin function to accerate in the middle
            #z = sine(d - realrunin, dist/4, zHeight, 1)
            line3D.append((xy.x, xy.y, z))
            
        if (d >= runoutDist):
            xy = line2D.interpolate(d)
            # ease in
            z = easeInBack((d - runoutDist), zHeight, zRaised - zHeight, realrunout)
            line3D.append((xy.x, xy.y, z))


    return rdp(line3D, 0.01)


def crop_path(line, distance):
        
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


def point_at_distance(P1, P2, distance):
    """
    Calculate a point at a specific distance from P1 towards P2.

    Parameters:
    P1 (array-like): Coordinates of the first point (x1, y1, z1).
    P2 (array-like): Coordinates of the second point (x2, y2, z2).
    distance (float): Distance from P1 to the new point.

    Returns:
    np.ndarray: Coordinates of the point at the specified distance.
    """
    P1 = np.array(P1)
    P2 = np.array(P2)
    
    # Calculate the direction vector from P1 to P2
    direction = P2 - P1
    
    # Calculate the distance between P1 and P2
    total_distance = np.linalg.norm(direction)
    
    # Normalize the direction vector
    if total_distance != 0:
        direction = direction / total_distance
    
    # Calculate the new point at the specified distance
    new_point = P1 + direction * distance
    
    return new_point.tolist()

def interpolate_points(P1, P2, easing_function):
    """
    Interpolate between two points in 3D space using a specified easing function.

    Parameters:
    P1 (array-like): Coordinates of the first point (x1, y1, z1).
    P2 (array-like): Coordinates of the second point (x2, y2, z2).
    num_points (int): Number of intermediate points to generate.
    easing_function (function): Easing function to use for interpolation.

    Returns:
    np.ndarray: Array of interpolated points.
    """
    P1 = np.array(P1)
    P2 = np.array(P2)
    dimensions = len(P1)

    num_points = int(distance_between_points(P1, P2) / 0.1)
    
    interpolated_points = []
    for i in range(num_points + 1):
        t = i / num_points
        interpolated_point = []
        for d in range(dimensions):
            value = P1[d] + (P2[d] - P1[d]) * easing_function(t)
            interpolated_point.append(value)
        interpolated_points.append(interpolated_point)
    
    return np.array(interpolated_points).tolist()

def circle(center: tuple, radius: float, num_points: int) -> list:
    """
    Generate a series of coordinates for a circle.

    Parameters:
    center (tuple): A tuple representing the (x, y) coordinates of the circle's center.
    radius (float): The radius of the circle.
    num_points (int): The number of points to generate along the circle's perimeter.

    Returns:
    list: A list of tuples representing the (x, y) coordinates of the points on the circle.
    """
    circle_coords = []
    angle_step = 2 * math.pi / num_points
    
    for i in range(num_points):
        angle = i * angle_step + math.pi/2
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        circle_coords.append((x, y))
    
    circle_coords.append(circle_coords[0])
    
    return circle_coords

def point_line_distance(point, start, end):
    point = np.array(point)
    start = np.array(start)
    end = np.array(end)
    if np.all(start == end):
        return np.linalg.norm(point - start)
    else:
        return np.linalg.norm(np.cross(end - start, start - point)) / np.linalg.norm(end - start)

def rdp(points, epsilon):
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
        d = point_line_distance(points[i], start, end)
        if d > dmax:
            index = i
            dmax = d

    # If max distance is greater than epsilon, recursively simplify
    if dmax > epsilon:
        # Recursive call
        left = rdp(points[:index + 1], epsilon)
        right = rdp(points[index:], epsilon)
        # Combine results
        return np.vstack((left[:-1], right))
    else:
        return np.array([start, end])
