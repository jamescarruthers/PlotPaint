import numpy as np
from shapely import LineString
from shapely.affinity import scale
import math


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

def extend_line(line, distance, forceMode="detect"):
    
    if forceMode == "detect":
        if (LineString(line).is_closed):
            mode = "poly"
        else:
            mode = "line"
    else:
        mode = forceMode
    
    extend = crop_path(line, distance)
        
    # if distance < 0:
    #     extend.reverse()

    if mode == "line":
        extend = scale(LineString(extend), yfact= -1, xfact= -1, origin = line[0])
    elif mode == "poly":
        extend = scale(LineString(extend), yfact= -1, origin = line[0])


    extend = list(extend.coords)
    
    if distance < 0:
        line.reverse()
        return line + extend[1:]
    else:
        extend.reverse()
        return extend[:-1] + line




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

    print("cropped")
    print(cropped_coords)
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
