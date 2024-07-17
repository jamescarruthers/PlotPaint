import numpy as np
import matplotlib.pyplot as plt
import math 

polygon = [(0,0), (2,0), (2,2), (0,2), (0,0)]

polygon = [
    (0, 0),
    (2, 1),
    (1, 3),
    (3, 4),
    (4, 2),
    (2, 2),
    (0, 0)
]

sqrt3_over_2 = np.sqrt(3) / 2

polygon = [
    (1, 0),
    (0.5, sqrt3_over_2),
    (-0.5, sqrt3_over_2),
    (-1, 0),
    (-0.5, -sqrt3_over_2),
    (0.5, -sqrt3_over_2),
    (1, 0)
]

# polygon = [(0,0), (2,0), (1.5,1),(2,2), (0,2), (0,0)]



newpolygon = np.array(polygon, dtype=float)
polygon = np.array(polygon, dtype=float)

distance = 0.5

def perp(a):
    return np.array([-a[1], a[0]])

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v

def calculate_line_coefficients(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    A = y2 - y1
    B = x1 - x2
    C = x2*y1 - x1*y2
    return A, B, C

def distance_from_line(point, A, B, C):
    # Calculate the denominator and ensure it's not zero
    denominator = np.sqrt(A**2 + B**2)
    if denominator == 0:
        return float('inf')  # Return infinity if denominator is zero (line equation is invalid)
    return np.abs(A * point[0] + B * point[1] + C) / denominator

def perpendicular_projection(p, p1, p2):
    x, y = p
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return p1
    t = ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)
    return (x1 + t * dx, y1 + t * dy)

def is_within_segment(proj, p1, p2):
    x, y = proj
    x1, y1 = p1
    x2, y2 = p2
    return min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2)

def points_within_distance(points, line_points, d):
    p1, p2 = line_points
    A, B, C = calculate_line_coefficients(p1, p2)
    filtered_points = []
    for point in points:
        proj = perpendicular_projection(point, p1, p2)
        if is_within_segment(proj, p1, p2) and distance_from_line(point, A, B, C) <= d:
            filtered_points.append(point)
    return filtered_points

def perpendicular_distance(point, A, B, C):
    return abs(A * point[0] + B * point[1] + C) / np.sqrt(A**2 + B**2)

def move_point(point, A, B, C, d):
    x, y = point
    normalizer = np.sqrt(A**2 + B**2)
    dx = d * A / normalizer
    dy = d * B / normalizer
    if (A * x + B * y + C) > 0:
        return (x - dx, y - dy)
    else:
        return (x + dx, y + dy)

def viz(polygon, newpolygon):
    # Plotting the original and shrunken pentagon
    plt.figure(figsize=(8, 8))
    x, y = zip(*polygon)
    plt.plot(x, y, 'b-', marker='o', label='Original polygon')

    x, y = zip(*newpolygon)
    plt.plot(x, y, 'r-', marker='o', label='Modified polygon')

    # plt.plot([original_x[-1], original_x[0]], [original_y[-1], original_y[0]], 'b-')

    plt.legend()
    plt.axis('equal')
    plt.title('Original and shrunken polygon')
    plt.show()


def move_points(points, line_points, distance):
    p1, p2 = line_points
    A, B, C = calculate_line_coefficients(p1, p2)
    moved_points = []
    for point in points:
        moved_points.append(move_point(point, A, B, C, distance))
    return moved_points

def distance_point_to_line(px, py, x1, y1, x2, y2):
    # Calculate the distance of point (px, py) to the line (x1, y1) - (x2, y2)
    numerator = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1)
    denominator = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    return numerator / denominator

def project_point_onto_line(px, py, x1, y1, x2, y2):
    # Project point (px, py) onto the line (x1, y1) - (x2, y2)
    A = px - x1
    B = py - y1
    C = x2 - x1
    D = y2 - y1
    
    dot = A * C + B * D
    len_sq = C * C + D * D
    param = dot / len_sq
    
    xx = x1 + param * C
    yy = y1 + param * D
    
    return xx, yy

def mirror_point_with_distance(px, py, x1, y1, x2, y2, distance):
    # Project the point onto the line
    proj_x, proj_y = project_point_onto_line(px, py, x1, y1, x2, y2)
    
    # Calculate the direction vector perpendicular to the line
    dx = y2 - y1
    dy = x1 - x2
    
    # Normalize the direction vector
    length = math.sqrt(dx * dx + dy * dy)
    dx /= length
    dy /= length
    
    # Determine the sign for the new point based on the original point's position
    original_distance = distance_point_to_line(px, py, x1, y1, x2, y2)
    sign = 1 if original_distance < 0 else -1
    
    # Calculate the new mirrored point
    mirror_x = proj_x + sign * distance * dx
    mirror_y = proj_y + sign * distance * dy
    
    return [mirror_x, mirror_y]


for i in range(len(polygon) * 2):
    i = i % len(polygon)
    segment_start = polygon[i]
    segment_end = polygon[(i + 1) % len(polygon)]
    segment_vector = segment_end - segment_start
    segment_direction = normalize(segment_vector)
    perpendicular_direction = perp(segment_direction)
    perpendicular_direction = normalize(perpendicular_direction)

    A, B, C = calculate_line_coefficients(segment_start, segment_end)

    # print([polygon[i], polygon[(i + 1) % len(polygon)]])

    for i, point in enumerate(newpolygon):
        
        proj = perpendicular_projection(point, segment_start, segment_end)
        if is_within_segment(proj, segment_start, segment_end) and distance_from_line(point, A, B, C) <= distance:

            new_point = mirror_point_with_distance(point[0], point[1], segment_start[0], segment_start[1], segment_end[0],segment_end[1], distance)
            print(new_point)
            newpolygon[i] = new_point

    viz(polygon, newpolygon)

p = np.array([2,3])
p1 = np.array([0,0])
p2 = np.array([0,4])

print(perpendicular_projection(p, p1, p2))

# print(newpolygon)


