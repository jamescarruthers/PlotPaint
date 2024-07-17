import numpy as np
import matplotlib.pyplot as plt

polygon = [(0,0), (2,0), (1.5,1),(2,2), (0,2), (0,0)]

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
    denominator = np.sqrt(A**2 + B**2)
    if denominator == 0:
        return float('inf')
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

def move_point_to_distance(point, proj, distance):
    direction = np.array(point) - np.array(proj)
    direction = normalize(direction)
    return proj + direction * distance

def viz(polygon, newpolygon):
    plt.figure(figsize=(8, 8))
    x, y = zip(*polygon)
    plt.plot(x, y, 'b-', marker='o', label='Original polygon')

    x, y = zip(*newpolygon)
    plt.plot(x, y, 'r-', marker='o', label='Modified polygon')

    plt.legend()
    plt.axis('equal')
    plt.title('Original and modified polygon')
    plt.show()

for i in range(len(polygon)):
    segment_start = polygon[i]
    segment_end = polygon[(i + 1) % len(polygon)]
    A, B, C = calculate_line_coefficients(segment_start, segment_end)

    for j, point in enumerate(newpolygon):
        proj = perpendicular_projection(point, segment_start, segment_end)
        if is_within_segment(proj, segment_start, segment_end) and distance_from_line(point, A, B, C) <= distance:
            new_point = move_point_to_distance(point, proj, distance)
            newpolygon[j] = new_point
            print(new_point)

viz(polygon, newpolygon)

print(newpolygon)