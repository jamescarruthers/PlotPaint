import numpy as np
import matplotlib.pyplot as plt

def expand_poly(old_points, offset, outer_ccw=1):
    old_points = np.array(old_points, dtype=float)  # Ensure input is a NumPy array with float type
    num_points = len(old_points)
    new_points = np.zeros_like(old_points)

    for curr in range(num_points):
        prev = (curr + num_points - 1) % num_points
        next = (curr + 1) % num_points

        vn = old_points[next] - old_points[curr]
        vn_norm = np.linalg.norm(vn)
        if vn_norm == 0:
            vnn = np.array([0, 0])
        else:
            vnn = vn / vn_norm
        nnnX = vnn[1]
        nnnY = -vnn[0]

        vp = old_points[curr] - old_points[prev]
        vp_norm = np.linalg.norm(vp)
        if vp_norm == 0:
            vpn = np.array([0, 0])
        else:
            vpn = vp / vp_norm
        npnX = vpn[1] * outer_ccw
        npnY = -vpn[0] * outer_ccw

        bisX = (nnnX + npnX) * outer_ccw
        bisY = (nnnY + npnY) * outer_ccw

        bis_length = np.linalg.norm([bisX, bisY])
        if bis_length != 0 and not np.isnan(bis_length):
            bisn = np.array([bisX, bisY]) / bis_length
            cosine_theta = (nnnX * npnX + nnnY * npnY)
            bislen = offset / np.sqrt((1 + cosine_theta) / 2)
            if not np.isnan(bislen):
                new_points[curr] = old_points[curr] + bislen * bisn
            else:
                new_points[curr] = old_points[curr]
        else:
            new_points[curr] = old_points[curr]

    return new_points

def signed_polygon_area(points):
    """ Calculate the signed area of a polygon given its vertices """
    x, y = points[:, 0], points[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))

def is_self_intersecting(polygon_points):
    num_points = len(polygon_points)
    for i in range(num_points):
        for j in range(i + 1, num_points):
            if abs(i - j) == 1 or (i == 0 and j == num_points - 1) or (i == num_points - 1 and j == 0):
                continue  # Skip adjacent edges and the closing edge
            p1, p2 = polygon_points[i], polygon_points[(i + 1) % num_points]
            q1, q2 = polygon_points[j], polygon_points[(j + 1) % num_points]
            if do_lines_intersect(p1, p2, q1, q2):
                return True
    return False

def do_lines_intersect(p1, p2, q1, q2):
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)

def validate_offset(polygon_points, offset, min_area=1e-6):
    old_points = np.array(polygon_points, dtype=float)
    old_area = abs(signed_polygon_area(old_points))

    new_polygon_points = expand_poly(polygon_points, offset)
    new_area = abs(signed_polygon_area(new_polygon_points))

    print(f"Old area: {old_area}, New area: {new_area}")

    # Check if the new polygon has a valid area and no overlapping points
    if new_area < min_area or np.any(np.linalg.norm(np.diff(new_polygon_points, axis=0), axis=1) < 1e-6):
        return False, new_polygon_points

    # Check if the polygon is self-intersecting
    if is_self_intersecting(new_polygon_points):
        return False, new_polygon_points

    # Check if any edge length is too small, indicating potential collapse
    for i in range(len(new_polygon_points)):
        if np.linalg.norm(new_polygon_points[i] - new_polygon_points[(i + 1) % len(new_polygon_points)]) < 1e-6:
            return False, new_polygon_points

    return True, new_polygon_points

# Define a standard pentagon as a list of tuples
pentagon = [(0, 1), (0.0, 0.309), (0.588, -0.809), (-0.588, -0.809), (-0.951, 0.309)]

# Offset to shrink the pentagon
negative_offset = 1

# Validate the negative offset
valid, shrunken_pentagon = validate_offset(pentagon, negative_offset)

if valid:
    print("The negative offset is valid.")
else:
    print("The negative offset has gone too far or caused the polygon to become invalid.")

# Plotting the original and shrunken pentagon
plt.figure(figsize=(8, 8))
original_x, original_y = zip(*pentagon)
plt.plot(original_x, original_y, 'b-', marker='o', label='Original Pentagon')
plt.plot([original_x[-1], original_x[0]], [original_y[-1], original_y[0]], 'b-')

if valid:
    shrunken_x, shrunken_y = shrunken_pentagon[:, 0], shrunken_pentagon[:, 1]
    plt.plot(shrunken_x, shrunken_y, 'g-', marker='o', label='Shrunken Pentagon')
    plt.plot(np.append(shrunken_x, shrunken_x[0]), np.append(shrunken_y, shrunken_y[0]), 'g-')
else:
    print("Resulting points: ", shrunken_pentagon)

plt.legend()
plt.axis('equal')
plt.title('Original and Shrunken Pentagon')
plt.show()

shrunken_pentagon