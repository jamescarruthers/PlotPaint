import numpy as np
from shapely.geometry import LineString, Polygon

class npline:
    
    def distance(coord1, coord2):
        return np.linalg.norm(np.array(coord1) - np.array(coord2))

    def reducepoints(coords, tolerance):
        if len(coords) == 0:
            return []

        print(f'Reducing {len(coords)} to ', end="")

        filtered_coords = [coords[0]]

        for i in range(1, len(coords)):
            distance = npline.distance(filtered_coords[-1], coords[i])
            if distance >= tolerance:
                filtered_coords.append(coords[i])

        print(f'{len(filtered_coords)} points')

        return filtered_coords

    # rotate in 2D
    @staticmethod
    def rotate(points, angle, origin=[0,0]):
            
            angle = np.radians(angle)
            matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
            ])

            translated = points - origin
            rotated = translated.dot(matrix)
            output = rotated + origin
            
            return output

    # 2D and 3D scaling
    # factors = [2,2] or [2,2,2] to scale * 2
    @staticmethod
    def scale(points, factors, origin=[0,0]):
        
            origin = [0,0,0] if points.shape[1] and origin == [0,0] else [0,0]
        
            matrix = np.diag(factors)

            translated = points - origin
            product = translated.dot(matrix)
            output = product + origin
            
            return output
        
    @staticmethod
    def interpolate(points, resolution):
        
        points = np.array(points)
        
        x_points = points[:, 0]
        y_points = points[:, 1]

        # Calculate cumulative distance along the path
        distances = np.sqrt(np.diff(x_points)**2 + np.diff(y_points)**2)
        cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
        
        # Calculate the number of interpolation points
        num_points = int(cumulative_distances[-1] / resolution)

        # Ensure original points are part of the interpolated distances
        interp_distances = np.linspace(0, cumulative_distances[-1], num_points - len(points) + 1)
        interp_distances = np.unique(np.concatenate((cumulative_distances, interp_distances)))

        # Interpolate the x and y coordinates separately
        x = np.interp(interp_distances, cumulative_distances, x_points)
        y = np.interp(interp_distances, cumulative_distances, y_points)
        
        return np.column_stack((x,y))

    @staticmethod
    def offset(old_points, offset, outer_ccw=1):
        old_points = old_points[:-1]
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

        new_points = np.concatenate((new_points, [new_points[0]]))

        return new_points

    def spiral(polygon, offset):
                
        spiral = polygon
        current_offset = offset
        
        while True:
            
            offset_polygon = npline.offset(polygon, current_offset)
            
            if npline.is_self_intersecting(offset_polygon):
                break
            
            spiral.append(offset_polygon)
            current_offset += offset
            
        return spiral
    
    @staticmethod
    def polygon_area(points):
        points = np.array(points)
        """ Calculate the signed area of a polygon given its vertices """
        x, y = points[:, 0], points[:, 1]
        return 0.5 * np.sum(x * np.roll(y, 1) - y * np.roll(x, 1))
    
    
    @staticmethod
    def do_lines_intersect(p1, p2, q1, q2):
        def orientation(a, b, c):
            val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
            if np.isclose(val, 0):
                return 0
            return 1 if val > 0 else 2

        def on_segment(a, b, c):
            if min(a[0], c[0]) <= b[0] <= max(a[0], c[0]) and min(a[1], c[1]) <= b[1] <= max(a[1], c[1]):
                return True
            return False

        o1 = orientation(p1, p2, q1)
        o2 = orientation(p1, p2, q2)
        o3 = orientation(q1, q2, p1)
        o4 = orientation(q1, q2, p2)

        if o1 != o2 and o3 != o4:
            return True

        if o1 == 0 and on_segment(p1, q1, p2):
            return True

        if o2 == 0 and on_segment(p1, q2, p2):
            return True

        if o3 == 0 and on_segment(q1, p1, q2):
            return True

        if o4 == 0 and on_segment(q1, p2, q2):
            return True

        return False

    @staticmethod
    def is_self_intersecting(polygon):
        # Convert the list of points into a Shapely Polygon object
        poly = Polygon(polygon)
        # Check if the polygon is simple (i.e., not self-intersecting)
        return not poly.is_simple

    @staticmethod
    def validate_offset(polygon_points, offset):
        new_polygon_points = npline.offset(polygon_points, offset)
        if npline.is_self_intersecting(new_polygon_points):
            return False, new_polygon_points
        return True, new_polygon_points


    @staticmethod
    def do_lines_intersect(p1, p2, q1, q2):
        """ Check if line segment p1-p2 intersects with line segment q1-q2 """
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)

    @staticmethod
    def is_self_intersecting(polygon_points):
        num_points = len(polygon_points)
        for i in range(num_points):
            for j in range(i + 1, num_points):
                if i == j or (i + 1) % num_points == j or i == (j + 1) % num_points:
                    continue  # Skip adjacent edges
                p1, p2 = polygon_points[i], polygon_points[(i + 1) % num_points]
                q1, q2 = polygon_points[j], polygon_points[(j + 1) % num_points]
                if npline.do_lines_intersect(p1, p2, q1, q2):
                    return True
        return False

    @staticmethod
    def is_simple_polygon(vertices):
        """
        Determines if a polygon is simple (i.e., it does not intersect itself).

        Parameters:
        vertices (numpy.ndarray): An Nx2 array of polygon vertices.

        Returns:
        bool: True if the polygon is simple, False otherwise.
        """
        def do_lines_intersect(p1, p2, q1, q2):
            """Returns True if the line segments p1p2 and q1q2 intersect."""
            def orientation(p, q, r):
                """Returns the orientation of the triplet (p, q, r).
                0 -> p, q and r are collinear
                1 -> Clockwise
                2 -> Counterclockwise
                """
                val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
                if val == 0:
                    return 0
                elif val > 0:
                    return 1
                else:
                    return 2

            def on_segment(p, q, r):
                """Checks if point q lies on line segment pr."""
                if (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
                    min(p[1], r[1]) <= q[1] <= max(p[1], r[1])):
                    return True
                return False

            o1 = orientation(p1, p2, q1)
            o2 = orientation(p1, p2, q2)
            o3 = orientation(q1, q2, p1)
            o4 = orientation(q1, q2, p2)

            if o1 != o2 and o3 != o4:
                return True

            if o1 == 0 and on_segment(p1, q1, p2):
                return True

            if o2 == 0 and on_segment(p1, q2, p2):
                return True

            if o3 == 0 and on_segment(q1, p1, q2):
                return True

            if o4 == 0 and on_segment(q1, p2, q2):
                return True

            return False

        n = len(vertices)
        if n < 4:
            return True

        for i in range(n - 1):
            for j in range(i + 1, n - 1):
                if abs(i - j) <= 1:
                    continue
                if i == 0 and j == n - 2:
                    continue
                if do_lines_intersect(vertices[i], vertices[i + 1], vertices[j], vertices[j + 1]):
                    return False

        return True
    
    @staticmethod
    def split_path(path, distance):
        
        distance *= 10
        
        path = npline.interpolate(path, 0.1)
        
        newpaths = []
        
        for i in range(0, len(path), distance):
            newpaths.append(path[i:i+distance])

        return newpaths
