import math
import shapely
import time
import numpy as np


def poly(center: tuple, radius: float, startAngle: float, num_points: int) -> list:


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

# draw a line centered on a point, at a length, at an angle
def line(center: tuple, length: float, angle: float) -> list:
    
    angle = (angle-90) * (math.pi / 180)
    
    radius = length / 2
    
    x0 = center[0] - radius * math.cos(angle)
    y0 = center[0] - radius * math.sin(angle)
    x1 = center[0] - radius * math.cos(angle + math.pi)
    y1 = center[0] - radius * math.sin(angle + math.pi)

    return [(x0, y0), (x1, y1)]


def contractexpand(poly, rotations, distance):
    
    poly = shapely.Polygon(poly)
    
    inner = []

    d = 0
    r = 0

    while(d < distance * rotations):
        
        inset = poly.buffer(-d, cap_style = 1, join_style = 2, mitre_limit=10)
        
        if isinstance(inset, shapely.MultiPolygon):
            for geom in inset.geoms:
                if geom.area > 1:
                    inset = geom
        
        if inset.is_empty:
            print(d)
            print(inset)
            break
        
        if  inset.length < 10:
            break
        
        rotRes = inset.length # change this number to increase/decrease resolution
        distStep = distance/rotRes
        rotStep = 1/rotRes
        
        if (d > 0):
            inset = reorder_polygon(inset, shapely.Point(inner[0]))
        
        point = inset.exterior.interpolate(r * inset.length)
        inner.append(point.coords[0])
        
        r += rotStep
        if (r > 1):
            r = 0
            
        d += distStep
    
    outer = []
        
    d = 0
    r = 0
            
    while(d < distance * rotations):
        
        inset = poly.buffer(d, cap_style = 1, join_style = 2, mitre_limit=10)
        
        if isinstance(inset, shapely.MultiPolygon):
            for geom in inset.geoms:
                if geom.area > 1:
                    inset = geom
        
        if inset.is_empty:
            print("outer")
            print(d)
            print(inset)
            break
        
        if  inset.length < 10:
            break
        
        rotRes = inset.length * 2 # change this number to increase/decrease resolution
        distStep = distance/rotRes
        rotStep = 1/rotRes
        
        #if (d > 0):
            #inset = self.reorder_polygon(inset, shapely.Point(outer[0]))
        
        point = inset.exterior.interpolate(r * inset.length)
        outer.append(point.coords[0])
        
        r += rotStep
        if (r > 1):
            r = 0
            
        d += distStep
        
    outer.reverse()

    newPoly = outer + inner
                    
    return newPoly

def concfill(poly, distance):
    
    poly = shapely.Polygon(poly)
    
    newPoly = []
    
    for v in poly.buffer(0, join_style = 2).exterior.coords:
        # self.vsk.geometry(shapely.Point(v).buffer(3))
        newPoly.append(v)
        
    d = 0
    r = 0
    
    buffertime = 0
    intertime = 0
    reodertime = 0
    
    while(True):
        
        start_time = time.time()
        inset = poly.buffer(-d, join_style = 1, mitre_limit=1)
        buffertime += time.time() - start_time
        
        if isinstance(inset, shapely.MultiPolygon):
            for geom in inset.geoms:
                if geom.area > 1:
                    inset = geom
        
        if inset.is_empty:
            print(d)
            print(inset)
            break
        
        if  inset.length < 10:
            break
        
        rotRes = inset.length # change this number to increase/decrease resolution
        distStep = distance/rotRes
        rotStep = 1/rotRes
        
        start_time = time.time()
        inset = reorder_polygon(inset, shapely.Point(newPoly[0]))
        reodertime += time.time() - start_time

        start_time = time.time()
        point = inset.exterior.interpolate(math.fmod(r,1) * inset.length)
        intertime += time.time() - start_time
        
        newPoly.append(point.coords[0])
        
        r += rotStep
        d += distStep
                    
    print(f'buffer = {buffertime}')
    print(f'reorder = {reodertime}')
    print(f'interp = {intertime}')



    return newPoly


def reorder_polygon(polygon, point):
    # Ensure the polygon is closed (last point should be the same as the first)
    if not polygon.is_closed:
        polygon = shapely.Polygon(list(polygon.exterior.coords))
    
    # Get the coordinates of the polygon
    coords = np.array(polygon.exterior.coords)
    
    # Extract point coordinates
    point_coords = np.array(point.coords[0])
    
    # Compute distances using vectorized operations
    distances = np.linalg.norm(coords[:-1] - point_coords, axis=1)
    
    # Find the index of the closest vertex
    closest_index = np.argmin(distances)
    
    # Reorder the polygon coordinates
    reordered_coords = np.vstack((coords[closest_index:], coords[:closest_index + 1]))
    
    # Create a new polygon with the reordered coordinates
    reordered_polygon = shapely.Polygon(reordered_coords)
    
    return reordered_polygon
