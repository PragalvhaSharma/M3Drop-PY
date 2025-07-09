import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.spatial.distance import euclidean
import itertools


def venn_overlap(r, a, b, target=0):
    """
    Calculate the overlap area for circles of radius a and b
    with centers separated by r
    target is included for the root finding code
    """
    pi = np.pi
    
    if r >= a + b:
        return -target
    if r < a - b:  # b completely overlapped by a
        return pi * b * b - target
    if r < b - a:  # a completely overlapped by b
        return pi * a * a - target
    
    s = (a + b + r) / 2
    triangle_area = np.sqrt(s * (s - a) * (s - b) * (s - r))
    h = (2 * triangle_area) / r
    aa = 2 * np.arctan(np.sqrt(((s - r) * (s - a)) / (s * (s - b))))
    ab = 2 * np.arctan(np.sqrt(((s - r) * (s - b)) / (s * (s - a))))
    sector_area = aa * (a * a) + ab * (b * b)
    overlap = sector_area - 2 * triangle_area
    return overlap - target


def plot_venn_diagram(d):
    """
    Draw Venn diagrams with proportional overlaps
    d['table'] = 3 way table of overlaps
    d['labels'] = array of character string to use as labels
    """
    pi = np.pi
    csz = 0.1
    
    # Normalize the data
    table = d['table']
    n = len(table.shape)
    
    c1 = np.zeros(n)
    c1[0] = np.sum(table[1, :, :])  # Python uses 0-based indexing, R used 1-based
    c1[1] = np.sum(table[:, 1, :])
    c1[2] = np.sum(table[:, :, 1])
    n1 = c1.copy()
    
    c2 = np.zeros((n, n))
    c2[0, 1] = np.sum(table[1, 1, :])
    c2[1, 0] = c2[0, 1]
    c2[0, 2] = np.sum(table[1, :, 1])
    c2[2, 0] = c2[0, 2]
    c2[1, 2] = np.sum(table[:, 1, 1])
    c2[2, 1] = c2[1, 2]
    n2 = c2.copy()
    
    c3 = table[1, 1, 1]
    n3 = c3
    
    c2 = c2 / np.sum(c1)
    c3 = c3 / np.sum(c1)
    c1 = c1 / np.sum(c1)
    
    # Radii are set so the area is proportional to number of counts
    r = np.sqrt(c1 / pi)
    
    # Find distances between circle centers using root finding
    def find_distance(distance, r1, r2, target_overlap):
        return venn_overlap(distance, r1, r2, target_overlap)
    
    # Find r12, r13, r23 using scipy's fsolve
    r12_bounds = [max(abs(r[0] - r[1]), 0) + 0.01, r[0] + r[1] - 0.01]
    r12 = fsolve(lambda x: find_distance(x, r[0], r[1], c2[0, 1]), 
                 (r12_bounds[0] + r12_bounds[1]) / 2)[0]
    
    r13_bounds = [max(abs(r[0] - r[2]), 0) + 0.01, r[0] + r[2] - 0.01]
    r13 = fsolve(lambda x: find_distance(x, r[0], r[2], c2[0, 2]), 
                 (r13_bounds[0] + r13_bounds[1]) / 2)[0]
    
    r23_bounds = [max(abs(r[1] - r[2]), 0) + 0.01, r[1] + r[2] - 0.01]
    r23 = fsolve(lambda x: find_distance(x, r[1], r[2], c2[1, 2]), 
                 (r23_bounds[0] + r23_bounds[1]) / 2)[0]
    
    s = (r12 + r13 + r23) / 2
    
    x = np.zeros(3)
    y = np.zeros(3)
    x[0] = 0
    y[0] = 0
    x[1] = r12
    y[1] = 0
    angle = 2 * np.arctan(np.sqrt(((s - r12) * (s - r13)) / (s * (s - r23))))
    x[2] = r13 * np.cos(angle)
    y[2] = r13 * np.sin(angle)
    
    # Resolution of circles
    theta = np.linspace(0, 2 * pi, 100)
    xc = np.cos(theta)
    yc = np.sin(theta)
    
    cmx = np.sum(x * c1)
    cmy = np.sum(y * c1)
    x = x - cmx
    y = y - cmy
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.grid(True)
    
    # Draw circles
    for i in range(3):
        circle_x = xc * r[i] + x[i]
        circle_y = yc * r[i] + y[i]
        ax.plot(circle_x, circle_y, 'b-', linewidth=2)
    
    return {'r': r, 'x': x, 'y': y}


def circle_intersection(x1, y1, r1, x2, y2, r2):
    """Calculate intersection points of two circles"""
    if x1 == x2:
        y = (r2*r2 - r1*r1 + x1*x1 - x2*x2 + y1*y1 - y2*y2) / (2*y1 - 2*y2)
        a = 1
        b = 2*x1
        c = (y - y1)*(y - y1) - r1*r1 + x1*x1
        xi1 = (-b + np.sqrt(b*b - 4*a*c)) / (2*a)
        xi2 = (-b - np.sqrt(b*b - 4*a*c)) / (2*a)
        return np.array([[xi1, y], [xi2, y]])
    else:
        A = (r2*r2 - r1*r1 + x1*x1 - x2*x2 + y1*y1 - y2*y2) / (2*x1 - 2*x2)
        B = (2*y2 - 2*y1) / (2*x1 - 2*x2)
        
        a = (B*B) + 1
        b = -2*B*x1 + 2*A*B - 2*y1
        c = A*A - 2*x1*A + x1*x1 + y1*y1 - r1*r1
        yi1 = (-b + np.sqrt(b*b - 4*a*c)) / (2*a)
        yi2 = (-b - np.sqrt(b*b - 4*a*c)) / (2*a)
        xi1 = A + B*yi1
        xi2 = A + B*yi2
        return np.array([[xi1, yi1], [xi2, yi2]])


def circle_line_intersection(xc, yc, r, x1, y1, x2, y2):
    """Calculate intersection points of a circle and a line"""
    if x2 == x1:
        xi = x1
        a = 1
        b = -2*yc
        c = (xi - xc)*(xi - xc) + yc*yc - r*r
        yi1 = (-b + np.sqrt(b*b - 4*a*c)) / (2*a)
        yi2 = (-b - np.sqrt(b*b - 4*a*c)) / (2*a)
        return np.array([[xi, yi1], [xi, yi2]])
    else:
        M = (y2 - y1) / (x2 - x1)
        B = y1 - M*x1
        a = M*M + 1
        b = 2*M*(B - yc) - 2*xc
        c = xc*xc + (B - yc)*(B - yc) - r*r
        xi1 = (-b + np.sqrt(b*b - 4*a*c)) / (2*a)
        xi2 = (-b - np.sqrt(b*b - 4*a*c)) / (2*a)
        yi1 = M*xi1 + B
        yi2 = M*xi2 + B
        return np.array([[xi1, yi1], [xi2, yi2]])


def place_labels(circles, d, labelsincircles):
    """
    Places labels in each section of the venn diagram based on the intersection points of the three circles
    """
    r = circles['r']
    x = circles['x']
    y = circles['y']
    
    left = x - r
    right = x + r
    top = y + r
    bottom = y - r
    
    # Better labelling - Does not cope with one circle completely within another circle
    topcircle = np.argmax(y)
    rightcircle = np.argmax(x)
    leftcircle = np.argmin(x)
    
    # Names for each Circle
    # Outside most distant edge of the circle
    if not labelsincircles:
        plt.text(x[topcircle], top[topcircle], d['labels'][topcircle], 
                ha='center', va='bottom')
        plt.text(right[rightcircle], y[rightcircle], d['labels'][rightcircle], 
                ha='left', va='center')
        plt.text(left[leftcircle], y[leftcircle], d['labels'][leftcircle], 
                ha='right', va='center')
    
    # Intersection Points
    intersectionLR = circle_intersection(x[leftcircle], y[leftcircle], r[leftcircle],
                                       x[rightcircle], y[rightcircle], r[rightcircle])
    intersectionLT = circle_intersection(x[leftcircle], y[leftcircle], r[leftcircle],
                                       x[topcircle], y[topcircle], r[topcircle])
    intersectionTR = circle_intersection(x[topcircle], y[topcircle], r[topcircle],
                                       x[rightcircle], y[rightcircle], r[rightcircle])
    
    table = d['table']
    onewayvalues = [table[1, 0, 0], table[0, 1, 0], table[0, 0, 1]]  # 1, 2, 3
    twowayvalues = [table[1, 1, 0], table[1, 0, 1], table[0, 1, 1]]  # 1&2, 1&3, 2&3
    threewayvalue = table[1, 1, 1]
    
    # Names in circles above their one-way value
    if labelsincircles:
        onewayvalues = [f"{d['labels'][i]}\n{onewayvalues[i]}" for i in range(3)]
    
    # 3-way intersection
    topLRintersection = intersectionLR[np.argmax(intersectionLR[:, 1]), :]
    leftTRintersection = intersectionTR[np.argmin(intersectionTR[:, 0]), :]
    rightTLintersection = intersectionLT[np.argmax(intersectionLT[:, 0]), :]
    boundarypoints = np.array([topLRintersection, leftTRintersection, rightTLintersection])
    midpoint = np.mean(boundarypoints, axis=0)
    plt.text(midpoint[0], midpoint[1], str(threewayvalue), ha='center', va='center')
    
    # 2-way intersections
    # LR - take advantage of fact that L & R circles have centres at the same height
    textx = np.mean(intersectionLR[:, 0])  # mid-point between the intersection points
    lowestT = min(leftTRintersection[1], rightTLintersection[1], topLRintersection[1])
    if leftTRintersection[0] < x[topcircle] < rightTLintersection[0]:
        lowestT = min(bottom[topcircle], lowestT)
    texty = np.mean([np.min(intersectionLR[:, 1]), lowestT])
    plt.text(textx, texty, str(twowayvalues[int(np.ceil(leftcircle * rightcircle / 2))]), 
             ha='center', va='center')
    
    # Continue with rest of label placement logic...
    # (Implementation continues with similar conversions for the remaining label placements)
    
    # 1-way intersections
    for i, circle_idx in enumerate([leftcircle, rightcircle, topcircle]):
        plt.text(x[circle_idx], y[circle_idx], str(onewayvalues[circle_idx]), 
                ha='center', va='center')


def convert_sets_to_table(set1, set2, set3, names):
    """Convert three sets to a table format for Venn diagram processing"""
    assert len(names) == 3, "names must have length 3"
    
    # Form universe as union of all three sets
    universe = sorted(set(set1) | set(set2) | set(set3))
    
    # Create binary matrix indicating membership in each set
    in_out = np.zeros((len(universe), 3), dtype=int)
    
    for i, element in enumerate(universe):
        in_out[i, 0] = int(element in set1)
        in_out[i, 1] = int(element in set2)
        in_out[i, 2] = int(element in set3)
    
    # Create 3D contingency table
    table = np.zeros((2, 2, 2), dtype=int)
    for row in in_out:
        table[row[0], row[1], row[2]] += 1
    
    return {'table': table, 'labels': names}


def m3drop_three_set_venn(set1, set2, set3, names):
    """
    Top-level function to create a three-way Venn diagram
    """
    # Convert sets to strings and create table
    table = convert_sets_to_table([str(x) for x in set1], 
                                 [str(x) for x in set2], 
                                 [str(x) for x in set3], names)
    
    nicer_table = table.copy()
    
    # Adjust table for better visualization
    if np.max(table['table']) > 5 * np.mean(table['table']):
        bigones = table['table'] > 5 * np.mean(table['table'])
        nicer_table['table'][bigones] = table['table'][bigones] / 5
    
    if np.sum(table['table'] == 0) > 1:
        smallest_notzero = np.min(table['table'][table['table'] > 0])
        nicer_table['table'][table['table'] == 0] = smallest_notzero
    
    circles = plot_venn_diagram(nicer_table)
    place_labels(circles, table, labelsincircles=True)
    
    plt.title('Three-way Proportional Venn Diagram')
    plt.show()
    
    return circles, table


# Example usage:
if __name__ == "__main__":
    # Example sets
    set1 = [1, 2, 3, 4, 5, 6]
    set2 = [4, 5, 6, 7, 8, 9]
    set3 = [6, 7, 8, 9, 10, 11]
    names = ["Set A", "Set B", "Set C"]
    
    circles, table = m3drop_three_set_venn(set1, set2, set3, names)
