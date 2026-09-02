def make_sweep(min_x, max_x, min_y, max_y, spacing):
    """Return a list fo (x,y) points that snake back and forther over the search area"""

    points= []
    y = min_y
    flip= False
    while y<=max_y + 0.001:
        if flip:
            points.append((max_x,y))
            points.append((min_x,y))
        else:
            points.append((min_x, y))
            points.append((max_x,y))
        flip = not flip
        y += spacing
    return points

print(make_sweep(-8, 8, -8, 8, 4.0))
