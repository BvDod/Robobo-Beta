def random_point_in_circle(circle_radius):
    """ Returns a random x,y point in a circle, with center 0,0 """
    angle = 2 * math.pi * random.random()           # random angle
    r = circle_radius * math.sqrt(random.random())  # random radius
    # calculating coordinates
    x = r * math.cos(angle)
    y = r * math.sin(angle)
    return x, y