def linear(t):
    return t

def ease_in_quad(t):
    return t**2

def ease_out_quad(t):
    return 1 - (1 - t)**2

def ease_in_out_quad(t):
    if t < 0.5:
        return 2 * t**2
    else:
        return 1 - (-2 * t + 2)**2 / 2

def ease_in_cubic(t):
    return t**3

def ease_out_cubic(t):
    return 1 - (1 - t)**3

def ease_in_out_cubic(t):
    if t < 0.5:
        return 4 * t**3
    else:
        return 1 - (-2 * t + 2)**3 / 2

def ease_in_quart(t):
    return t**4

def ease_out_quart(t):
    return 1 - (1 - t)**4

def ease_in_out_quart(t):
    if t < 0.5:
        return 8 * t**4
    else:
        return 1 - (-2 * t + 2)**4 / 2
    