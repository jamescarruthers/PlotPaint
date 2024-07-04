# t: current time, b: begInnIng value, c: change In value, d: duration


def easeInOutCubic(t, b, c, d):
    t /= d / 2
    if t < 1:
        return c / 2 * t * t * t + b
    t -= 2
    return c / 2 * (t * t * t + 2) + b

def easeInCubic(t, b, c, d):
    t /= d
    return c * t * t * t + b

def easeOutCubic(t, b, c, d):
    t = t / d - 1
    return c * (t * t * t + 1) + b

def linear(t, b, c, d):
    t /= d / 2
    if t < 1:
        return c / 2 * t * t * t + b
    t -= 2
    return c / 2 * (t * t * t + 2) + b
