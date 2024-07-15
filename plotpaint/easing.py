import math
import numpy as np
class Easing:
    # easing functions
    # t = current time, b = beginning value, c = change in value, d = total time
    @staticmethod
    def circOutNP(x):
        x = np.clip(x, 0, 1)
        return np.sqrt(1 - np.power(x - 1, 2))
    
    @staticmethod
    def circInNP(x):
        x = np.clip(x, 0, 1)
        return 1 - np.sqrt(1 - np.power(x, 2))


    def linear(self, t, b, c, d):
        return c * t / d + b

    def quadIn(self, t, b, c, d):
        t /= d
        return c * t * t + b

    def quadOut(self, t, b, c, d):
        t /= d
        return -c * t * (t - 2) + b

    def quadInOut(self, t, b, c, d):
        t /= d / 2
        if t < 1:
            return c / 2 * t * t + b
        t -= 1
        return -c / 2 * (t * (t - 2) - 1) + b

    def cubicIn(self, t, b, c, d):
        t /= d
        return c * t * t * t + b

    def cubicOut(self, t, b, c, d):
        t = t / d - 1
        return c * (t * t * t + 1) + b

    def cubicInOut(self, t, b, c, d):
        t /= d / 2
        if t < 1:
            return c / 2 * t * t * t + b
        t -= 2
        return c / 2 * (t * t * t + 2) + b

    def quartIn(self, t, b, c, d):
        t /= d
        return c * t * t * t * t + b

    def quartOut(self, t, b, c, d):
        t = t / d - 1
        return -c * (t * t * t * t - 1) + b

    def quartInOut(self, t, b, c, d):
        t /= d / 2
        if t < 1:
            return c / 2 * t * t * t * t + b
        t -= 2
        return -c / 2 * (t * t * t * t - 2) + b

    def quintIn(self, t, b, c, d):
        t /= d
        return c * t * t * t * t * t + b

    def quintOut(self, t, b, c, d):
        t = t / d - 1
        return c * (t * t * t * t * t + 1) + b

    def quintInOut(self, t, b, c, d):
        t /= d / 2
        if t < 1:
            return c / 2 * t * t * t * t * t + b
        t -= 2
        return c / 2 * (t * t * t * t * t + 2) + b

    def sineIn(self, t, b, c, d):
        return -c * math.cos(t / d * (math.pi / 2)) + c + b

    def sineOut(self, t, b, c, d):
        return c * math.sin(t / d * (math.pi / 2)) + b

    def sineInOut(self, t, b, c, d):
        return -c / 2 * (math.cos(math.pi * t / d) - 1) + b

    def expoIn(self, t, b, c, d):
        return b if t == 0 else c * math.pow(2, 10 * (t / d - 1)) + b

    def expoOut(self, t, b, c, d):
        return b + c if t == d else c * (-math.pow(2, -10 * t / d) + 1) + b

    def expoInOut(self, t, b, c, d):
        if t == 0:
            return b
        if t == d:
            return b + c
        t /= d / 2
        if t < 1:
            return c / 2 * math.pow(2, 10 * (t - 1)) + b
        t -= 1
        return c / 2 * (-math.pow(2, -10 * t) + 2) + b

    def circIn(self, t, b, c, d):
        t /= d
        return -c * (math.sqrt(1 - t * t) - 1) + b

    def circOut(self, t, b, c, d):
        t = t / d - 1
        return c * math.sqrt(1 - t * t) + b

    def circInOut(self, t, b, c, d):
        t /= d / 2
        if t < 1:
            return -c / 2 * (math.sqrt(1 - t * t) - 1) + b
        t -= 2
        return c / 2 * (math.sqrt(1 - t * t) + 1) + b

    def elasticIn(self, t, b, c, d, s=1.70158):
        p = 0
        a = c
        if t == 0:
            return b
        t /= d
        if t == 1:
            return b + c
        if not p:
            p = d * 0.3
        if a < abs(c):
            a = c
            s = p / 4
        else:
            s = p / (2 * math.pi) * math.asin(c / a)
        t -= 1
        return -(a * math.pow(2, 10 * t) * math.sin((t * d - s) * (2 * math.pi) / p)) + b

    def elasticOut(self, t, b, c, d, s=1.70158):
        p = 0
        a = c
        if t == 0:
            return b
        t /= d
        if t == 1:
            return b + c
        if not p:
            p = d * 0.3
        if a < abs(c):
            a = c
            s = p / 4
        else:
            s = p / (2 * math.pi) * math.asin(c / a)
        return a * math.pow(2, -10 * t) * math.sin((t * d - s) * (2 * math.pi) / p) + c + b

    def elasticInOut(self, t, b, c, d, s=1.70158):
        p = 0
        a = c
        if t == 0:
            return b
        t /= d / 2
        if t == 2:
            return b + c
        if not p:
            p = d * (0.3 * 1.5)
        if a < abs(c):
            a = c
            s = p / 4
        else:
            s = p / (2 * math.pi) * math.asin(c / a)
        if t < 1:
            t -= 1
            return -0.5 * (a * math.pow(2, 10 * t) * math.sin((t * d - s) * (2 * math.pi) / p)) + b
        t -= 1
        return a * math.pow(2, -10 * t) * math.sin((t * d - s) * (2 * math.pi) / p) * 0.5 + c + b

    def backIn(self, t, b, c, d, s=1.70158):
        t /= d
        return c * t * t * ((s + 1) * t - s) + b

    def backOut(self, t, b, c, d, s=1.70158):
        t = t / d - 1
        return c * (t * t * ((s + 1) * t + s) + 1) + b

    def backInOut(self, t, b, c, d, s=1.70158):
        t /= d / 2
        if t < 1:
            s *= 1.525
            return c / 2 * (t * t * ((s + 1) * t - s)) + b
        t -= 2
        s *= 1.525
        return c / 2 * (t * t * ((s + 1) * t + s) + 2) + b

    def bounceIn(self, t, b, c, d):
        return c - self.bounceOut(d - t, 0, c, d) + b

    def bounceOut(self, t, b, c, d):
        t /= d
        if t < (1 / 2.75):
            return c * (7.5625 * t * t) + b
        elif t < (2 / 2.75):
            t -= (1.5 / 2.75)
            return c * (7.5625 * t * t + 0.75) + b
        elif t < (2.5 / 2.75):
            t -= (2.25 / 2.75)
            return c * (7.5625 * t * t + 0.9375) + b
        else:
            t -= (2.625 / 2.75)
            return c * (7.5625 * t * t + 0.984375) + b                        
    
    def bounceInOut(self, t, b, c, d):
        if t < d / 2:
            return self.bounceIn(t * 2, 0, c, d) * 0.5 + b
        return self.bounceOut(t * 2 - d, 0, c, d) * 0.5 + c * 0.5 + b