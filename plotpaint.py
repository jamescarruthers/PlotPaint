from dataclasses import dataclass

@dataclass
class Dab():
    x: float
    y: float

class Plotpaint:
    
    def __init__ (self):
        self.strokes = []
        
    def dab(self, x, y):
        self.strokes.append(Dab(x, y))
        
    def output(self):

        output = ""
        
        for stroke in self.strokes:
            output += stroke.x
        
        print(output)
        