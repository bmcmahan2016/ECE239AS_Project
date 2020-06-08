import numpy as np

class env:
    def __init__(self, b = 0.1, m=0.2, dt=0.001, x0=None, v0=None):
        self.b = b
        self.dt = dt
        self.m = m
        
        if x0 is None:
            self.x0 = np.zeros(2)
        else:
            self.x0 = x0
        if v0 is None:
            self.v0 = np.zeros(2)
        else:
            self.v0 = v0
        
        self.x = self.x0
        self.v = self.v0
        
        self.target = np.random.randn(2)
        return
    
    def step(self, force):
        a = force
        a -= self.b*self.v*np.linalg.norm(self.v)
        a /= self.m
        self.x += self.v*self.dt + 0.5*a*self.dt**2
        self.v += a*self.dt
        return np.hstack((self.x, self.v, self.target))
    
    def setTarget(self, target=None):
        self.target = target
        return target
    
    def reset(self, x0=None, v0=None):
        if x0 is None:
            self.x0 = np.zeros(2)
        else:
            self.x0 = x0.copy()
        if v0 is None:
            self.v0 = np.zeros(2)
        else:
            self.v0 = v0.copy()
        
        self.x = self.x0
        self.v = self.v0
        
        self.target = np.random.randn(2)
        return np.hstack((self.x, self.v, self.target))