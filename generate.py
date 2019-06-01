import math
import numpy as np

def random_generate(n):
    np.random.seed(2222)
    
    x = np.linspace(-5, 4, n)
    
    spacing = 1./n
    noisex = np.random.uniform(spacing, 1-spacing, n)
    x = (x + noisex).reshape(-1,1)

    xmin, xmax = x.min(0), x.max(0)
    length = xmax - xmin
    
    lx = np.linspace(xmin, xmax, 500).reshape(500,1)
    elx = np.linspace(xmin-length/4., xmax+length/4., 500).reshape(500,1)
    
    y = np.sin(x).reshape(-1)
    R = cost(x,y)
    print "Minimum R for interpolation: %.2f" % R
    return x, y, lx, elx
    
def cost(x,y):
    flatx = x.reshape(-1)
    p = np.argsort(flatx)
    ordered_x = flatx[p]
    ordered_y = y[p]

    slopes = [(ordered_y[i+1] - ordered_y[i])/(ordered_x[i+1] - ordered_x[i]) for i in range(y.shape[0]-1)]
    absdiffs = [abs(slopes[i+1] - slopes[i]) for i in range(len(slopes)-1)]
    R = sum(absdiffs)
    return R
    
    
    
    
