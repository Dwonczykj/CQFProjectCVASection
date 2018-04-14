import numpy as np
def IntegrateNum_Trapezoid(x,y):
    if not len(x) == len(y):
        raise ValueError("x is not same length as y")
    n = len(x)
    intSum = 0
    for i in range(1,n):
        intSum += ((y[i] + y[i-1])/2) * (x[i] - x[i-1])
    return intSum

def WeightedNumericalIntFromZero(tau,f):
    dt = 0.01

    if tau == 0:
        return 0

    N  = int(tau/dt)
    dt = tau / N
    
    intSum = 0.5 * dt * f(0)
    for t in np.arange(dt,tau,dt):
        intSum += f(t) * dt
    intSum += f(tau) * dt * 0.5
    intSum *= f(tau)
    return intSum