from mpl_toolkits import mplot3d
import pandas as pd
import numpy as np
import numba as nb
import matplotlib.pyplot as plt

@nb.jit 
def Fx(y,z):
    """ X Prime """
    xt = -y - z
    return xt

@nb.jit
def Fy(x,y,a):
    yt = x + a*y
    return yt

@nb.jit
def Fz(x,z,b,c):
    zt = b + z*(x-c)
    return zt

@nb.jit
def solve_odes1(c, T = 500, dt = 0.001):
    """Solving the 4th Runge-Kutta Method"""
    t = np.arange(0, T, dt)
    x = np.zeros_like(t)
    y = np.zeros_like(t)
    z = np.zeros_like(t)
    a = 0.2
    b = 0.2

    for i in range(1, len(t)):
        x1 = dt * Fx(y[i - 1], z[i - 1])
        x2 = dt * Fx(y[i - 1] + (x1 / 2), z[i - 1] + (x1 / 2))
        x3 = dt * Fx(y[i - 1] + (x2 / 2), z[i - 1] + (x2 / 2))
        x4 = dt * Fx(y[i - 1] + (x3), z[i - 1] + (x3))
        x[i] = x[i - 1] + (x1 + 2 * x2 + 2 * x3 + x4) / 6
        
        y1 = dt * Fy(x[i - 1], y[i - 1],a)
        y2 = dt * Fy(x[i - 1] + (y1 / 2), y[i - 1] + (y1 / 2),a)
        y3 = dt * Fy(x[i - 1] + (y2 / 2), y[i - 1] + (y2 / 2),a)
        y4 = dt * Fy(x[i - 1] + (y3), y[i - 1] + (y3),a)
        y[i] = y[i - 1] + (y1 + 2 * y2 + 2 * y3 + y4) / 6
        
        z1 = dt * Fz(x[i - 1], z[i - 1],b,c)
        z2 = dt * Fz(x[i - 1] + (z1 / 2), z[i - 1] + (z1 / 2),b,c)
        z3 = dt * Fz(x[i - 1] + (z2 / 2), z[i - 1] + (z2 / 2),b,c)
        z4 = dt * Fz(x[i - 1] + (z3), z[i - 1] + (z3),b,c)
        z[i] = z[i - 1] + (z1 + 2 * z2 + 2 * z3 + z4) / 6
    return (t,x,y,z)

def solve_odes(c, T = 500, dt = 0.001):
    """Wrapper Function"""
    t,x,y,z = solve_odes1(c,T,dt)
    return pd.DataFrame({"t": t, "x": x, "y": y, "z": z})
            

@nb.jit
def plotx(sol):
    """Plots x(t)"""
    t = sol['t']
    x = sol['x']
    plot = plt.figure(figsize=(10, 10))
    plt.plot(t, x, color="red")
    plt.title("x vs t")
    plt.xlabel("t")
    plt.ylabel("x")
    T = 500
    plt.xlim(0, T)
    plt.ylim(-10, 10)
    plt.legend
    plt.show()


@nb.jit
def ploty(sol):
    """Plots y(t)"""
    t = sol['t']
    x = sol['y']
    T = 500
    plot = plt.figure(figsize=(10, 10))
    plt.plot(t, x, color="blue")
    plt.title("y vs t")
    plt.xlabel("t")
    plt.ylabel("y")
    plt.xlim(0, T)
    plt.ylim(-10, 10)
    plt.legend
    plt.show()


@nb.jit
def plotz(sol):
    """Plots z(t)"""
    t = sol['t']
    z = sol['z']
    T = 500
    plot = plt.figure(figsize=(10, 10))
    plt.plot(t, z, color="green")
    plt.title("z vs t")
    plt.xlabel("t")
    plt.ylabel("z")
    plt.xlim(0, T)
    plt.ylim(-10, 10)
    plt.legend
    plt.show()

@nb.jit
def plotxy(sol, S=100):
    """Plots x and y"""
    dt = 0.001
    N = int(S/dt)
    x = sol['x'][N:]
    y = sol['y'][N:]
    plot = plt.figure(figsize=(12, 12))
    plt.plot(x, y, color="red")
    plt.title("x vs y")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(-12, 12)
    plt.ylim(-12, 12)
    plt.legend
    plt.show()

@nb.jit
def plotyz(sol, S=100):
    """Plots y and z"""
    dt = 0.001
    N = int(S/dt)
    y = sol['y'][N:]
    z = sol['z'][N:]
    plot = plt.figure(figsize=(12, 12))
    plt.plot(y, z, color="blue")
    plt.title("y vs z")
    plt.xlabel("y")
    plt.ylabel("z")
    plt.xlim(-12, 12)
    plt.ylim(-12, 12)
    plt.legend
    plt.show()

@nb.jit
def plotxz(sol, S=100):
    """Plots x and z"""
    dt = 0.001
    N = int(S/dt)
    x = sol['x'][N:]
    z = sol['z'][N:]
    plot = plt.figure(figsize=(12, 12))
    plt.plot(x, z, color="yellow")
    plt.title("x vs z")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.xlim(-12, 12)
    plt.ylim(-12, 12)
    plt.legend
    plt.show()

@nb.jit
def plotxyz(sol, S=100):
    """Plots x, y and z"""
    dt = 0.001
    N = int(S/dt)
    x = sol['x'][N:]
    y = sol['y'][N:]
    z = sol['z'][N:]

    ax = plt.axes(projection='3d')
    ax.plot3D(x, y, z, 'black')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.title("x vs y vs z")
    ax.set_xlim(-12, 12)
    ax.set_ylim(-12, 12)
    ax.set_zlim(0, 25)
    ax.legend
    plt.show()

#@nb.jit
def findmaxima(x, S = 100):
    xmax = []
    for i in range (S+1, len(x)-1):
        if (x[i] > x[i-1] and x[i] > x[i+1]):
            xmax.append(x[i])
    return np.array(xmax)

#@nb.jit
def scatter(dc = 0.1):
    """Creates a scatter plot of (c,x) where x is the local maxima"""
    clist = np.arange(2,6,dc)

    for c in clist:
        sol = solve_odes(c)
        x = sol['x']
        a = findmaxima(x)
        b = np.ones_like(a)*c
        plt.plot(b, a, 'k.', ms=1)
    plt.title("Scatterplot of Maxima")
    plt.xlabel("c")
    plt.ylabel("local maxima")
    plt.xlim(2, 6)
    plt.ylim(3, 12)
    plt.show()
