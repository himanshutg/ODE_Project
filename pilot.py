import numpy as np 
import matplotlib.pyplot as plt 
import math
'''
Y = [y1,y2,...yn]
yi(x) : R -> R
Y : R -> R^n
Y' = f(x,Y)
f : R^(n + 1) -> R^n
Y(x0) = Y0
'''


'''
Functions 
'''
def f1(x,Y):
    return math.sin(Y[1])
def f2(x,Y):
    return math.cos(-200*Y[0] - 102*Y[1])
def f(x,Y):
    return np.array([f1(x,Y),f2(x,Y)])
epsilon = 1e-8

def Del(g,Y,i):
    # del g/del yi
    g1 = g(Y)
    Y[i] += epsilon
    g2 = g(Y)
    return (g2 - g1)/epsilon

def F(Y,x_n,Y_n,h):
    return Y_n + h/2 * (f(x_n + h,Y) + f(x_n,Y_n)) - Y

def newtonRapson(Y_n,x_n,h):
    Y_k = Y_n
    J_k = np.zeros((N,N))
    # J_k
    # J_k[i,j] = del Fi / del yj
    tolerance = 1e-6
    error = 1
    iterations = 10000
    # print("starting Newton Rapson")
    while(iterations and error > tolerance):
        for i in range(N):
            for j in range(N):
                def Fi(Y_):
                    return F(Y_,x_n,Y_n,h)[i]
                J_k[i,j] = Del(Fi,Y_k,j)
        # print(" J = \n",J_k)
        # print(" y* => ",Y_k)
        Y_k = Y_k - np.linalg.inv(J_k)@F(Y_k,x_n,Y_n,h)
        iterations -= 1
        error = np.linalg.norm(F(Y_k,x_n,Y_n,h))
    return Y_k
def eqSolver(x0,x1,numPoints,Y0,N):
    h = (x1 - x0)/numPoints
    Y_n = Y0
    x_n = x0
    '''
    Y_(n + 1) = Y_(n) + h/2*(f(x_n + h,Y_(n + 1)) + f(x_n,Y_n))
    '''
    Output = np.zeros((N,numPoints + 1))
    for i in range(N):
        Output[i][0] = Y_n[i]
    
    print(Y_n)
    for n in range(1,numPoints + 1):
        '''
        Y_(n + 1) = Y_(n) + h/2*(f(x_n + h,Y_(n + 1)) + f(x_n,Y_n))
        Now we have to solve Y_(n + 1) [vector of size Nx1]
        Newton-Rapson
        F(Y_(n + 1)) = Y_(n) + h/2*(f(x_n + h,Y_(n + 1)) + f(x_n,Y_n)) - Y_(n + 1) 
        Find Y_(n + 1) = [y1,y2,...yN] s.t
        F(Y_(n + 1)) = 0
        '''
        # print("Y = ",Y_n)
        Y_n = newtonRapson(Y_n,x_n,h)
        x_n += h
        for i in range(N):
            Output[i][n] = Y_n[i]

    return Output
N = 2;'''Number of Functions'''
x0,x1 = 0,10
Y0 = np.array([1.0,-2.0])

numPoints = 10000
x_array = np.linspace(x0,x1,numPoints + 1)
y_array = eqSolver(x0,x1,numPoints,Y0,N)



plt.figure(figsize=(8, 6))
for i in range(N):
    plt.plot(x_array, y_array[i], label=f'y_{i}(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.savefig('save.png')
plt.show()


