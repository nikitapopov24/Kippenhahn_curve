import numpy as np
import matplotlib.pyplot as plt

# helper functions
def lambda_index(A, theta, ind):
    A_ = np.exp(1j * theta) * A 
    A__ = (A_ + A_.T.conj()) / 2
    eigs = (np.linalg.eigvals(A__)).real
    eigs.sort()
    return eigs[ind]

def tangent_lines(A, p=100):
    thetas = np.linspace(0, 2*np.pi, p, endpoint=False)
    d = 1
    line = np.array([0,-d*1j,d*1j])
    n = A.shape[0]
    eigs = np.zeros((n,p,line.shape[0]), dtype=complex)
    for i in range(n):
        eigs[i] = np.array([(lambda_index(A, theta,i).repeat(line.shape[0])+line) * np.exp(-1j * theta) for theta in thetas])  

    return eigs

def neighboor_intersections(lines):
    intersections = np.array([])
    n = lines.shape[0]
    for i in range(n):
        current_line = lines[i]
        next_line = lines[(i+1)%n]
        # find intersection of current and next
        x1, y1 = current_line[0].real, current_line[0].imag
        x2, y2 = current_line[1].real, current_line[1].imag
        x3, y3 = next_line[0].real, next_line[0].imag
        x4, y4 = next_line[1].real, next_line[1].imag
        d = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if d == 0:
            continue
        xi = ((x3-x4)*(x1*y2-y1*x2)-(x1-x2)*(x3*y4-y3*x4))/d
        yi = ((y3-y4)*(x1*y2-y1*x2)-(y1-y2)*(x3*y4-y3*x4))/d
        intersections = np.append(intersections, [xi, yi])
    return intersections.reshape(-1,2)

# Kippenhahn curve: putting it all together

def Kippenhahn_curve(A, num_lines=100, center=True, show=True):
    lines = tangent_lines(A, num_lines)
    components = np.zeros((A.shape[0], lines.shape[1], 2), dtype=float)
    for i in range(A.shape[0]): 
        components[i] = neighboor_intersections(lines[i])
        if show:
            plt.scatter(components[i,:,0], components[i,:,1], s=0.01)
    if show:
        if center:
            plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
    return components

def graph_components(components, center=True, show=True, s=0.01):
    # create axis and figure
    fig, ax = plt.subplots()
    for i in range(components.shape[0]):
        ax.scatter(components[i,:,0], components[i,:,1], s=s)
    if center:
        fig.gca().set_aspect('equal', adjustable='box')
    if show:
        fig.show()
    return fig, ax
    
# example
# A = np.array([[0, 0, 0.99843816,   0.01924716,   0,             0.052366039    ],
#               [0, 0, 0,            0.937118467,  -0.0594182575, -0.343363113   ],
#               [0, 0, 0.0558680672, -0.343972794, 0,             -0.935852165   ],
#               [0, 0, 0,            0.0558680672, 0.996668566,   -0.02047081    ],
#               [0, 0, 0,            0,            0.0558680672,  0.0000102909091],
#               [0, 0, 0,            0,            0,             0.0558680672   ]])

# kipp_curve = Kippenhahn_curve(A, 2000)