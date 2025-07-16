import math
import numpy as np

def rgb255(v):
    return min(255, max(0, v))

def b1(v):
    if v > 0.0031308:
        return v**(1/2.4) * 269.025 - 14.025
    else:
        return v * 3294.6

def b2(v):
    if v > 0.2068965:
        return v**3
    else:
        return (v - 4/29) * (108/841)

def a1(v):
    if v > 10.314724:
        return ((v + 14.025) / 269.025)**2.4
    else:
        return v / 3294.6

def a2(v):
    if v > 0.0088564:
        return v**(1/3)
    else:
        return v / (108/841) + 4/29

def fromHCL(h, c, l):
    l = (l + 16) / 116
    y = b2(l)
    x = b2(l + (c / 500) * math.cos(h * math.pi / 180))
    z = b2(l - (c / 200) * math.sin(h * math.pi / 180))
    return [rgb255(b1(x * 3.021973625 - y * 1.617392459 - z * 0.404875592)),
            rgb255(b1(x * -0.943766287 + y * 1.916279586 + z * 0.027607165)),
            rgb255(b1(x * 0.069407491 - y * 0.22898585 + z * 1.159737864))]

def toHCL(r, g, b):
    r = a1(r)
    g = a1(g)
    b = a1(b)
    y = a2(r * 0.222488403 + g * 0.716873169 + b * 0.06060791)
    l = 500 * (a2(r * 0.452247074 + g * 0.399439023 + b * 0.148375274) - y)
    q = 200 * (y - a2(r * 0.016863605 + g * 0.117638439 + b * 0.865350722))
    h = math.atan2(q, l) * (180 / math.pi)
    return [h + 360 if h < 0 else h, math.sqrt(l**2 + q**2), 116 * y - 16]

def HCLtoRGB(H,C,L):
    shape = H.shape
    H2 = H.flatten()
    C2 = H.flatten()
    L2 = H.flatten()

    R2 = np.zeros(H2.shape)
    G2 = np.zeros(H2.shape)
    B2 = np.zeros(H2.shape)
    i = 0
    for (h,c,l) in zip(H2,C2,L2):
        r,g,b = fromHCL(h,c,l)
        R2[i]=r
        G2[i]=g
        B2[i]=b
    R = R2.reshape(shape)
    G = G2.reshape(shape)
    B = B2.reshape(shape)
    return (R,G,B)
