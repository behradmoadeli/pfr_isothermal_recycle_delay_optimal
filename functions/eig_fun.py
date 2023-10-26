def eig_fun_1(x, *args):
    
    import numpy as np
    
    par = args[0]
    (k, v, D, t, R) = (par['k'], par['v'], par['D'], par['tau'], par['R'])
    l = args[1]

    p = v**2 - 4*D * (k-l)
    p_sqrt = np.sqrt(p)
    
    try:
        b = args[2]
    except:
        b = 1
    
    r = [(v+p_sqrt)/(2*D),
         (v-p_sqrt)/(2*D)
    ]

    a = - (r[1]/r[0]) * np.exp(r[1]-r[0])

    phi = a * np.exp(r[0]*x) + np.exp(r[1]*x)

    return b * phi

def eig_fun_2(x, *args):
    
    import numpy as np
    
    par = args[0]
    (k, v, D, t, R) = (par['k'], par['v'], par['D'], par['tau'], par['R'])
    l = args[1]

    p = v**2 - 4*D * (k-l)
    p_sqrt = np.sqrt(p)
    
    try:
        b = args[2]
    except:
        b = 1
    
    r = [(v+p_sqrt)/(2*D),
         (v-p_sqrt)/(2*D)
    ]

    c = (1 - r[1]/r[0]) * np.exp(r[1]-t*l)

    psi = c * np.exp(t*l*x)

    return b * psi


def eig_fun_adj_1(x, *args):
    
    import numpy as np
    
    par = args[0]
    (k, v, D, t, R) = (par['k'], par['v'], par['D'], par['tau'], par['R'])
    l = args[1]

    p = v**2 - 4*D * (k-l)
    p_sqrt = np.sqrt(p)
    
    try:
        b = args[2]
    except:
        b = 1
    
    r = [-(v-p_sqrt)/(2*D),
         -(v+p_sqrt)/(2*D)
    ]

    a = -r[1]/r[0]
    
    phi_star = a * np.exp(r[0]*x) + np.exp(r[1]*x)

    return b * phi_star

def eig_fun_adj_2(x, *args):
    
    import numpy as np
    
    par = args[0]
    (k, v, D, t, R) = (par['k'], par['v'], par['D'], par['tau'], par['R'])
    l = args[1]

    p = v**2 - 4*D * (k-l)
    p_sqrt = np.sqrt(p)
    
    try:
        b = args[2]
    except:
        b = 1
    
    r = [-(v-p_sqrt)/(2*D),
         -(v+p_sqrt)/(2*D)
    ]

    c = R * v * t * (1 - r[1]/r[0])
    
    psi_star = c * np.exp(-t*l*x)

    return b * psi_star

def eig_fun_1_prime(x, *args):
    
    import numpy as np
    
    par = args[0]
    (k, v, D, t, R) = (par['k'], par['v'], par['D'], par['tau'], par['R'])
    l = args[1]

    p = v**2 - 4*D * (k-l)
    p_sqrt = np.sqrt(p)
    
    try:
        b = args[2]
    except:
        b = 1
    
    r = [(v+p_sqrt)/(2*D),
         (v-p_sqrt)/(2*D)
    ]

    a = - (r[1]/r[0]) * np.exp(r[1]-r[0])

    phi = a * r[0] * np.exp(r[0]*x) + r[1] * np.exp(r[1]*x)

    return b * phi

def eig_fun_adj_1_prime(x, *args):
    
    import numpy as np
    
    par = args[0]
    (k, v, D, t, R) = (par['k'], par['v'], par['D'], par['tau'], par['R'])
    l = args[1]

    p = v**2 - 4*D * (k-l)
    p_sqrt = np.sqrt(p)
    
    try:
        b = args[2]
    except:
        b = 1
    
    r = [-(v-p_sqrt)/(2*D),
         -(v+p_sqrt)/(2*D)
    ]

    a = -r[1]/r[0]
    
    phi_star = a * r[0] * np.exp(r[0]*x) + r[1] * np.exp(r[1]*x)

    return b * phi_star

def arbit_fun_1(x, *args):
    
    import numpy as np

    # y = np.ones_like(x) * 6
    y = np.cos(2*np.pi*x)
    
    return y

def arbit_fun_2(x, *args):
    
    import numpy as np
    
    par = args[0]
    (k, v, D, t, R) = (par['k'], par['v'], par['D'], par['tau'], par['R'])
    
    y = 1/R + (np.cos(2*np.pi - 1/R)) * x

    return y

def eig_fun_mul_1(x, *args):
    
    par = args[0]
        
    if hasattr(args[1], '__iter__'):
        l = args[1]
    else:
        l = [args[1]]*2
    
    try:
        b = args[2]
    except:
        b = 1

    phi = eig_fun_1(x, par, l[0], b)
    psi = eig_fun_2(x, par, l[0], b)
    
    phi_star = eig_fun_adj_1(x, par, l[1], b)
    psi_star = eig_fun_adj_2(x, par, l[1], b)

    return phi * phi_star + psi * psi_star

def eig_fun_mul_2(x, *args):
    
    par = args[0]
    l = args[1]
    
    try:
        b = args[2]
    except:
        b = 1

    z_1 = arbit_fun_1(x, par)
    z_2 = arbit_fun_2(x, par)
    
    phi_star = eig_fun_adj_1(x, par, l, b)
    psi_star = eig_fun_adj_2(x, par, l, b)

    return z_1 * phi_star + z_2 * psi_star