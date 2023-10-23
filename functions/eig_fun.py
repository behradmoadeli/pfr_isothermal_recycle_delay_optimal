def eig_fun_1(x, *args):
    
    import numpy as np
    
    par = args[0]
    (k, v, D, t, R) = (par['k'], par['v'], par['D'], par['tau'], par['R'])
    l = args[1]

    p = v**2 - 4*D * (k-l)
    p_sqrt = np.sqrt(p)
    
    r = [(v+p_sqrt)/(2*D),
         (v-p_sqrt)/(2*D)
    ]

    a = - (r[1]/r[0]) * np.exp(r[1]-r[0])

    phi = a * np.exp(r[0]*x) + np.exp(r[1]*x)

    return phi

def eig_fun_2(x, *args):
    
    import numpy as np
    
    par = args[0]
    (k, v, D, t, R) = (par['k'], par['v'], par['D'], par['tau'], par['R'])
    l = args[1]

    p = v**2 - 4*D * (k-l)
    p_sqrt = np.sqrt(p)
    
    r = [(v+p_sqrt)/(2*D),
         (v-p_sqrt)/(2*D)
    ]

    c = (1 - r[1]/r[0]) * np.exp(r[1]-t*l)

    psi = c * np.exp(t*l*x)

    return psi


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
    
    r = [-(v+p_sqrt)/(2*D),
         -(v-p_sqrt)/(2*D)
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
    
    r = [-(v+p_sqrt)/(2*D),
         -(v-p_sqrt)/(2*D)
    ]

    c = 1 - r[1]/r[0]
    
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
    
    r = [-(v+p_sqrt)/(2*D),
         -(v-p_sqrt)/(2*D)
    ]

    a = -r[1]/r[0]
    
    phi_star = a * r[0] * np.exp(r[0]*x) + r[1] * np.exp(r[1]*x)

    return phi_star

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

    phi = eig_fun_1(x, par, l[0])
    psi = eig_fun_2(x, par, l[0])
    
    phi_star = eig_fun_adj_1(x, par, l[1], b)
    psi_star = eig_fun_adj_2(x, par, l[1], b)

    return phi * phi_star + psi * psi_star