import np.exp as exp
import np.sqrt as sqrt
D=1
l=2
v=3
k=4
t=5
R=6
p = v^2 - 4*k*D + 4*l*D
g = (v + (sqrt(p)))/(2*D)
f = (v - (sqrt(p)))/(2*D)


(
    + D*v^2*exp(l*t)*exp(g)
    + 2*k*v^2*exp(l*t)*exp(g)
    - 2*l*v^2*exp(l*t)*exp(g)
    - D*v^2*exp(f)*exp(l*t)
    - D*exp(l*t)*exp(g)*(p)
    - 2*k*v^2*exp(f)*exp(l*t)
    + 2*l*v^2*exp(f)*exp(l*t)
    + D*exp(f)*exp(l*t)*(p)
    + 2*v*(sqrt(p)) * (
        + k*exp(l*t)*exp(g)
        - l*exp(l*t)*exp(g)
        + k*exp(f)*exp(l*t)
        - l*exp(f)*exp(l*t)
        + 2*R*k*exp(f)*exp(g)
        - 2*R*l*exp(f)*exp(g)
        )
    )


-(
    exp(-l*t)*(
        D*exp(f)*(p) 
        + D*v^2*exp(g) 
        + 2*k*v^2*exp(g) 
        - 2*l*v^2*exp(g) 
        - D*v^2*exp(f) 
        - D*exp(g)*(p) 
        - 2*k*v^2*exp(f) 
        + 2*l*v^2*exp(f) 
        + 2*k*v*exp(f)*sqrt(p) 
        - 2*l*v*exp(f)*sqrt(p) 
        + 2*k*v*exp(g)*sqrt(p) 
        - 2*l*v*exp(g)*sqrt(p) 
        + 4*R*k*v*exp(f)*exp(l*t)*exp(g)*sqrt(p)
        - 4*R*l*v*exp(f)*exp(l*t)*exp(g)*sqrt(p)
    )
)/(
    4*(k - l)*sqrt(p)
)
 