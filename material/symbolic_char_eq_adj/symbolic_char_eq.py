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


-(
    D*v^2*exp(l*t)*exp((v + (v^2 - 4*k*D + 4*l*D)^(1/2))/(2*D)) + 2*k*v^2*exp(l*t)*exp((v + (v^2 - 4*k*D + 4*l*D)^(1/2))/(2*D)) - 2*l*v^2*exp(l*t)*exp((v + (v^2 - 4*k*D + 4*l*D)^(1/2))/(2*D)) - D*v^2*exp((v - (v^2 - 4*k*D + 4*l*D)^(1/2))/(2*D))*exp(l*t) - D*exp(l*t)*exp((v + (v^2 - 4*k*D + 4*l*D)^(1/2))/(2*D))*(v^2 - 4*k*D + 4*l*D) - 2*k*v^2*exp((v - (v^2 - 4*k*D + 4*l*D)^(1/2))/(2*D))*exp(l*t) + 2*l*v^2*exp((v - (v^2 - 4*k*D + 4*l*D)^(1/2))/(2*D))*exp(l*t) + D*exp((v - (v^2 - 4*k*D + 4*l*D)^(1/2))/(2*D))*exp(l*t)*(v^2 - 4*k*D + 4*l*D) + 2*k*v*exp(l*t)*exp((v + (v^2 - 4*k*D + 4*l*D)^(1/2))/(2*D))*(v^2 - 4*k*D + 4*l*D)^(1/2) - 2*l*v*exp(l*t)*exp((v + (v^2 - 4*k*D + 4*l*D)^(1/2))/(2*D))*(v^2 - 4*k*D + 4*l*D)^(1/2) + 2*k*v*exp((v - (v^2 - 4*k*D + 4*l*D)^(1/2))/(2*D))*exp(l*t)*(v^2 - 4*k*D + 4*l*D)^(1/2) - 2*l*v*exp((v - (v^2 - 4*k*D + 4*l*D)^(1/2))/(2*D))*exp(l*t)*(v^2 - 4*k*D + 4*l*D)^(1/2) - 4*R*k*v*exp((v - (v^2 - 4*k*D + 4*l*D)^(1/2))/(2*D))*exp((v + (v^2 - 4*k*D + 4*l*D)^(1/2))/(2*D))*(v^2 - 4*k*D + 4*l*D)^(1/2) + 4*R*l*v*exp((v - (v^2 - 4*k*D + 4*l*D)^(1/2))/(2*D))*exp((v + (v^2 - 4*k*D + 4*l*D)^(1/2))/(2*D))*(v^2 - 4*k*D + 4*l*D)^(1/2)
)/(
    4*(k - l)*(v^2 - 4*k*D + 4*l*D)^(1/2)
    )

-(
    D*v^2*exp(l*t)*exp(g) + 2*k*v^2*exp(l*t)*exp(g) - 2*l*v^2*exp(l*t)*exp(g) - D*v^2*exp(f)*exp(l*t) - D*exp(l*t)*exp(g)*(p) - 2*k*v^2*exp(f)*exp(l*t) + 2*l*v^2*exp(f)*exp(l*t) + D*exp(f)*exp(l*t)*(p) + 2*k*v*exp(l*t)*exp(g)*sqrt(p) - 2*l*v*exp(l*t)*exp(g)*sqrt(p) + 2*k*v*exp(f)*exp(l*t)*sqrt(p) - 2*l*v*exp(f)*exp(l*t)*sqrt(p) - 4*R*k*v*exp(f)*exp(g)*sqrt(p) + 4*R*l*v*exp(f)*exp(g)*sqrt(p)
)/(
    4*(k - l)*sqrt(p)
)

-(
    D*v^2*exp(l*t)*exp(g) 
    + 2*k*v^2*exp(l*t)*exp(g) 
    - 2*l*v^2*exp(l*t)*exp(g) 
    - D*v^2*exp(f)*exp(l*t) 
    - D*exp(l*t)*exp(g)*(p) 
    - 2*k*v^2*exp(f)*exp(l*t) 
    + 2*l*v^2*exp(f)*exp(l*t) 
    + D*exp(f)*exp(l*t)*(p) 
    + 2*k*v*exp(l*t)*exp(g)*sqrt(p) 
    - 2*l*v*exp(l*t)*exp(g)*sqrt(p) 
    + 2*k*v*exp(f)*exp(l*t)*sqrt(p) 
    - 2*l*v*exp(f)*exp(l*t)*sqrt(p) 
    - 4*R*k*v*exp(f)*exp(g)*sqrt(p) 
    + 4*R*l*v*exp(f)*exp(g)*sqrt(p)
)/(
    4*(k - l)*sqrt(p)
)