syms v R D k l t
A = [0 1 0;
    -(k-l)/D, v/D, 0;
    0 0 l*t];
q = reshape(expm(A),[1,9]);
% g = D * q(4) * q(9) + v * (q(5) * q(9) + R * (q(5) - q(2) * q(4)));

A_bar = [-v, D, R*v;
    q(4), q(5), 0;
    q(1), q(2), -q(9);
    ];

k = 1.5;
D = 0.05;
v = 1;
t = 0.8;
R = 0.6;
l = 0.5459;

det(A_bar)
eval(det(A_bar))
