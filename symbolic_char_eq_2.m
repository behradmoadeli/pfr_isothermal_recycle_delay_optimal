syms v R D k l t
A = [0 1 0;
    (l-k)/D, v/D, 0;
    0 0 l*t];
q = reshape(expm(A),[1,9]);
g = D * q(4) * q(9) + v * (q(5) * q(9) + R * (q(5) - q(2) * q(4)));

A_bar = [-v, D, R*v;
    q(1), q(2), q(9);
    q(4), q(5), 0];

det(A_bar)

