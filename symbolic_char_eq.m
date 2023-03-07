syms v R D k l t
A = [0 1 0;
    (l-k)/D, v/D, 0;
    0 0 t*l];
q = reshape(expm(A),[1,9]);
g = D * q(4) * q(9) + v * (q(5) * q(9) + R * (q(5) - q(2) * q(4)))