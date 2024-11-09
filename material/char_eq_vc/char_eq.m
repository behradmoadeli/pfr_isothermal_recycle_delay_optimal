% Define the function representing the system of equations

function F = char_eq(l)
    % Extract variables

    l_r = l(1);
    l_i = l(2);
    l = l_r + l_i * 1i;

    % Define parameters

    D = [0.025, 0.015];

    k = [
        1e-5, 8e-5;
        5e-6, -0.85
        ];

    v = 0.1;

    t = 0.8;
    
    R = 0.3;

    %Define matrices
    
    L = [
        0, 1, 0, 0, 0, 0;
        (l-k(1,1))/D(1), v/D(1), -k(1,2)/D(1), 0, 0, 0;
        0, 0, 0, 1, 0, 0;
        -k(2,1)/D(2), 0, (l-k(2,2))/D(2), v/D(2), 0, 0;
        0, 0, 0, 0, t*l, 0;
        0, 0, 0, 0, 0, t*l;
        ];

    Q = expm(L);

    row_1 = [-v, D(1), 0, 0, R*v, 0];
    row_2 = [0, 0, -v, D(2), 0, R*v];
    row_3 = Q(2, :);
    row_4 = Q(4, :);
    row_5 = Q(1,:) - Q(5,:);
    row_6 = Q(3,:) - Q(6,:);
    
    A = [row_1; row_2; row_3; row_4; row_5; row_6];
    x = det(A);

    %Penalty function

    pen = exp(-1e6*(l_r^2 + l_i^2));
    
    % Define the equations
    
    F(1) = real(x) + pen;
    F(2) = imag(x);

end