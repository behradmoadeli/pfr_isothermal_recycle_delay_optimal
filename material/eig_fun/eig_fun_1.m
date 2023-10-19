function phi = eig_fun_1(x,  l)
    args = struct('k', 1.5, 'D', 0.05, 'v', 1, 'tau', 0.8, 'R', 0.6);
    k = args.k;
    v = args.v;
    D = args.D;
    t = args.tau;
    R = args.R;

    p = v^2 - 4 * D * (k - l);
    p_sqrt = sqrt(p);

    r = [(v + p_sqrt) / (2 * D), (v - p_sqrt) / (2 * D)];

    a = - (r(2) / r(1)) * exp(r(2) - r(1));

    phi = a * exp(r(1) * x) + exp(r(2) * x);
end
