% eig_fun_adj_2
function psi_star = eig_fun_adj_2(x,  l)
    args = struct('k', 1.5, 'D', 0.05, 'v', 1, 'tau', 0.8, 'R', 0.6);
    k = args.k;
    v = args.v;
    D = args.D;
    t = args.tau;
    R = args.R;

    p = v^2 - 4 * D * (k - l);
    p_sqrt = sqrt(p);

    r = [-(v + p_sqrt) / (2 * D), -(v - p_sqrt) / (2 * D)];

    c = 1 - r(2) / r(1);

    psi_star = c * exp(-t * l * x);
end