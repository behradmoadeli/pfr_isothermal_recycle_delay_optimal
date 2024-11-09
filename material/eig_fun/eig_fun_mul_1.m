% eig_fun_mul_1
function result = eig_fun_mul_1(x, args)
    par = args{1};
    
    if length(args) > 1
        l = args{2};
    else
        l = [args{2}, args{2}];
    end

    phi = eig_fun_1(x, par, l(1));
    psi = eig_fun_2(x, par, l(1));

    phi_star = eig_fun_adj_1(x, par, l(2));
    psi_star = eig_fun_adj_2(x, par, l(2));

    result = phi * phi_star + psi * psi_star;
end