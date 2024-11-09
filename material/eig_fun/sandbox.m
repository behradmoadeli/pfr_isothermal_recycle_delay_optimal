clear all; clc

format long;
options = optimset('fsolve');
options.OptimalityTolerance = 1e-10; % Set the tolerance to a smaller value for higher accuracy

x_0 = 0.5;  % Initial guess
x_1 = fsolve(@(l) char_eq(l), x_0, options)

% x_1 = fsolve(char_eq, 0.5)
% x_2 = fsolve(char_eq_adj, 0.5)
% 
% char_eq(0.5)
