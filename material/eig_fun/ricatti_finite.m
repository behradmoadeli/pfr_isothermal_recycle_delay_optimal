clear all; clc

A = [-11.4,  6.4,  0,  0,  0,  3.9,  0,  0,  0,  0;
      5.2, -4.9,  1.2,  0,  0,  0,  0,  0,  0,  0;
      0,  5.2, -4.9,  1.2,  0,  0,  0,  0,  0,  0;
      0,  0,  5.2, -4.9,  1.2,  0,  0,  0,  0,  0;
      0,  0,  0,  6.4, -4.9,  0,  0,  0,  0,  0;
      0,  0,  0,  0,  0, -5,  5,  0,  0,  0;
      0,  0,  0,  0,  0,  0, -5,  5,  0,  0;
      0,  0,  0,  0,  0,  0,  0, -5,  5,  0;
      0,  0,  0,  0,  5,  0,  0,  0, -5,  0;
      0,  0,  0,  6.4, -4.9,  0,  0,  0,  0,  0];

E = eig (A);

B = [2.6, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ]';

rank(ctrb(A,B));

% [P,K,L,info] = icare(A,B,1,1,0,eye(10),0)

[k, p, l] = lqr(A, B, eye(10), 1);

sys_ol = ss(A,B,eye(length(A)), zeros(length(A),1));

t = 0:0.1:10;
u = zeros(length(t),1)';
x0 = ones(length(A),1);
lsim(sys_ol,u,t,x0)
grid on


A_cl = A - B * k;

sys_cl = ss(A_cl,B,eye(length(A)), zeros(length(A),1));

t = 0:0.1:10;
u = zeros(length(t),1)';
x0 = ones(length(A),1);
lsim(sys_cl,u,t,x0)
grid on