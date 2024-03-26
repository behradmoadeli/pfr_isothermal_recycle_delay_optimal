clear all; clc

A = [0.9649    0.9572    0.1419;
    0.1576    0.4854    0.4218;
    0.9706    0.8003    0.9157];

eig(A)

B = [1,0,0]';

rank(ctrb(A,B))

[P,K,L,info] = icare(A,B,1,1,0,eye(3),0)



