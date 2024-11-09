% Define symbolic variables
syms T11 T12 T33 T21 T22 v D R I1 I2 I3 U

% Define the matrix A
A = [T11, T12, -T33;
     T21, T22, 0;
     v, D, v*(1-R)];

% Define the vector b
b = [I1-I3; I2; -v*R*U];

% Form the augmented matrix [A|b]
AugmentedMatrix = [A b];

% Transform the augmented matrix to row echelon form using rref
RowEchelonForm = rref(AugmentedMatrix);

% Display the result
disp('The augmented matrix in row echelon form is:');
disp(RowEchelonForm);

b_prime = RowEchelonForm(:,4);

b_prime_simplified = simplify(b_prime)
