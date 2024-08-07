clear all; clc

% Load the .mat file
data = load('data.mat');

% The matrices will be available as fields in the struct 'data'
A_cl = data.A_cl;
C = data.C;

% Ensure C is dense if needed
if issparse(C)
    C = full(C);
end

% Calculate the eigenvalues of A_cl
eig_A_cl = eig(A_cl);

% Find the negative eigenvalues (consider only the real part less than -1e-6)
negative_eigenvalues = eig_A_cl(real(eig_A_cl) < -1e-6);

% Extract the largest negative real part eigenvalue
[~, idx] = max(real(negative_eigenvalues));
max_real_part = real(negative_eigenvalues(idx));
max_real_part_new = max_real_part * 3;

% Construct the new eigenvalues with the adjusted real parts and the same imaginary parts
eig_A_est = eig_A_cl;
for i = 1:length(eig_A_cl)
    if real(eig_A_cl(i)) > max_real_part_new && real(eig_A_cl(i)) < -1e-6
        eig_A_est(i) = max_real_part_new + 1i * imag(eig_A_cl(i));
    end
end

% Compute the observer gain L
L = place(A_cl', C', eig_A_est)';

% Save the observer gain
save('L_data.mat', 'L');
disp('Observer gain saved as L_data.mat');

% Plot the elements of L on a scatter plot
figure;
scatter(1:numel(L), L(:), 'filled');
xlabel('Element Index');
ylabel('Value');
title('Scatter Plot of Observer Gain Elements');
grid on;
