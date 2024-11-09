clear all; clc;
options = optimoptions('fsolve','StepTolerance',1e-18,'FunctionTolerance',1e-4,'Display','off');

% Define the range of initial guesses
x1_range = linspace(-20, 5, 1000); % Define the range for x1
x2_range = linspace(0, 20, 200); % Define the range for x2

% Store solutions
all_solutions = zeros(0,2); % Initialize an empty matrix to store solutions


% Loop through initial guesses
for i = 1:length(x1_range)
    for j = 1:length(x2_range)
        x0 = [x1_range(i), x2_range(j)]; % Set the initial guess
        
        % Call fsolve
        [x, fval, exitflag, output] = fsolve(@char_eq, x0, options);
        
        % Check if fsolve converged successfully
        if exitflag > 0
            % Append solution to the matrix
            all_solutions = [all_solutions; x];
            all_solutions = [all_solutions; [x(1), -x(2)]];

        end
    end
end

% Plot the solutions
scatter(all_solutions(:,1), all_solutions(:,2), 'filled');
hold on;

% Add grid
grid on;

% Add x=0 and y=0 lines
xline(0, 'k'); % Black solid line for x=0
yline(0, 'k'); % Black solid line for y=0

xlabel('Re');
ylabel('Im');
title('Solution Points');
