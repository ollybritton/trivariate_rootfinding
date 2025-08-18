function J = jac(f1, f2, f3)
% JAC   Computes the Jacobian matrix for a system of three functions.
%   J = JAC(F1, F2, F3) returns a function handle J(x,y,z) that computes
%   the 3x3 Jacobian matrix.
%
%   The inputs F1, F2, F3 can be either chebfun3 objects or anonymous
%   function handles.
%
%   - If the inputs are chebfuns, it uses symbolic differentiation via GRAD.
%   - If the inputs are function handles, it uses numerical finite differences.

    % Check the type of the first input to decide the method
    if isa(f1, 'chebfun')
        %% --- Chebfun Path (Symbolic Differentiation) ---
        
        [diffF1_1, diffF1_2, diffF1_3] = grad(f1);
        [diffF2_1, diffF2_2, diffF2_3] = grad(f2);
        [diffF3_1, diffF3_2, diffF3_3] = grad(f3);

        % Assemble the Jacobian using feval for chebfuns
        J = @(x,y,z) [feval(diffF1_1, x, y, z),  feval(diffF1_2, x, y, z),  feval(diffF1_3, x, y, z);
                        feval(diffF2_1, x, y, z),  feval(diffF2_2, x, y, z),  feval(diffF2_3, x, y, z);
                        feval(diffF3_1, x, y, z),  feval(diffF3_2, x, y, z),  feval(diffF3_3, x, y, z)];

    elseif isa(f1, 'function_handle')
        %% --- Function Handle Path (Numerical Differentiation) ---

        % Define a small step size for the finite difference approximation
        h = 1e-7;

        % Create anonymous functions for each partial derivative using central differences
        diffF1_1 = @(x,y,z) (f1(x+h, y, z) - f1(x-h, y, z)) / (2*h);
        diffF1_2 = @(x,y,z) (f1(x, y+h, z) - f1(x, y-h, z)) / (2*h);
        diffF1_3 = @(x,y,z) (f1(x, y, z+h) - f1(x, y, z-h)) / (2*h);

        diffF2_1 = @(x,y,z) (f2(x+h, y, z) - f2(x-h, y, z)) / (2*h);
        diffF2_2 = @(x,y,z) (f2(x, y+h, z) - f2(x, y-h, z)) / (2*h);
        diffF2_3 = @(x,y,z) (f2(x, y, z+h) - f2(x, y, z-h)) / (2*h);

        diffF3_1 = @(x,y,z) (f3(x+h, y, z) - f3(x-h, y, z)) / (2*h);
        diffF3_2 = @(x,y,z) (f3(x, y+h, z) - f3(x, y-h, z)) / (2*h);
        diffF3_3 = @(x,y,z) (f3(x, y, z+h) - f3(x, y, z-h)) / (2*h);

        % Assemble the Jacobian by directly calling the anonymous functions
        J = @(x,y,z) [diffF1_1(x, y, z),  diffF1_2(x, y, z),  diffF1_3(x, y, z);
                        diffF2_1(x, y, z),  diffF2_2(x, y, z),  diffF2_3(x, y, z);
                        diffF3_1(x, y, z),  diffF3_2(x, y, z),  diffF3_3(x, y, z)];

    else
        error('Inputs must be either chebfun objects or function handles.');
    end
end