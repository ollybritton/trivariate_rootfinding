close all; clear; clc;

% --- Simulation Parameters ---
plot_dist = 1;
plot_coeff = 0;
degree_approx = 2; % Degree of polynomial approximation for the resultant
sigma = 1;         % No longer used in this linear example
p = 1;             % Polynomial degree in the example
hVals = logspace(0.5, -2, 40); % Subregion widths to test
    ep1 = 10^(-5);


% --- Loop setup ---
num_Q = 1;          % Number of random systems to average over
num_root_loc = 10;  % Number of random root locations to average over
[~, num_h] = size(hVals);
data_size = [num_h, num_Q, num_root_loc];

% --- Metrics to Record ---
distVals = nan(data_size);          % Observed error: |z - z_*|
predictedDistVals = nan(data_size); % Predicted error based on the formula
coordinateMatrixMagnitude = nan(data_size); % max ||A_i||_2 over slices (for debug)

f = waitbar(0, 'Please wait...');

for i_Q = 1:num_Q
    f1 = @(x1,x2,x3) 1e-5 * x1.^p + sigma .* (Q(1,1).*x1 + Q(1,2).*x2 + Q(1,3).*x3));
    f2 = @(x1,x2,x3) 1e-8 * x2.^p + sigma .* (Q(2,1).*x1 + Q(2,2).*x2 + Q(2,3).*x3));
    f3 = @(x1,x2,x3) 1e-8 * x3.^p + sigma .* (Q(3,1).*x1 + Q(3,2).*x2 + Q(3,3).*x3));

    for i_root_loc = 1:num_root_loc
        completion = ((i_Q-1) * num_root_loc + (i_root_loc-1)) / (num_Q * num_root_loc);
        f = waitbar(completion, f, sprintf("Averaging runs... %d%%", round(100*completion)));
        
        % Choose a random root location in the unit cube
        expected = (2*rand(1,3) - 1);
        
        % Define chebfun3 objects for the translated problem
        p1 = chebfun3(@(x1,x2,x3) f1(x1-expected(1), x2-expected(2), x3-expected(3)));
        p2 = chebfun3(@(x1,x2,x3) f2(x1-expected(1), x2-expected(2), x3-expected(3)));
        p3 = chebfun3(@(x1,x2,x3) f3(x1-expected(1), x2-expected(2), x3-expected(3)));

        % --- Pre-calculate conditioning of the physical problem ---
        J_func = jac(p1,p2,p3);
        J = J_func(expected(1), expected(2), expected(3));
        condJinv = norm(inv(J));
        detJ_phys = abs(det(J));

        for k = 1:numel(hVals)
            h = hVals(k);
            
            % Define the sub-cube (box) of width h around the root
            a = expected - h/2;
            b = expected + h/2;
            cube_scale = (b - a)/2;
            cube_shift = (b + a)/2;
            remap = @(x,idx) cube_scale(idx).*x + cube_shift(idx);

            % Scale the functions to have unit norm on the box (||f_i||_Omega)
            v = [-1 1];
            [X,Y,Z] = ndgrid(v,v,v);
            x = remap(X(:),1); y = remap(Y(:),2); z = remap(Z(:),3);
            c1 = max(abs(p1(x,y,z))); % This is ||f_1||_Omega
            c2 = max(abs(p2(x,y,z))); % This is ||f_2||_Omega
            c3 = max(abs(p3(x,y,z))); % This is ||f_3||_Omega

            % Define the transformed, rescaled functions
            p1_u = @(x1,x2,x3) p1(remap(x1,1), remap(x2,2), remap(x3,3)) / c1;
            p2_u = @(x1,x2,x3) p2(remap(x1,1), remap(x2,2), remap(x3,3)) / c2;
            p3_u = @(x1,x2,x3) p3(remap(x1,1), remap(x2,2), remap(x3,3)) / c3;
            
            % --- Solve the PEP and gather data for the error formula ---
            [roots_z_unit, R, V, W, approx_err] = roots_z(p1_u, p2_u, p3_u, [-1 -1 -1], [1 1 1], degree_approx);
            
            n_z = size(R,3);
            Ai_norm = arrayfun(@(t) norm(R(:,:,t),2), 1:n_z);
            max_A2 = max(Ai_norm); % max ||A_i||_F for the eigensolver error
            coordinateMatrixMagnitude(k, i_Q, i_root_loc) = max_A2;

            if isempty(roots_z_unit)
                distVals(k, i_Q, i_root_loc) = NaN;
            else
                % Map computed z-roots back to physical coordinates
                roots_z_remapped = remap(roots_z_unit(:,1), 3);
                % Find the root closest to the true root and record its error
                d = abs(roots_z_remapped - expected(3));
                [min_dist, min_idx] = min(d);
                distVals(k, i_Q, i_root_loc) = min_dist;

                % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
                % --- PRECISE ERROR BOUND CALCULATION ---
                % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
                
                % Get eigenvectors for the specific root that was closest.
                v_sel = V(:, min_idx);
                w_sel = W(:, min_idx);
                
                % 1. C_vec: The eigenvector norm factor
                C_vec = norm(v_sel, 2) * norm(w_sel, 2);
                
                % 2. Condition Number of the PEP (kappa_PEP)
                kappa_PEP = C_vec * (c1 * c2 * c3) / (detJ_phys * (h/2)^3);

                % 3. Total Backward Error of the PEP (epsilon_back)
                N_xy = size(R, 1);
                u = eps;
                eta = approx_err;
                C_solve = N_xy; % A pessimistic model for the solver constant
                C_interp = 1.0; % A pessimistic model for the interp constant
                epsilon_back = (C_solve * u * max_A2) + (C_interp * eta);
                
                % 4. Introduce a single calibration constant to correct for the
                % pessimistic C_solve and C_interp models.
                C_calib = 0.01; % Based on the plot, this should be ~1/100.
                
                % 5. Predicted error in physical coordinates
                error_resultant = C_calib * (h/2) * kappa_PEP * epsilon_back;
                
                % The error cannot be better than the intrinsic conditioning allows.
                error_floor = condJinv * u;
                
                predictedDistVals(k, i_Q, i_root_loc) = max(error_resultant, error_floor);
            end
        end
    end
end
delete(f);

%% Plot: box width vs distance to root (z-component)
if plot_dist
    figure;
    h1 = loglog(hVals, mean(distVals,[2,3],"omitnan"), 'o-','LineWidth',1.2,'MarkerSize',6);
    hold on;
    h2 = loglog(hVals, mean(predictedDistVals,[2,3],"omitnan"), '--','LineWidth',1.5,'MarkerSize',6);
    
    leg = legend([h1 h2], {'Observed Error','Predicted Error'}, 'Location','best');
    leg.AutoUpdate = 'off';
    grid on;
    xlabel('Box Width, \(h\)', 'Interpreter', 'latex');
    ylabel('Error in \(z\)-component', 'Interpreter', 'latex');
    title('Observed vs. Predicted Error', 'Interpreter', 'latex');
    yline(condJinv * eps, 'r--', 'Label', '\(u \cdot \kappa_{root}\)', 'Interpreter', 'latex');
    set(gca, 'FontSize', 14);
end

%% Optional: width vs. max ||A_i||_2 (kept for convenience)
if plot_coeff
    figure;
    loglog(hVals, mean(coordinateMatrixMagnitude,[2,3],"omitnan"), 'o-','LineWidth',1.2,'MarkerSize',6);
    grid on;
    xlabel('Box Width, \(h\)', 'Interpreter', 'latex');
    ylabel('\(\max \|A_i\|_2\)', 'Interpreter', 'latex');
    title('Effect of Box Width on Resultant Matrix Coefficients', 'Interpreter', 'latex');
    set(gca, 'FontSize', 14);
end