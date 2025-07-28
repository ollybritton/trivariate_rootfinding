close all; clear; clc;

plot_dist = 1;
plot_coeff = 0;
degree_approx = 4;

% Controls difficulty of problem, by default the condition number of the
% root finding problem is 1/sigma.
sigma = 1e-6;

% Widths of subregions considered
hVals = logspace( 0, -10, 20 );

num_Q = 1;
num_root_loc = 1;
[~, num_h] = size(hVals);

data_size = [num_h num_Q num_root_loc];

% Distances of z-components to expected root
distVals = nan(data_size);
testDistVals = nan(data_size);
predictedDistVals = nan(data_size);

% Maximum magnitude of coordinate matrices
coordinateMatrixMagnitude = nan(data_size);
predictedCoordinateMatrixMagnitude = nan(data_size);

f = waitbar(0, 'Please wait...');

for i_Q = 1:num_Q
    f = waitbar(i_Q/num_Q, f, sprintf("Trying with orthogonal matrix #%d", i_Q));

    % Random orthogonal matrix
    Q = rand_orth_mat(3);
    peturb = rand(1,3);
        
    % Devastating example as in Noferini-Townsend
    f1 = @(x1,x2,x3) x1.^2 + sigma .* (Q(1,1).*x1 + Q(1,2).*x2 + Q(1,3).*x3) + dot([x1.^4 x2.^4 x3.^4], peturb);
    f2 = @(x1,x2,x3) x2.^2 + sigma .* (Q(2,1).*x1 + Q(2,2).*x2 + Q(2,3).*x3) + dot([x1.^4 x2.^4 x3.^4], peturb);
    f3 = @(x1,x2,x3) x3.^2 + sigma .* (Q(3,1).*x1 + Q(3,2).*x2 + Q(3,3).*x3) + dot([x1.^4 x2.^4 x3.^4], peturb);
    
    for i_root_loc = 1:num_root_loc
        fprintf("root_loc number %d\n", i_root_loc);
        % Chosen location of the root (by default it is the origin, but that might
        % be especially easy to find for some reason).
        expected = 2*rand(1,3) - 1;
        
        % Chebfun3 objects corresponding to translated problem
        p1 = chebfun3(@(x1,x2,x3) f1(x1-expected(1), x2-expected(2), x3-expected(3)));
        p2 = chebfun3(@(x1,x2,x3) f2(x1-expected(1), x2-expected(2), x3-expected(3)));
        p3 = chebfun3(@(x1,x2,x3) f3(x1-expected(1), x2-expected(2), x3-expected(3)));

        % Condition number of the rootfinding problem
        J_func = jac(p1,p2,p3);
        J = J_func(expected(1), expected(2), expected(3));
        J_inv = inv(J);
        cond = norm(J_inv);
        err_estimate = 1e-15 * cond;

        % Another test where I multiply the inputs by a matrix
        preconditioner = J_inv;
        p1_q = preconditioner(1,1) * p1 + preconditioner(1,2) * p2 + preconditioner(1,3) * p3;
        p2_q = preconditioner(2,1) * p1 + preconditioner(2,2) * p2 + preconditioner(2,3) * p3;
        p3_q = preconditioner(3,1) * p1 + preconditioner(3,2) * p2 + preconditioner(3,3) * p3;
        
        % Condition number of the eigenproblem
        cond_eig = 1/abs(det(J));
        err_estimate_eig = 1e-15 * cond_eig;
        
        for k = 1:numel(hVals)
            h = hVals(k);
        
            % Cube around expected root, not placing expected root at the origin as
            % in practice the subregion won't be exactly centered on the root
            a = [expected(1) - h/3, expected(2) + h/3, expected(3) - h/4];
            b = [expected(1) + 2*h/3, expected(2) - 2*h/3, expected(3) + 3*h/4];
        
            cube_scale = (b - a)/2;
            cube_shift = (b + a)/2;
            
            remap = @(x,idx) cube_scale(idx).*x + cube_shift(idx);
                
            % Scaling the remapped functions is important! If they are not scaled,
            % then roots might actually be missed.
            v = [-1 1];
            [X,Y,Z] = ndgrid(v,v,v);
            x = remap(X(:),1);  y = remap(Y(:),2);  z = remap(Z(:),3);
            
            c1 = max(abs(p1(x,y,z)));
            c2 = max(abs(p2(x,y,z)));
            c3 = max(abs(p3(x,y,z)));
            
            p1_u = @(x1,x2,x3) p1(remap(x1,1), remap(x2,2), remap(x3,3)) / c1;
            p2_u = @(x1,x2,x3) p2(remap(x1,1), remap(x2,2), remap(x3,3)) / c2;
            p3_u = @(x1,x2,x3) p3(remap(x1,1), remap(x2,2), remap(x3,3)) / c3;
            
            p1_u_c = chebfun3(p1_u, [degree_approx degree_approx degree_approx]);
            p2_u_c = chebfun3(p2_u, [degree_approx degree_approx degree_approx]);
            p3_u_c = chebfun3(p3_u, [degree_approx degree_approx degree_approx]);

            J_hat_func = jac(p1_u_c, p2_u_c, p3_u_c);
            J_hat = J_hat_func(remap(expected(1),1), remap(expected(2),2), remap(expected(3),3));
            cond_hat = 1/abs(det(J_hat));
            err_estimate_hat = 1e-15 * cond_hat * h;

            scale_factor = (abs(det(J_hat)) / norm(J,2)).^(1/3);

            p1_u = @(x1,x2,x3) scale_factor * p1(remap(x1,1), remap(x2,2), remap(x3,3)) / c1;
            p2_u = @(x1,x2,x3) scale_factor * p2(remap(x1,1), remap(x2,2), remap(x3,3)) / c2;
            p3_u = @(x1,x2,x3) scale_factor * p3(remap(x1,1), remap(x2,2), remap(x3,3)) / c3;


            % Don't know why this is so small
        
            % Locate all roots inside this cube, in the "unit" coordinates.
            [roots_z_unit, R, V, W, approx_err] = roots_z(p1_u, p2_u, p3_u, [-1 -1 -1], [1 1 1], degree_approx);
            [roots_z_preconditioned, ~, ~, ~] = roots_z(p1_q, p2_q, p3_q, [-1 -1 -1], [1 1 1], degree_approx);
        
            % Calculate maximum magintude over coordinate matrices 
            n_z = size(R,3);
            Ai_norms = arrayfun(@(k) norm(R(:,:,k),2), 1:n_z);
            max_A2 = max(Ai_norms);
            coordinateMatrixMagnitude(k, i_Q, i_root_loc) = max_A2;
            predictedCoordinateMatrixMagnitude(k, i_Q, i_root_loc) = 1;%h^2;
        
            if isempty(roots_z_unit)
                distVals(k, i_Q, i_root_loc) = NaN;
                warning('No roots found for h = %.3g – recording NaN', h);
            else
                roots_z_remapped = remap(roots_z_unit(:,1),3);
                d = abs(roots_z_remapped - expected(3));
                [min_dist,min_dist_index] = min(d);
                distVals(k, i_Q, i_root_loc) = min_dist;
        
                % Rough theoretical predictions
                predictedDistVals(k, i_Q, i_root_loc) = approx_err;
            end

            if isempty(roots_z_preconditioned)
                testDistVals(k, i_Q, i_root_loc) = NaN;
                warning('No roots found for h = %.3g – recording NaN', h);
            else
                d = abs(roots_z_preconditioned - expected(3));
                [min_dist,min_dist_index] = min(d);
                testDistVals(k, i_Q, i_root_loc) = min_dist;       
            end            
        end
    end
end

delete(f);

%% Plot for width vs distance to root
if plot_dist
    figure;
    
    loglog(hVals, mean(distVals, [2, 3], "omitnan"), 'o-','LineWidth',1.2,'MarkerSize',6);
    hold on;
    loglog(hVals, mean(predictedDistVals, [2, 3], "omitnan"), 'o-','LineWidth',1.2,'MarkerSize',6);
    % loglog(hVals, mean(testDistVals, [2, 3], "omitnan"), 'o-','LineWidth',1.2,'MarkerSize',6);
    grid on;
    xlabel('box width \(h\)','Interpreter','latex');
    ylabel('error in \(z\)-component','Interpreter','latex');
    
    title(sprintf(['Effect of shrinking the domain on error ' '(σ = %.0e)'], sigma), 'Interpreter','latex');
    
    % line showing how small the domain should be for good error
    xline(1/cond,'r--', ...                       
          'Interpreter','latex', ...
          'Label','\(h \approx 1/\mathrm{cond}\)', ...
          'LabelOrientation','horizontal', ...
          'LabelVerticalAlignment','bottom', ...
          'LabelHorizontalAlignment','center');
    
    yline(cond * 1e-15,'r--', ...                       
          'Interpreter','latex', ...
          'Label','\(\mathrm{err} \approx u\cdot\mathrm{cond}\)', ...
          'LabelOrientation','horizontal', ...
          'LabelVerticalAlignment','bottom', ...
          'LabelHorizontalAlignment','center');
    
    yline(cond_eig * 1e-15,'r--', ...                       
          'Interpreter','latex', ...
          'Label','\(\mathrm{err} \approx u\cdot\mathrm{eigcond} \approx u \cdot \mathrm{cond}^3\)', ...
          'LabelOrientation','horizontal', ...
          'LabelVerticalAlignment','bottom', ...
          'LabelHorizontalAlignment','center');
end
    
if plot_coeff
    %% Plot for width vs modulus of coordinate matrices
    figure;
    
    loglog(hVals, coordinateMatrixMagnitude, 'o-','LineWidth',1.2,'MarkerSize',6);
    hold on;
    loglog(hVals, predictedCoordinateMatrixMagnitude, 'o-','LineWidth',1.2,'MarkerSize',6);
    grid on;
    xlabel('box width  \(h\)','Interpreter','latex');
    ylabel('\(\max ||A_i||_2\)','Interpreter','latex');
    
    title('Effect of shrinking the domain on \(\max ||A_i||_2\)', 'Interpreter','latex');
end