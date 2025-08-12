close all; clear; clc;

plot_dist    = 1;
plot_coeff   = 0;
degree_approx = 2;

% Difficulty parameter: typical root-finding conditioning scales like 1/sigma.
sigma = 1;

p = 2;                     % polynomial degree in the example

% Subregion widths
hVals = logspace(2, -2, 40);

num_Q        = 1;
num_root_loc = 50;
[~, num_h]   = size(hVals);
data_size    = [num_h, num_Q, num_root_loc];

% Metrics we record
distVals                   = nan(data_size);   % |z - z_*|
predictedDistVals          = nan(data_size);
coordinateMatrixMagnitude  = nan(data_size);   % max ||A_i||_2 over slices

f = waitbar(0, 'Please wait...');

for i_Q = 1:num_Q
    % Random orthogonal matrix (external helper)
    Q = rand_orth_mat(3);
    perturb = 0*rand(1,3);

    % Noferini–Townsend-style example
    f1 = @(x1,x2,x3) x1.^p + sigma .* (Q(1,1).*x1 + Q(1,2).*x2 + Q(1,3).*x3) + dot([x1.^4, x2.^4, x3.^4], perturb);
    f2 = @(x1,x2,x3) x2.^p + sigma .* (Q(2,1).*x1 + Q(2,2).*x2 + Q(2,3).*x3) + dot([x1.^4, x2.^4, x3.^4], perturb);
    f3 = @(x1,x2,x3) x3.^p + sigma .* (Q(3,1).*x1 + Q(3,2).*x2 + Q(3,3).*x3) + dot([x1.^4, x2.^4, x3.^4], perturb);

    % Linear but poor conditioning dependence
    % et = 10^(-7);
    C = 100;
    ep1 = 10^(-7);
    ep2 = ep1;
    ep3 = ep1;
    % del = 10^(-7);
    % f1 = @(x1,x2,x3) 1 * ((1/3 + ep1) * x1 + 1/3 * x2 + (1/3 - ep1) * x3);
    % f2 = @(x1,x2,x3) 1 * ((1/3 + ep2) * x1 + (1/3 - ep2) * x2 + 1/3 * x3);
    % f3 = @(x1,x2,x3) 1 * ((1/3) * x1 + (1/3 - ep3) * x2 + (1/3 + ep3) * x3);

    for i_root_loc = 1:num_root_loc
        completion = ((i_Q-1) * num_root_loc + (i_root_loc-1)) / (num_Q * num_root_loc);
        f = waitbar(completion, f, sprintf("Trying with orthogonal matrix #%d | root loc #%d", i_Q, i_root_loc));

        fprintf("root_loc number %d\n", i_root_loc);

        % Choose a (small) translated root location
        expected = (2*rand(1,3) - 1)/100;

        % Chebfun3 objects for translated problem
        p1 = chebfun3(@(x1,x2,x3) f1(x1-expected(1), x2-expected(2), x3-expected(3)));
        p2 = chebfun3(@(x1,x2,x3) f2(x1-expected(1), x2-expected(2), x3-expected(3)));
        p3 = chebfun3(@(x1,x2,x3) f3(x1-expected(1), x2-expected(2), x3-expected(3)));

        % Conditioning at the root
        J_func = jac(p1,p2,p3);
        J      = J_func(expected(1), expected(2), expected(3));
        condJinv = norm(inv(J));
        cond_eig = 1 / abs(det(J));

        for k = 1:numel(hVals)
            h = hVals(k);

            % Asymmetric cube around the expected root
            a = [expected(1) - h/3,   expected(2) + h/3,   expected(3) - h/4];
            b = [expected(1) + 2*h/3, expected(2) - 2*h/3, expected(3) + 3*h/4];

            cube_scale = (b - a)/2;
            cube_shift = (b + a)/2;

            remap = @(x,idx) cube_scale(idx).*x + cube_shift(idx);

            % Scale the remapped functions to O(1) on the box
            v = [-1 1];
            [X,Y,Z] = ndgrid(v,v,v);
            x = remap(X(:),1);  y = remap(Y(:),2);  z = remap(Z(:),3);
            c1 = max(abs(p1(x,y,z)));
            c2 = max(abs(p2(x,y,z)));
            c3 = max(abs(p3(x,y,z)));

            p1_u = @(x1,x2,x3) p1(remap(x1,1), remap(x2,2), remap(x3,3)) / c1;
            p2_u = @(x1,x2,x3) p2(remap(x1,1), remap(x2,2), remap(x3,3)) / c2;
            p3_u = @(x1,x2,x3) p3(remap(x1,1), remap(x2,2), remap(x3,3)) / c3;

            % Solve in unit cube and gather block data
            [roots_z_unit, R, V, W, approx_err, eig_err] = roots_z(p1_u, p2_u, p3_u, [-1 -1 -1], [1 1 1], degree_approx);

            n_z     = size(R,3);
            Ai_norm = arrayfun(@(t) norm(R(:,:,t),2), 1:n_z);
            max_A2  = max(Ai_norm);
            coordinateMatrixMagnitude(k, i_Q, i_root_loc) = max_A2;

            if isempty(roots_z_unit)
                distVals(k, i_Q, i_root_loc) = NaN;
            else
                % roots_z_unit is a column of z-values (unit coords); map them to z
                roots_z_remapped = remap(roots_z_unit(:,1), 3);
                d = abs(roots_z_remapped - expected(3));
                [min_dist, min_idx] = min(d);
                distVals(k, i_Q, i_root_loc) = min_dist;

                % crude forward-error proxy from block size & scaling
                v_sel = V(min_idx);
                w_sel = W(min_idx);
                vfac = norm(v_sel,2);
                wfac = norm(w_sel,2);

                % The size of the resultant matrix
                N_xy = size(R, 1);
                
                error_resultant = h * (4 * vfac * wfac * (c1 * c2 * c3) / (h^3 * abs(det(J))) * (32 * approx_err * N_xy + eig_err(min_idx)));
                error_floor = 1e-15 * condJinv;
                err_est_hat = max(error_resultant, error_floor);

                % This error estimate is only valid when the approximation
                % error is not super high. Here it is fixed, but in
                % practice it would actually depend dynamically on the
                % problem.

                predictedDistVals(k, i_Q, i_root_loc) = err_est_hat;
            end
        end
    end
end

delete(f);

%% Plot: box width vs distance to root (z-component)
if plot_dist
    figure;

    h1 = loglog(hVals, mean(distVals,[2,3],"omitnan"), 'o-','LineWidth',1.2,'MarkerSize',6, ...
                'DisplayName','\(\mathrm{observed\ error}\)');
    hold on;
    h2 = loglog(hVals, mean(predictedDistVals,[2,3],"omitnan"), 'o-','LineWidth',1.2,'MarkerSize',6, ...
                'DisplayName','\(\mathrm{predicted\ error}\)');

    leg = legend([h1 h2], {'\(\mathrm{observed\ error}\)','\(\mathrm{predicted\ error}\)'}, ...
                 'Interpreter','latex','Location','best');
    leg.AutoUpdate = 'off';
    grid on;
    xlabel('box width \(h\)','Interpreter','latex');
    ylabel('error in \(z\)-component','Interpreter','latex');
    title(sprintf('Effect of shrinking the domain on error (σ = %.0e)', sigma), 'Interpreter','latex');

    yline(condJinv * 1e-15,'r--', ...
          'Interpreter','latex', ...
          'Label','\(\mathrm{err} \approx u\cdot\mathrm{cond}\)', ...
          'LabelOrientation','horizontal', ...
          'LabelVerticalAlignment','bottom', ...
          'LabelHorizontalAlignment','center');

    yline(cond_eig * 1e-15,'r--', ...
          'Interpreter','latex', ...
          'Label','\(\mathrm{err} \approx u\cdot\mathrm{eigcond}\)', ...
          'LabelOrientation','horizontal', ...
          'LabelVerticalAlignment','bottom', ...
          'LabelHorizontalAlignment','center');
end

%% Optional: width vs. max ||A_i||_2 (kept for convenience)
if plot_coeff
    figure;
    loglog(hVals, coordinateMatrixMagnitude, 'o-','LineWidth',1.2,'MarkerSize',6);
    grid on;
    xlabel('box width  \(h\)','Interpreter','latex');
    ylabel('\(\max ||A_i||_2\)','Interpreter','latex');
    title('Effect of shrinking the domain on \(\max ||A_i||_2\)', 'Interpreter','latex');
end
