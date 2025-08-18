close all; clear; clc;

plot_dist    = 1;
plot_coeff   = 0;
degree_approx = 2;

sigma = 1e-8;
del = 1e-8;

% polynomial degree in the example
p = 2;

should_precondition = false;
should_rescale = true;

% Subregion widths
hVals = logspace(-6, 4, 80);

num_Q        = 1;
num_root_loc = 50;
[~, num_h]   = size(hVals);
data_size    = [num_h, num_Q, num_root_loc];

% Metrics we record
distVals                   = nan(data_size);  % |z - z_*|
predictedDistVals          = nan(data_size);  % estimate of distVals
coordinateMatrixMagnitude  = nan(data_size);  % max ||A_i||_2 over slices

f = waitbar(0, 'Please wait...');

for i_Q = 1:num_Q
    % Random orthogonal matrix (external helper)
    Q = rand_orth_mat(3);

    % Noferini–Townsend-style example
    f1 = @(x1,x2,x3) x1.^p + sigma .* (Q(1,1).*x1 + Q(1,2).*x2 + Q(1,3).*x3);
    f2 = @(x1,x2,x3) x2.^p + sigma .* (Q(2,1).*x1 + Q(2,2).*x2 + Q(2,3).*x3);
    f3 = @(x1,x2,x3) x3.^p + sigma .* (Q(3,1).*x1 + Q(3,2).*x2 + Q(3,3).*x3);

    % Linear and poor conditioning
    % f1 = @(x1,x2,x3) (1/3 + del) * x1 + 1/3 * x2 + (1/3 - del) * x3;
    % f2 = @(x1,x2,x3) (1/3 + del) * x1 + (1/3 - del) * x2 + 1/3 * x3;
    % f3 = @(x1,x2,x3) (1/3) * x1 + (1/3 + del) * x2 + (1/3 - del) * x3;

    % Conditioning at the root
    J_func = jac(f1,f2,f3);
    J      = J_func(0, 0, 0);
    J_inv  = inv(J);
    cond_root = norm(inv(J));
    cond_eig = 1 / abs(det(J));

    % One approach is to do preconditioning here:
    % 
    % if should_precondition
    %     f1tmp = f1; f2tmp = f2; f3tmp = f3; 
    %     f1 = @(x1,x2,x3) J_inv(1,1)*f1tmp(x1,x2,x3) + J_inv(1,2)*f2tmp(x1,x2,x3) + J_inv(1,3)*f3tmp(x1,x2,x3); 
    %     f2 = @(x1,x2,x3) J_inv(2,1)*f1tmp(x1,x2,x3) + J_inv(2,2)*f2tmp(x1,x2,x3) + J_inv(2,3)*f3tmp(x1,x2,x3); 
    %     f3 = @(x1,x2,x3) J_inv(3,1)*f1tmp(x1,x2,x3) + J_inv(3,2)*f2tmp(x1,x2,x3) + J_inv(3,3)*f3tmp(x1,x2,x3); 
    % end

    for i_root_loc = 1:num_root_loc
        completion = ((i_Q-1) * num_root_loc + (i_root_loc-1)) / (num_Q * num_root_loc);
        f = waitbar(completion, f, sprintf("Trying with orthogonal matrix #%d | root loc #%d", i_Q, i_root_loc));

        fprintf("root_loc number %d\n", i_root_loc);

        % Choose a (small) translated root location
        expected = (2*rand(1,3) - 1)/100;

        for k = 1:numel(hVals)
            h = hVals(k);

            % Asymmetric cube around the expected root
            a = [expected(1) - h/2, expected(2) - h/2, expected(3) - h/2];
            b = [expected(1) + h/2, expected(2) + h/2, expected(3) + h/2];

            cube_scale = (b - a)/2;
            cube_shift = (b + a)/2;

            remap = @(x,idx) cube_scale(idx).*x + cube_shift(idx);

            f1_remapped = @(x1,x2,x3) f1(remap(x1,1)-expected(1), remap(x2,2)-expected(2), remap(x3,3)-expected(3));
            f2_remapped = @(x1,x2,x3) f2(remap(x1,1)-expected(1), remap(x2,2)-expected(2), remap(x3,3)-expected(3));
            f3_remapped = @(x1,x2,x3) f3(remap(x1,1)-expected(1), remap(x2,2)-expected(2), remap(x3,3)-expected(3));

            v = [-1 1];
            [x,y,z] = ndgrid(v,v,v);
            x = x(:); y = y(:); z = z(:);
            c1 = max(abs(f1_remapped(x,y,z)));
            c2 = max(abs(f2_remapped(x,y,z)));
            c3 = max(abs(f3_remapped(x,y,z)));

            if should_rescale
                % Option 1: rescale function to have unit inf norm
                p1 = @(x1,x2,x3) f1_remapped(x1,x2,x3) / c1;
                p2 = @(x1,x2,x3) f2_remapped(x1,x2,x3) / c2;
                p3 = @(x1,x2,x3) f3_remapped(x1,x2,x3) / c3;
            else
                % Option 2: no rescaling
                p1 = f1_remapped;
                p2 = f2_remapped;
                p3 = f3_remapped;
            end

            % Condition number of transformed problem, ideally these should be close to one
            J_func_transformed = jac(p1,p2,p3);
            J_transformed      = J_func_transformed(0, 0, 0);
            J_inv_transformed = inv(J_transformed);
            cond_root_transformed = norm(J_inv_transformed);
            cond_eig_transformed = 1 / abs(det(J_transformed));

            if should_precondition
                p1tmp = p1; p2tmp = p2; p3tmp = p3;

                disp(cond_root_transformed);

                p1 = @(x1,x2,x3) J_inv_transformed(1,1)*p1tmp(x1,x2,x3) + J_inv_transformed(1,2)*p2tmp(x1,x2,x3) + J_inv_transformed(1,3)*p3tmp(x1,x2,x3); 
                p2 = @(x1,x2,x3) J_inv_transformed(2,1)*p1tmp(x1,x2,x3) + J_inv_transformed(2,2)*p2tmp(x1,x2,x3) + J_inv_transformed(2,3)*p3tmp(x1,x2,x3); 
                p3 = @(x1,x2,x3) J_inv_transformed(3,1)*p1tmp(x1,x2,x3) + J_inv_transformed(3,2)*p2tmp(x1,x2,x3) + J_inv_transformed(3,3)*p3tmp(x1,x2,x3); 

                % Recalculate condition number if we have preconditioned using
                % this information, should now both be one.
                J_func_transformed = jac(p1,p2,p3);
                J_transformed      = J_func_transformed(0, 0, 0);
                J_inv_transformed = inv(J_transformed);
                cond_root_transformed = norm(J_inv_transformed);
                cond_eig_transformed = 1 / abs(det(J_transformed));
            end

            disp(cond_eig_transformed);

            % Solve in unit cube and gather block data
            [roots_z_unit, R, V, W, approx_err, eig_err] = roots_z(p1, p2, p3, degree_approx);

            n_z     = size(R,3);
            Ai_norm = arrayfun(@(t) norm(R(:,:,t),2), 1:n_z);
            max_A2  = max(Ai_norm);
            coordinateMatrixMagnitude(k, i_Q, i_root_loc) = max_A2;

            if isempty(roots_z_unit)
                distVals(k, i_Q, i_root_loc) = NaN;
                predictedDistVals(k, i_Q, i_root_loc) = NaN;
                continue
            end

            % Map z-roots back to physical coordinates, pick closest to expected root
            roots_z_remapped = remap(roots_z_unit(:,1), 3);
            d = abs(roots_z_remapped - expected(3));
            [min_dist, min_idx] = min(d);
            distVals(k, i_Q, i_root_loc) = min_dist;

            % ---- Error analysis ----
            z_unit = roots_z_unit(min_idx, 1);     % z in unit coordinates
            nz = size(R,3);
            n  = size(R,1);

            % Chebyshev T_k(z): T0=1, T1=z, T_k=2z T_{k-1} - T_{k-2}
            T = zeros(nz,1);                 % stores T_k at index k+1
            T(1) = 1;
            if nz >= 2, T(2) = z_unit; end
            for jj = 3:nz
                T(jj) = 2*z_unit*T(jj-1) - T(jj-2);
            end

            % Chebyshev U_k(z): U0=1, U1=2z, U_k=2z U_{k-1} - U_{k-2}
            % We will need U_{k-1} for k=1..nz-1, so compute U_0..U_{nz-2}
            U = zeros(nz,1);                 % store U_k at index k+1
            U(1) = 1;                        % U_0
            if nz >= 2, U(2) = 2*z_unit; end % U_1
            for jj = 3:nz
                U(jj) = 2*z_unit*U(jj-1) - U(jj-2);
            end
            % Note: derivative uses k * U_{k-1}(z); in 1-based indexing, U(k) = U_{k-1}.

            % Assemble R(z) and R'(z)
            Rz = zeros(n);
            Rp = zeros(n);
            for kk = 0:nz-1
                Rz = Rz + R(:,:,kk+1) * T(kk+1);
                if kk >= 1
                    Rp = Rp + R(:,:,kk+1) * (kk * U(kk));  % U(kk) = U_{kk-1}
                end
            end

            % Polynomial eigenvectors: left/right nullspaces of R(z)
            [Umat, Smat, Vmat] = svd(Rz);
            v_poly = Vmat(:, end);           % right poly eigenvector
            w_poly = Umat(:, end);           % left poly eigenvector

            % Condition number of the PEP eigenvalue
            denom = abs(w_poly' * (Rp * v_poly));
            eig_cond = (norm(v_poly,2) * norm(w_poly,2)) / denom;

            % Backward pieces: from linearisation residual and interpolation
            eig_back   = eig_err(min_idx);       % ||ΔR_eig(z)||_2 (unit coords) per-eigenpair
            build_back = 32 * nz * approx_err;   % use nz, not N_xy

            % First-order forward error for physical z
            error_resultant = (h/2) * eig_cond * (eig_back + build_back);
            error_floor     = eps * cond_root;
            err_est_hat     = max(error_resultant, error_floor);

            predictedDistVals(k, i_Q, i_root_loc) = err_est_hat;
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
    title(sprintf('Effect of shrinking the domain on error (σ = %.0e, δ = %.0e)', sigma, del), 'Interpreter','latex');

    yline(cond_root * 1e-15,'r--', ...
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
    loglog(hVals, mean(distVals,[2,3],"omitnan"), 'o-','LineWidth',1.2,'MarkerSize',6);
    grid on;
    xlabel('box width  \(h\)','Interpreter','latex');
    ylabel('\(\max ||A_i||_2\)','Interpreter','latex');
    title('Effect of shrinking the domain on \(\max ||A_i||_2\)', 'Interpreter','latex');
end
