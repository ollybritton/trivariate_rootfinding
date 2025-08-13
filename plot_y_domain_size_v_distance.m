close all; clear; clc;

plot_dist    = 1;
plot_coeff   = 0;
degree_approx = 2;

% Difficulty parameter: typical root-finding conditioning scales like 1/sigma.
sigma = 1e-3;

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

    % % Linear but poor conditioning dependence
    % % et = 10^(-7);
    % ep1 = 10^(-8);
    % ep2 = ep1;
    % ep3 = ep1;
    % % del = 10^(-7);
    % f1 = @(x1,x2,x3) (1 * ((1/3 + ep1) * x1 + 1/3 * x2 + (1/3 - ep1) * x3));
    % f2 = @(x1,x2,x3) (1 * ((1/3 + ep2) * x1 + (1/3 - ep2) * x2 + 1/3 * x3));
    % f3 = @(x1,x2,x3) (1 * ((1/3) * x1 + (1/3 - ep3) * x2 + (1/3 + ep3) * x3));

    for i_root_loc = 1:num_root_loc
        completion = ((i_Q-1) * num_root_loc + (i_root_loc-1)) / (num_Q * num_root_loc);
        f = waitbar(completion, f, sprintf("Trying with orthogonal matrix #%d | root loc #%d", i_Q, i_root_loc));

        fprintf("root_loc number %d\n", i_root_loc);

        % Choose a (small) translated root location
        expected = (2*rand(1,3) - 1)/100;

        % Chebfun3 objects for translated problem
        f1_translated = chebfun3(@(x1,x2,x3) f1(x1-expected(1), x2-expected(2), x3-expected(3)));
        f2_translated = chebfun3(@(x1,x2,x3) f2(x1-expected(1), x2-expected(2), x3-expected(3)));
        f3_translated = chebfun3(@(x1,x2,x3) f3(x1-expected(1), x2-expected(2), x3-expected(3)));

        % Conditioning at the root
        J_func = jac(f1_translated,f2_translated,f3_translated);
        J      = J_func(expected(1), expected(2), expected(3));
        J_inv  = inv(J);
        cond_root = norm(inv(J));
        cond_eig = 1 / abs(det(J));

        best_guess = nan;

        for k = 1:numel(hVals)
            h = hVals(k);

            if isnan(best_guess)
                p1 = f1_translated;
                p2 = f2_translated;
                p3 = f3_translated;
            else
                % Simulate uncertainty
                J_func_approx = jac(f1_translated,f2_translated,f3_translated);
                J_approx      = J_func(best_guess(1), best_guess(2), best_guess(3));
                J_inv_approx  = inv(J);

                % Multiply by inverse of Jacobian at the root (this will not be
                % known in practice)
                p1 = chebfun3(@(x1,x2,x3) J_inv_approx(1,1)*f1_translated(x1,x2,x3) + J_inv_approx(1,2)*f2_translated(x1,x2,x3) + J_inv_approx(1,3)*f3_translated(x1,x2,x3), [2 2 2]);
                p2 = chebfun3(@(x1,x2,x3) J_inv_approx(2,1)*f1_translated(x1,x2,x3) + J_inv_approx(2,2)*f2_translated(x1,x2,x3) + J_inv_approx(2,3)*f3_translated(x1,x2,x3), [2 2 2]);
                p3 = chebfun3(@(x1,x2,x3) J_inv_approx(3,1)*f1_translated(x1,x2,x3) + J_inv_approx(3,2)*f2_translated(x1,x2,x3) + J_inv_approx(3,3)*f3_translated(x1,x2,x3), [2 2 2]);
        
                % New condition numbers, these should be close to one
                J_func_translated = jac(p1,p2,p3);
                J_translated      = J_func_translated(expected(1), expected(2), expected(3));
                cond_root_translated = norm(inv(J_translated));
                cond_eig_translated = 1 / abs(det(J_translated));
            end

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
                predictedDistVals(k, i_Q, i_root_loc) = NaN;
                continue
            end

            % Map z-roots back to physical coordinates, pick closest to expected root
            roots_z_remapped = remap(roots_z_unit(:,1), 3);
            d = abs(roots_z_remapped - expected(3));
            [min_dist, min_idx] = min(d);
            distVals(k, i_Q, i_root_loc) = min_dist;
            
            % Compute a best guess of the full root
            best_guess = roots_z_remapped(min_idx);

            % ---- Error predictor (PEP-based) ----
            z_unit = roots_z_unit(min_idx, 1);     % z in unit coordinates
            nz = size(R,3);
            n  = size(R,1);

            % Guard: need at least degree-1 in z for a meaningful derivative
            if nz < 2
                predictedDistVals(k, i_Q, i_root_loc) = NaN;
                continue
            end

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

            % Condition number of the PEP eigenvalue (scale-invariant)
            denom = abs(w_poly' * (Rp * v_poly));
            if denom == 0
                kappa_PEP = Inf;
            else
                kappa_PEP = (norm(v_poly,2) * norm(w_poly,2)) / denom;
            end

            % Backward pieces: from linearisation residual and interpolation
            eig_back   = eig_err(min_idx);       % ||ΔR_eig(z)||_2 (unit coords) per-eigenpair
            build_back = 32 * nz * approx_err;   % use nz, not N_xy

            % First-order forward error for physical z
            error_resultant = (h/2) * kappa_PEP * (eig_back + build_back);
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
    title(sprintf('Effect of shrinking the domain on error (σ = %.0e)', sigma), 'Interpreter','latex');

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
    loglog(hVals, coordinateMatrixMagnitude, 'o-','LineWidth',1.2,'MarkerSize',6);
    grid on;
    xlabel('box width  \(h\)','Interpreter','latex');
    ylabel('\(\max ||A_i||_2\)','Interpreter','latex');
    title('Effect of shrinking the domain on \(\max ||A_i||_2\)', 'Interpreter','latex');
end
