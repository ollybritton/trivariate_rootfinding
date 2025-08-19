close all; clear; clc;

plot_dist    = 1;
plot_coeff   = 0;
degree_approx = 2;

sigma = 1e-2;
del = 1e-7;

% polynomial degree in the example
p = 1;

should_precondition = true;
should_rescale = false;

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
    % f1 = @(x1,x2,x3) x1.^p + sigma .* (Q(1,1).*x1 + Q(1,2).*x2 + Q(1,3).*x3);
    % f2 = @(x1,x2,x3) x2.^p + sigma .* (Q(2,1).*x1 + Q(2,2).*x2 + Q(2,3).*x3);
    % f3 = @(x1,x2,x3) x3.^p + sigma .* (Q(3,1).*x1 + Q(3,2).*x2 + Q(3,3).*x3);

    % Linear and poor conditioning
    f1 = @(x1,x2,x3) (1/3 + del) * x1 + 1/3 * x2 + (1/3 - del) * x3;
    f2 = @(x1,x2,x3) (1/3 + del) * x1 + (1/3 - del) * x2 + 1/3 * x3;
    f3 = @(x1,x2,x3) (1/3) * x1 + (1/3 + del) * x2 + (1/3 - del) * x3;

    % Example that straightforward subdivision couldn't find
    % f1 = @(x,y,z) cos(2*pi*x).*cos(2*pi*y).*cos(2*pi*z);
    % f2 = @(x,y,z) y;
    % f3 = @(x,y,z) x.^2 + y.^2 + z.^2 - 1;

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
        expected = [-sqrt(7)/4  0  3/4] + (2*rand(1,3) - 1)/100;

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
            z_unit = roots_z_unit(min_idx,1);
            
            dz_phys_est = estimate_z_error_from_R_fd(R, z_unit, h);
            
            error_floor = eps * cond_root_transformed;           % same floor as before
            predictedDistVals(k,i_Q,i_root_loc) = max(dz_phys_est, error_floor);

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

function dz_phys_est = estimate_z_error_from_R_fd(R, z_hat, h)
    % Forward error estimator: |Δz_phys| ≈ (h/2) * smin / |w' R'(z_hat) v|

    % ---------- Remove a numerically-negligible tail ----------
    nz_full = size(R,3);
    Ak = arrayfun(@(t) norm(R(:,:,t),2), 1:nz_full).';
    relTol = 1e-12 * max(Ak);                % relative gate
    absTol = 1e2  * eps  * sum(Ak);          % absolute guard
    gate   = max(relTol, absTol);
    k_eff  = find(Ak > gate, 1, 'last');
    if isempty(k_eff), k_eff = 1; end
    R = R(:,:,1:k_eff);
    nz = k_eff;  n = size(R,1);

    % ---------- Chebyshev evaluation of R(z_hat) ----------
    [T,~] = cheb_TU_at(z_hat, nz);
    Rz = zeros(n);
    for k = 0:nz-1
        Rz = Rz + R(:,:,k+1) * T(k+1);
    end

    % left/right vectors and residual size
    [U,S,V] = svd(Rz);
    w = U(:,end);           % unit 2-norm
    v = V(:,end);           % unit 2-norm
    smin = S(end,end);      % = min ||R(z_hat) u||_2 over ||u||=1

    % ---------- Numerical derivative of R at z_hat ----------
    % TODO: can probably replace with determinant calculation
    dz = max(1e-7, sqrt(eps)) * max(1, abs(z_hat));

    [Tp,~] = cheb_TU_at(z_hat + dz, nz);
    [Tm,~] = cheb_TU_at(z_hat - dz, nz);

    Rp = zeros(n);  Rm = zeros(n);
    for k = 0:nz-1
        Rp = Rp + R(:,:,k+1) * Tp(k+1);
        Rm = Rm + R(:,:,k+1) * Tm(k+1);
    end
    Rprime_num = (Rp - Rm) / (2*dz);

    den = abs( w' * (Rprime_num * v) );
    den = max(den, eps);    % safety

    % ---------- First-order forward error (map unit->physical) ----------
    % calibration factor to make it more likely that we overestimate rather
    % than underestimate
    calibration_factor = 1e1;

    dz_unit_est  = smin / den;
    dz_phys_est  = (h/2) * dz_unit_est * calibration_factor;
end

function [T,U] = cheb_TU_at(z, nz)
% T(k+1) = T_k(z),  U(k+1) = U_k(z), k=0..nz-1
    T = zeros(nz,1);  U = zeros(nz,1);
    T(1) = 1;                 U(1) = 1;          % T0, U0
    if nz >= 2
        T(2) = z;             U(2) = 2*z;        % T1, U1
    end
    for k = 3:nz
        T(k) = 2*z*T(k-1) - T(k-2);
        U(k) = 2*z*U(k-1) - U(k-2);
    end
end
