% scan_perturbations_large_h1_error.m
close all; clear; clc;

plot_dist      = 1;     % keep your plots
plot_coeff     = 0;
degree_approx  = 2;

% Difficulty parameter from your template (not used by the linear case below)
sigma = 0;

% Polynomial degree used in commented example (kept for completeness)
p = 1;

% Subregion widths (same as your code)
hVals = logspace(1, -3, 40);
[~, num_h] = size(hVals);

% --- choose the discrete index closest to h = 1
[~, idx_h1] = min(abs(hVals - 1));
h_near_1    = hVals(idx_h1);

% Experiment controls
num_Q          = 1;     % keep as in your script
num_root_loc   = 20;    % "several tries" over random translated roots
num_trials     = 40;    % number of slightly-perturbed coefficient sets to test
thresholdLarge = 1e-1;  % report if mean observed error at h≈1 exceeds this
rng(1);                 % reproducibility

% Very slight absolute perturbation magnitude to the 3x3 coefficient matrix
noise_abs = 1e-10;      % tweak if you want slightly stronger/weaker perturbations

% Storage for metrics per run
data_size  = [num_h, num_Q, num_root_loc];
best.avg   = -Inf; best.A = []; best.trial = NaN;

fprintf('h closest to 1 is h = %.6g (index %d of %d)\n', h_near_1, idx_h1, num_h);
hit_count = 0;

for trial = 1:num_trials
    % --- Base nearly rank-1 linear example (your "poor conditioning" case)
    et0  = 1e-7;  ep0  = 1e-7;  del0 = 1e-7;
    A0 = [ 1/3 + et0,  1/3,        1/3 - et0;
           1/3 + ep0,  1/3 - ep0,  1/3;
           1/3,        1/3 - del0, 1/3 + del0 ];

    % --- Very slight random perturbation (row-mean centered to keep row sums ≈ unchanged)
    dA = noise_abs * randn(3,3);
    dA = dA - mean(dA,2) .* [1 1 1];     % preserve each row sum approximately
    A  = A0 + dA;

    % Allocate per-trial arrays
    distVals                  = nan(data_size);   % |z - z_*|
    predictedDistVals         = nan(data_size);
    coordinateMatrixMagnitude = nan(data_size);

    for i_Q = 1:num_Q
        % Random orthogonal matrix (kept for parity with your template)
        Q       = rand_orth_mat(3); %#ok<NASGU>  % not used in the linear case
        perturb = 0*rand(1,3);      %#ok<NASGU>

        % Define the slightly perturbed linear maps
        f1 = @(x1,x2,x3) A(1,1)*x1 + A(1,2)*x2 + A(1,3)*x3;
        f2 = @(x1,x2,x3) A(2,1)*x1 + A(2,2)*x2 + A(2,3)*x3;
        f3 = @(x1,x2,x3) A(3,1)*x1 + A(3,2)*x2 + A(3,3)*x3;

        for i_root_loc = 1:num_root_loc
            % Choose a (small) translated root location
            expected = (2*rand(1,3) - 1)/100;

            % Chebfun3 objects for translated problem
            p1 = chebfun3(@(x1,x2,x3) f1(x1-expected(1), x2-expected(2), x3-expected(3)));
            p2 = chebfun3(@(x1,x2,x3) f2(x1-expected(1), x2-expected(2), x3-expected(3)));
            p3 = chebfun3(@(x1,x2,x3) f3(x1-expected(1), x2-expected(2), x3-expected(3)));

            % Conditioning at the (translated) root
            J_func   = jac(p1,p2,p3);
            J        = J_func(expected(1), expected(2), expected(3));
            condJinv = norm(inv(J));        %#ok<NASGU>
            cond_eig = 1 / abs(det(J));     %#ok<NASGU>

            for k = 1:numel(hVals)
                h = hVals(k);

                % Asymmetric cube around the expected root (your mapping)
                a = [expected(1) - h/3,   expected(2) + h/3,   expected(3) - h/4];
                b = [expected(1) + 2*h/3, expected(2) - 2*h/3, expected(3) + 3*h/4];

                cube_scale = (b - a)/2;
                cube_shift = (b + a)/2;
                remap = @(x,idx) cube_scale(idx).*x + cube_shift(idx);

                % Scale to O(1) on the box
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
                % --- SAFE call to roots_z: ignore "Output argument ... not assigned" cases
                [roots_z_unit, R, V, W, approx_err] = deal([]);
                try
                    [roots_z_unit, R, V, W, approx_err] = roots_z(p1_u, p2_u, p3_u, [-1 -1 -1], [1 1 1], degree_approx);
                catch ME
                    if contains(ME.message,'Output argument') && contains(ME.message,'roots_z')
                        % Ignore and proceed with empty outputs
                    else
                        rethrow(ME);
                    end
                end

                if ~isempty(R)
                    n_z     = size(R,3);
                    Ai_norm = arrayfun(@(t) norm(R(:,:,t),2), 1:n_z);
                    max_A2  = max(Ai_norm);
                    coordinateMatrixMagnitude(k, i_Q, i_root_loc) = max_A2; %#ok<NASGU>
                end

                if isempty(roots_z_unit)
                    distVals(k, i_Q, i_root_loc) = NaN;
                else
                    % observed error (same as before)
                    roots_z_remapped = remap(roots_z_unit(:,1), 3);
                    d = abs(roots_z_remapped - expected(3));
                    [min_dist, min_idx] = min(d);
                    distVals(k, i_Q, i_root_loc) = min_dist;
                
                    % compute max_A2 only if R is available
                    if ~isempty(R)
                        n_z     = size(R,3);
                        Ai_norm = arrayfun(@(t) norm(R(:,:,t),2), 1:n_z);
                        max_A2  = max(Ai_norm);
                        coordinateMatrixMagnitude(k, i_Q, i_root_loc) = max_A2;
                    end
                
                    % predicted error: only if everything needed is available
                    if ~isempty(R) && ~isempty(V) && ~isempty(W) && ~isempty(approx_err) && min_idx <= numel(V) && min_idx <= numel(W)
                        v_sel = V(min_idx);  w_sel = W(min_idx);
                        vfac = norm(v_sel,2);  wfac = norm(w_sel,2);
                        N_xy = size(R, 1);
                
                        err_res = 4 * N_xy * (c1 * c2 * c3) * vfac * wfac / (h^2 * abs(det(J))) * ...
                                  (approx_err + 1e-15 * (exist('max_A2','var') && ~isempty(max_A2)) * max_A2);
                        predictedDistVals(k, i_Q, i_root_loc) = max(err_res);
                    else
                        predictedDistVals(k, i_Q, i_root_loc) = NaN;
                    end
                end
            end
        end
    end

    % --- compute the trial's mean observed error at h≈1 across all roots
    avg_err_h1 = mean(distVals(idx_h1,:,:), [2,3], 'omitnan');

    % Track the worst trial
    if avg_err_h1 > best.avg
        best.avg   = avg_err_h1;
        best.A     = A;
        best.trial = trial;
    end

    % Print when it's "very large" by your threshold
    if avg_err_h1 > thresholdLarge
        hit_count = hit_count + 1;
        fprintf('\n[HIT %d] trial %d: mean observed error at h≈%.6g is %.6g\n', ...
                hit_count, trial, h_near_1, avg_err_h1);
        fprintf('Coefficient matrix A (very slightly perturbed):\n');
        disp(A);
    else
        fprintf('trial %d: mean observed error at h≈%.6g is %.6g (below threshold %.3g)\n', ...
                trial, h_near_1, avg_err_h1, thresholdLarge);
    end
end

if hit_count == 0
    fprintf('\nNo trial exceeded the threshold %.3g at h≈%.6g.\n', thresholdLarge, h_near_1);
    fprintf('Worst case found was trial %d with mean observed error %.6g.\n', best.trial, best.avg);
    fprintf('Coefficient matrix A for the worst case:\n');
    disp(best.A);
end

%% Plot: box width vs distance to root (z-component) for the last trial
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
    title('Effect of shrinking the domain on error (last trial)', 'Interpreter','latex');

    % Mark h closest to 1
    xline(h_near_1,'k--', ...
          'Interpreter','latex', ...
          'Label','\(h \approx 1\)', ...
          'LabelOrientation','horizontal', ...
          'LabelVerticalAlignment','bottom', ...
          'LabelHorizontalAlignment','center');
end
