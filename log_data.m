% ---------------------------------------------------------------
%  MULTI‑SIGMA ROOT‑FINDING EXPERIMENT
%  Outputs a single TXT file with all rows for every σ, Q, root_loc, h
% ---------------------------------------------------------------
close all; clear; clc;3

plot_dist   = 0;     % turn plotting off for batch runs
plot_coeff  = 0;
degree_approx = 3;

% Sweep σ = 1e0, 1e‑1, … , 1e‑10
sigmaVals = 10.^(-(0:8));     % 1e0 to 1e‑10 inclusive
num_sigma = numel(sigmaVals);

% Widths of sub‑regions
hVals = logspace( 0, -9, 20 );

num_Q         = 3;
num_root_loc  = 1;

% ---------------- CSV / TXT LOGGING ----------------
results  = struct([]);          % dynamic; fine for ≤ 10⁵ rows
row      = 0;
timestamp = datestr(now,'yyyymmdd_HHMMSS');
txt_file  = sprintf('csv/results_all_sigmas_deg%d_%s.txt', degree_approx, timestamp);
% ---------------------------------------------------

f = waitbar(0,'Running all σ …');

for i_sigma = 1:num_sigma
    sigma = sigmaVals(i_sigma);

    for i_Q = 1:num_Q
        waitbar(((i_sigma-1)*num_Q + i_Q)/(num_sigma*num_Q), ...
                f, sprintf('σ = %.0e  |  Q #%d', sigma, i_Q));

        % Random orthogonal matrix
        Q = rand_orth_mat(3);
        peturb = rand(1,3);

        % Devastating example (Noferini–Townsend)
        f1 = @(x1,x2,x3) x1.^2 + sigma .* (Q(1,1).*x1 + Q(1,2).*x2 + Q(1,3).*x3) + dot([x1.^4 x2.^4 x3.^4], peturb);
        f2 = @(x1,x2,x3) x2.^2 + sigma .* (Q(2,1).*x1 + Q(2,2).*x2 + Q(2,3).*x3) + dot([x1.^4 x2.^4 x3.^4], peturb);
        f3 = @(x1,x2,x3) x3.^2 + sigma .* (Q(3,1).*x1 + Q(3,2).*x2 + Q(3,3).*x3) + dot([x1.^4 x2.^4 x3.^4], peturb);

        for i_root_loc = 1:num_root_loc
            % Root location
            expected = 2*rand(1,3) - 1;

            % Chebfun3 objects
            p1 = chebfun3(@(x1,x2,x3) f1(x1-expected(1), x2-expected(2), x3-expected(3)));
            p2 = chebfun3(@(x1,x2,x3) f2(x1-expected(1), x2-expected(2), x3-expected(3)));
            p3 = chebfun3(@(x1,x2,x3) f3(x1-expected(1), x2-expected(2), x3-expected(3)));

            % Jacobian at the exact root
            J_func = jac(p1,p2,p3);
            J      = J_func(expected(1), expected(2), expected(3));
            J_inv  = inv(J);
            cond_root = norm(J_inv,2);
            detJ      = det(J);

            preconditioner = J_inv;
            p1_q = preconditioner(1,1)*p1 + preconditioner(1,2)*p2 + preconditioner(1,3)*p3;
            p2_q = preconditioner(2,1)*p1 + preconditioner(2,2)*p2 + preconditioner(2,3)*p3;
            p3_q = preconditioner(3,1)*p1 + preconditioner(3,2)*p2 + preconditioner(3,3)*p3;

            cond_eig         = 1/abs(detJ);
            err_estimate     = 1e-15 * cond_root;
            err_estimate_eig = 1e-15 * cond_eig;

            % ---------- sweep over h ----------
            for k = 1:numel(hVals)
                h = hVals(k);

                % Build cube [a,b]
                a = [expected(1)-h/3, expected(2)+h/3, expected(3)-h/4];
                b = [expected(1)+2*h/3, expected(2)-2*h/3, expected(3)+3*h/4];
                cube_scale = (b-a)/2;
                cube_shift = (b+a)/2;
                remap = @(x,idx) cube_scale(idx).*x + cube_shift(idx);

                % Scaling constants
                v = [-1 1];
                [X,Y,Z] = ndgrid(v,v,v);
                x = remap(X(:),1);  y = remap(Y(:),2);  z = remap(Z(:),3);
                c1 = max(abs(p1(x,y,z)));
                c2 = max(abs(p2(x,y,z)));
                c3 = max(abs(p3(x,y,z)));

                % Unit‑cube versions
                p1_u = @(x1,x2,x3) p1(remap(x1,1), remap(x2,2), remap(x3,3))/c1;
                p2_u = @(x1,x2,x3) p2(remap(x1,1), remap(x2,2), remap(x3,3))/c2;
                p3_u = @(x1,x2,x3) p3(remap(x1,1), remap(x2,2), remap(x3,3))/c3;

                p1_u_c = chebfun3(p1_u,[degree_approx degree_approx degree_approx]);
                p2_u_c = chebfun3(p2_u,[degree_approx degree_approx degree_approx]);
                p3_u_c = chebfun3(p3_u,[degree_approx degree_approx degree_approx]);

                J_hat_func = jac(p1_u_c,p2_u_c,p3_u_c);
                J_hat = J_hat_func(remap(expected(1),1), remap(expected(2),2), remap(expected(3),3));
                detJ_hat = det(J_hat);
                cond_hat = 1/abs(detJ_hat);
                err_estimate_hat = 1e-15 * cond_hat * h;

                scale_factor = (abs(detJ_hat)/norm(J,2)).^(1/3);
                p1_u = @(x1,x2,x3) scale_factor * p1(remap(x1,1), remap(x2,2), remap(x3,3))/c1;
                p2_u = @(x1,x2,x3) scale_factor * p2(remap(x1,1), remap(x2,2), remap(x3,3))/c2;
                p3_u = @(x1,x2,x3) scale_factor * p3(remap(x1,1), remap(x2,2), remap(x3,3))/c3;

                % Roots
                [roots_z_unit, R, ~, ~, approx_err] = roots_z(p1_u,p2_u,p3_u,[-1 -1 -1],[1 1 1],degree_approx);
                [roots_z_prec, ~, ~, ~]            = roots_z(p1_q,p2_q,p3_q,[-1 -1 -1],[1 1 1],degree_approx);

                % max ||A_k||_2
                Ai_norms = arrayfun(@(kk) norm(R(:,:,kk),2), 1:size(R,3));
                max_A2   = max(Ai_norms);

                % Distances
                dist_z           = NaN;
                predicted_dist_z = NaN;
                test_dist_z      = NaN;

                if ~isempty(roots_z_unit)
                    roots_z_rem = remap(roots_z_unit(:,1),3);
                    dist_z = min(abs(roots_z_rem - expected(3)));
                    predicted_dist_z = approx_err * h;
                end
                if ~isempty(roots_z_prec)
                    test_dist_z = min(abs(roots_z_prec - expected(3)));
                end

                % ----------- write a row ------------
                row = row + 1;
                results(row).sigma         = sigma;
                results(row).i_sigma       = i_sigma;
                results(row).i_Q           = i_Q;
                results(row).i_root_loc    = i_root_loc;
                results(row).degree_approx = degree_approx;

                results(row).h             = h;
                results(row).detJ          = detJ;
                results(row).normJinv      = cond_root;
                results(row).cond_root     = cond_root;
                results(row).cond_eig      = cond_eig;
                results(row).err_est_root  = err_estimate;
                results(row).err_est_eig   = err_estimate_eig;

                results(row).detJ_hat      = detJ_hat;
                results(row).cond_hat      = cond_hat;
                results(row).err_est_hat   = err_estimate_hat;

                results(row).dist_z        = dist_z;
                results(row).pred_dist_z   = predicted_dist_z;
                results(row).test_dist_z   = test_dist_z;

                results(row).max_A2        = max_A2;
                results(row).approx_err    = approx_err;
                results(row).scale_factor  = scale_factor;

                results(row).expected_x    = expected(1);
                results(row).expected_y    = expected(2);
                results(row).expected_z    = expected(3);
            end % h
        end % root_loc
    end % Q
end % σ

delete(f);

% --------------- write everything once ---------------
T = struct2table(results);
writetable(T, txt_file, 'Delimiter', '\t');  % tab‑separated = plain text
fprintf('Wrote %d rows (%d σ values) to %s\n', height(T), num_sigma, txt_file);
