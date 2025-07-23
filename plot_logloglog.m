close all; clear; clc;

plot_dist = 1;
plot_coeff = 0;

% Sweep over sigma values
sigmaVals = logspace(0, -10, 10); % Range of sigma values
hVals = logspace(0, -7, 25);      % Widths of subregions considered

num_Q = 50;
num_root_loc = 1;
[~, num_h] = size(hVals);
num_sigma = numel(sigmaVals);

% Data storage
distVals = nan(num_h, num_sigma);
predictedDistVals = nan(num_h, num_sigma);

for i_sigma = 1:num_sigma
    sigma = sigmaVals(i_sigma);
    fprintf("Processing sigma = %.2e\n", sigma);

    tmpDistVals = nan(num_h, num_Q, num_root_loc);
    tmpPredDistVals = nan(num_h, num_Q, num_root_loc);

    for i_Q = 1:num_Q
        % Random orthogonal matrix
        Q = rand_orth_mat(3);

        peturb = randn(1,3);

        % Define functions
        f1 = @(x1,x2,x3) x1.^2 + sigma .* (Q(1,1).*x1 + Q(1,2).*x2 + Q(1,3).*x3);
        f2 = @(x1,x2,x3) x2.^2 + sigma .* (Q(2,1).*x1 + Q(2,2).*x2 + Q(2,3).*x3);
        f3 = @(x1,x2,x3) x3.^2 + sigma .* (Q(3,1).*x1 + Q(3,2).*x2 + Q(3,3).*x3);

        for i_root_loc = 1:num_root_loc
            % Random root location
            expected = 2*rand(1,3) - 1;

            % Chebfun3 objects
            p1 = chebfun3(@(x1,x2,x3) f1(x1-expected(1), x2-expected(2), x3-expected(3)));
            p2 = chebfun3(@(x1,x2,x3) f2(x1-expected(1), x2-expected(2), x3-expected(3)));
            p3 = chebfun3(@(x1,x2,x3) f3(x1-expected(1), x2-expected(2), x3-expected(3)));

            for k = 1:num_h
                h = hVals(k);

                % Define cube around expected root
                a = [expected(1) - h/3, expected(2) + h/3, expected(3) - h/4];
                b = [expected(1) + 2*h/3, expected(2) - 2*h/3, expected(3) + 3*h/4];

                cube_scale = (b - a)/2;
                cube_shift = (b + a)/2;

                remap = @(x,idx) cube_scale(idx).*x + cube_shift(idx);

                % Corners of cube
                v = [-1 1];
                [X,Y,Z] = ndgrid(v,v,v);
                x = remap(X(:),1);  y = remap(Y(:),2);  z = remap(Z(:),3);

                % Scaling
                c1 = max(abs(p1(x,y,z)));
                c2 = max(abs(p2(x,y,z)));
                c3 = max(abs(p3(x,y,z)));

                p1_u = @(x1,x2,x3) p1(remap(x1,1), remap(x2,2), remap(x3,3)) / c1;
                p2_u = @(x1,x2,x3) p2(remap(x1,1), remap(x2,2), remap(x3,3)) / c2;
                p3_u = @(x1,x2,x3) p3(remap(x1,1), remap(x2,2), remap(x3,3)) / c3;

                % Root finding
                [roots_z_unit, ~] = roots_z(p1_u, p2_u, p3_u, [-1 -1 -1], [1 1 1], 3);

                if isempty(roots_z_unit)
                    tmpDistVals(k, i_Q, i_root_loc) = NaN;
                else
                    roots_z_remapped = remap(roots_z_unit(:,1),3);
                    d = abs(roots_z_remapped - expected(3));
                    tmpDistVals(k, i_Q, i_root_loc) = min(d);
                end

                tmpPredDistVals(k, i_Q, i_root_loc) = h;
            end
        end
    end

    % Store average over Q and root locations
    distVals(:, i_sigma) = mean(tmpDistVals, [2, 3], 'omitnan');
    predictedDistVals(:, i_sigma) = mean(tmpPredDistVals, [2, 3], 'omitnan');
end

%% Plot 3D log-log-log surface
if plot_dist
    [H, S] = meshgrid(hVals, sigmaVals);
    Z = distVals';  % shape (sigma x h)

    figure;
    surf(log10(H), log10(S), log10(Z));
    xlabel('\(\log_{10} h\)','Interpreter','latex');
    ylabel('\(\log_{10} \sigma\)','Interpreter','latex');
    zlabel('\(\log_{10} \mathrm{error}\)','Interpreter','latex');
    title('3D log-log-log plot of error vs \(h\) and \(\sigma\)', 'Interpreter','latex');
    colorbar;
    grid on;
end
