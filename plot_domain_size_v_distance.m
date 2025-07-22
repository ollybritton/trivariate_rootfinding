close all; clear; clc;

plot_coeff = 0;

num_Q = 3;
num_root_loc = 3;

% Random orthogonal matrix
Q = rand_orth_mat(3);

% Controls difficulty of problem, by default the condition number of the
% root finding problem is 1/sigma.
sigma = 1e-1;

peturb = randn(1,3) * 1e-15 * 0;

% Devastating example as in Noferini-Townsend
f1 = @(x1,x2,x3) x1.^2 + sigma .* (Q(1,1).*x1 + Q(1,2).*x2 + Q(1,3).*x3) + x1*peturb(1);
f2 = @(x1,x2,x3) x2.^2 + sigma .* (Q(2,1).*x1 + Q(2,2).*x2 + Q(2,3).*x3) + x2*peturb(2);
f3 = @(x1,x2,x3) x3.^2 + sigma .* (Q(3,1).*x1 + Q(3,2).*x2 + Q(3,3).*x3) + x3*peturb(3);

% Chosen location of the root (by default it is the origin, but that might
% be especially easy to find for some reason).
expected = [0.9834675, -0.2374, -0.36024];

% Chebfun3 objects corresponding to translated problem
p1 = chebfun3(@(x1,x2,x3) f1(x1-expected(1), x2-expected(2), x3-expected(3)));
p2 = chebfun3(@(x1,x2,x3) f2(x1-expected(1), x2-expected(2), x3-expected(3)));
p3 = chebfun3(@(x1,x2,x3) f3(x1-expected(1), x2-expected(2), x3-expected(3)));

% Conditioning analysis
J_func = jac(p1,p2,p3);
J = J_func(expected(1), expected(2), expected(3));
J_inv = inv(J);
cond = norm(J_inv);
err_estimate = 1e-15 * cond;

hVals = logspace( 0, -5, 25 ); % Widths of subregions considered

% Distances of z-components to expected root
distVals = nan(size(hVals));
predictedDistVals = nan(size(hVals));

% Maximum magnitude of coordinate matrices
coordinateMatrixMagnitude = nan(size(hVals));
predictedCoordinateMatrixMagnitude = nan(size(hVals));

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
    
    % No need to scale these functions if scaling is done in roots_z
    c1 = max(abs(p1(x,y,z)));
    c2 = max(abs(p2(x,y,z)));
    c3 = max(abs(p3(x,y,z)));
    
    p1_u = @(x1,x2,x3) p1(remap(x1,1), remap(x2,2), remap(x3,3)) / c1;
    p2_u = @(x1,x2,x3) p2(remap(x1,1), remap(x2,2), remap(x3,3)) / c2;
    p3_u = @(x1,x2,x3) p3(remap(x1,1), remap(x2,2), remap(x3,3)) / c3;

    % Locate all roots inside this cube, in the "unit" coordinates.
    [roots_z_unit, R] = roots_z(p1_u, p2_u, p3_u, [-1 -1 -1], [1 1 1], 3);

    % Calculate maximum magintude over coordinate matrices 
    n_z = size(R,3);
    Ai_norms = arrayfun(@(k) norm(R(:,:,k),2), 1:n_z);
    max_A2 = max(Ai_norms);
    coordinateMatrixMagnitude(k) = max_A2;
    predictedCoordinateMatrixMagnitude(k) = 1;%h^2;

    if isempty(roots_z_unit)
        distVals(k) = NaN;
        warning('No roots found for h = %.3g – recording NaN', h);
    else
        roots_z_remapped = remap(roots_z_unit(:,1),3);
        d = abs(roots_z_remapped - expected(3));
        distVals(k) = min(d);

        % Rough theoretical predictions
        predictedDistVals(k) = h;
    end
end

%% Plot for width vs distance to root
    figure;
    
    loglog(hVals, distVals, 'o-','LineWidth',1.2,'MarkerSize',6);
    hold on;
    loglog(hVals, predictedDistVals, 'o-','LineWidth',1.2,'MarkerSize',6);
    grid on;
    xlabel('box width \(h\)','Interpreter','latex');
    ylabel('error in \(z\)-component','Interpreter','latex');
    
    title(sprintf(['Effect of shrinking the domain on error' '(σ = %.0e)'], sigma), 'Interpreter','latex');
    
    % line showing how small the domain should be for good error
    % not sure if it should be cond or 1/cond??
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
    
    yline(cond^3 * 1e-15,'r--', ...                       
          'Interpreter','latex', ...
          'Label','\(\mathrm{err} \approx u\cdot\mathrm{cond}^3\)', ...
          'LabelOrientation','horizontal', ...
          'LabelVerticalAlignment','bottom', ...
          'LabelHorizontalAlignment','center');
    
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