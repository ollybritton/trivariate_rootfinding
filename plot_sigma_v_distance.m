%% Root‐distance versus σ
% ---------------------------------------------------------------
% Hand‑picked orthogonal matrix
Q = [ ...
    -0.22081075  -0.29306608  -0.93024453; ...
    -0.67705210  -0.64047179   0.36248634; ...
    -0.70202783   0.70986489  -0.05699795 ];

% "True" root location (chosen translation of the origin)
expected = [0.35298, 0.23487, -0.36024];

% σ values to test (adjust the range / density if you like)
sigmaVals = logspace(-3, 0, 20);   % 10⁻⁴ → 10⁰ on a log scale
distVals  = zeros(size(sigmaVals));

% Subregion for root finding (keep wide enough for all σ)
lo = [-1 -1 -1];
hi = [ 1  1  1];

for k = 1:numel(sigmaVals)
    sigma = sigmaVals(k);

    % Define the three components of F(x) for this σ
    f1 = @(x1,x2,x3) x1.^2 + sigma .* (Q(1,1).*x1 + Q(1,2).*x2 + Q(1,3).*x3);
    f2 = @(x1,x2,x3) x2.^2 + sigma .* (Q(2,1).*x1 + Q(2,2).*x2 + Q(2,3).*x3);
    f3 = @(x1,x2,x3) x3.^2 + sigma .* (Q(3,1).*x1 + Q(3,2).*x2 + Q(3,3).*x3);

    % Shift so the "true" root lies at the translation point
    p1 = chebfun3(@(x1,x2,x3) f1(x1-expected(1), x2-expected(2), x3-expected(3)));
    p2 = chebfun3(@(x1,x2,x3) f2(x1-expected(1), x2-expected(2), x3-expected(3)));
    p3 = chebfun3(@(x1,x2,x3) f3(x1-expected(1), x2-expected(2), x3-expected(3)));

    % Locate all roots inside the bounding box
    rootsAll = roots_xyz_subregion(p1, p2, p3, lo, hi, 3, 0.1);

    % Euclidean distance from each root to the expected location
    d = vecnorm(rootsAll - expected, 2, 2);
    distVals(k) = min(d);    % store the closest one
end

%% Plot: distance vs σ
figure;
loglog(sigmaVals, distVals, 'o-','LineWidth',1.2,'MarkerSize',6);
grid on;
xlabel('\sigma','Interpreter','tex');
ylabel('Euclidean distance to expected root');
title('Convergence of the closest root as \sigma varies');
