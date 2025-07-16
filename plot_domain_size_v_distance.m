%% Distance to expected root  vs  search‑box width h
% ---------------------------------------------------------------
% Hand‑picked orthogonal matrix
Q = [ ...
    -0.22081075  -0.29306608  -0.93024453; ...
    -0.67705210  -0.64047179   0.36248634; ...
    -0.70202783   0.70986489  -0.05699795 ];

% "True" root location (chosen translation of the origin)
% expected = [0.34298, 0.23487, -0.36024];
expected = [0, 0, 0];

%% Fix σ here
sigma = 1e-3;                       % <-- change if you want

% Components of F(x) (independent of h)
f1 = @(x1,x2,x3) x1.^2 + sigma .* (Q(1,1).*x1 + Q(1,2).*x2 + Q(1,3).*x3);
f2 = @(x1,x2,x3) x2.^2 + sigma .* (Q(2,1).*x1 + Q(2,2).*x2 + Q(2,3).*x3);
f3 = @(x1,x2,x3) x3.^2 + sigma .* (Q(3,1).*x1 + Q(3,2).*x2 + Q(3,3).*x3);

% Chebfun3 objects (only need to build once)
p1 = chebfun3(@(x1,x2,x3) f1(x1-expected(1), x2-expected(2), x3-expected(3)));
p2 = chebfun3(@(x1,x2,x3) f2(x1-expected(1), x2-expected(2), x3-expected(3)));
p3 = chebfun3(@(x1,x2,x3) f3(x1-expected(1), x2-expected(2), x3-expected(3)));

%% Range of box widths h (full side length)
hVals = logspace( 0, -3, 25 );      % 10⁰ → 10⁻³
distVals = nan(size(hVals));        % pre‑allocate

for k = 1:numel(hVals)
    h = hVals(k);

    % Cube centred on 'expected' with side length h
    lo = expected - h/2;
    hi = expected + h/2;

    disp(lo);
    disp(hi);

    % Locate all roots inside that cube
    rootsAll = roots_z(p1, p2, p3, lo, hi, 3); % no subdiv

    if isempty(rootsAll)
        % No root detected in this tiny box
        distVals(k) = NaN;
        warning('No roots found for h = %.3g – recording NaN', h);
    else
        % Euclidean distance of each root to the expected point
        d = abs(rootsAll - expected(3));
        distVals(k) = min(d);        % closest root
    end
end

%% What does the analysis say?
J_func = jac(p1,p2,p3);
J = J_func(expected(1), expected(2), expected(3));
J_inv = inv(J);
cond = norm(J_inv);
err_estimate = 1e-15 * cond;

%% Plot: distance vs h
figure;
loglog(hVals, distVals, 'o-','LineWidth',1.2,'MarkerSize',6);
grid on;
xlabel('box width  \(h\)','Interpreter','latex');
ylabel('Euclidean distance to expected root','Interpreter','latex');
title(sprintf(['Effect of shrinking the domain ' ...
               '(\\sigma = %.0e)'], sigma), ...
      'Interpreter','latex');

% Optional: overlay the line distance = h (visual guide)
hold on;

legend('nearest‑root error','best possible error','Location','northwest');

% line showing how small the domain should be for good error
% not sure if it should be cond or 1/cond??
xline(1/cond,'r--', ...                       
      'Interpreter','latex', ...
      'Label','\(h \approx \mathrm{cond}\)', ...
      'LabelOrientation','horizontal', ...
      'LabelVerticalAlignment','bottom', ...
      'LabelHorizontalAlignment','center');

yline(cond * 1e-15,'r--', ...                       
      'Interpreter','latex', ...
      'Label','\(\mathrm{err} \approx u\cdot\mathrm{cond}\)', ...
      'LabelOrientation','horizontal', ...
      'LabelVerticalAlignment','bottom', ...
      'LabelHorizontalAlignment','center');