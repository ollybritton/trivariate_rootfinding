%% Distance to expected root  vs  search‑box width h
% ---------------------------------------------------------------
% Hand‑picked orthogonal matrix
Q = [ ...
    -0.70506063  -0.70914702; ...
     0.70914702  -0.70506063  ];

% "True" root location (chosen translation of the origin)
% expected = [0.34298, 0.23487, -0.36024];
expected = [0.68298, 0.23487];

%% Fix σ here
sigma = 1e-3;                       % <-- change if you want

% Components of F(x) (independent of h)
f1 = @(x1,x2) x1.^2 + sigma .* (Q(1,1).*x1 + Q(1,2).*x2);
f2 = @(x1,x2) x2.^2 + sigma .* (Q(2,1).*x1 + Q(2,2).*x2);

% Chebfun3 objects (only need to build once)
p1 = chebfun2(@(x1,x2) f1(x1-expected(1), x2-expected(2)));
p2 = chebfun2(@(x1,x2) f2(x1-expected(1), x2-expected(2)));

%% Range of box widths h (full side length)
hVals = logspace( 1, -9, 10 );      %
distVals = nan(size(hVals));        % pre‑allocate

for k = 1:numel(hVals)
    tic
    h = hVals(k);

    % Cube centred on 'expected' with side length h
    lo = [expected(1) - 0.6*h, expected(2) - 0.4*h];
    hi = [expected(1) + 0.4*h, expected(2) + 0.6*h];

    disp(lo);
    disp(hi);

    scale = (hi - lo)/2;
    shift = (hi + lo)/2;
    
    remap = @(x,idx) scale(idx).*x + shift(idx);
    new_expected = [remap(expected(1),1), remap(expected(2),2)];
        
    p1_unit = @(x1,x2) p1(remap(x1,1), remap(x2,2));
    p2_unit = @(x1,x2) p2(remap(x1,1), remap(x2,2));

    % Locate all roots inside that square
    roots_y_unit = yuji(p1_unit, p2_unit); % no subdiv

    roots_y = remap(roots_y_unit(:,1),2);

    if isempty(roots_y)
        % No root detected in this tiny box
        distVals(k) = NaN;
        warning('No roots found for h = %.3g – recording NaN', h);
    else
        % Euclidean distance of each root to the expected point
        d = abs(roots_y - expected(2));
        distVals(k) = min(d);
    end
    toc
end

%% Plot: distance vs h
figure;
loglog(hVals, distVals, 'o-','LineWidth',1.2,'MarkerSize',6);
grid on;
xlabel('box width  \(h\)','Interpreter','latex');
ylabel('Euclidean distance to expected root','Interpreter','latex');
title(sprintf(['Effect of shrinking the domain ' ...
               '(\\sigma = %.0e)'], sigma), ...
      'Interpreter','latex');

xline(sigma,'r--', ...                       
      'Interpreter','latex', ...
      'Label','\(h \approx \mathrm{cond}\)', ...
      'LabelOrientation','horizontal', ...
      'LabelVerticalAlignment','bottom', ...
      'LabelHorizontalAlignment','center');

yline(1/sigma * 1e-15,'r--', ...                       
      'Interpreter','latex', ...
      'Label','\(\mathrm{err} \approx u\cdot\mathrm{cond}\)', ...
      'LabelOrientation','horizontal', ...
      'LabelVerticalAlignment','bottom', ...
      'LabelHorizontalAlignment','center');

% Optional: overlay the line distance = h (visual guide)
hold on;