underlying_function = @(x,y,z) sin(x + y * z);

eps = 1e-15;

width = 0.1;
dom = [-width width -width width -width width];

f1 = chebfun3(underlying_function, dom, "eps", eps);

disp(length(f1))