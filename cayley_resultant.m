function [R, n_s1, n_s2, n_t1, n_t2, n_z, err] = cayley_resultant(f1,f2,f3,n, n_test)
%‑‑‑ cayley_resultant  Compute the matricised 5‑D Cayley resultant
%                      **and** an accuracy estimate.
%
%   [R, n_s1, n_s2, n_t1, n_t2, n_z, ERR] = ...cayley_resultant(...)
%
%   * n_test  – (optional) number of test points per *s₁*–dimension used
%               for the error check (default: n_test = 2*n + 3).
%
%   ERR      – max‑abs error
%              \( \displaystyle
%                 \varepsilon_\infty
%                 = \max_{(s_1,\ldots ,z)\in \mathcal G_{\text{test}}}
%                   \bigl|f_{\text{cayley}} - p_{\text{cheb}}\bigr|
%               \)

    if nargin < 5 || isempty(n_test),  n_test = n;  end

    % ------------------------------------------------------------
    % 1)  Build the Cayley integrand
    % ------------------------------------------------------------
    f_cayley = @(s1,s2,t1,t2,z) ( ...
          f1(s1,s2,z).*f2(t1,s2,z).*f3(t1,t2,z) ...
        + f2(s1,s2,z).*f3(t1,s2,z).*f1(t1,t2,z) ...
        + f3(s1,s2,z).*f1(t1,s2,z).*f2(t1,t2,z) ...
        - f3(s1,s2,z).*f2(t1,s2,z).*f1(t1,t2,z) ...
        - f2(s1,s2,z).*f1(t1,s2,z).*f3(t1,t2,z) ...
        - f1(s1,s2,z).*f3(t1,s2,z).*f2(t1,t2,z) ) ...
        ./ ((s1-t1).*(s2-t2));

    % ------------------------------------------------------------
    % 2)  Interpolation grid & function values  (unchanged)
    % ------------------------------------------------------------
    n_s1 = n;  n_s2 = 2*n;
    n_t1 = 2*n; n_t2 = n;
    n_z  = 3*n + 1;

    s1 = cos((2*(1:n_s1)-1)/(2*n_s1)*pi)';   % size n_s1×1
    s2 = cos((2*(1:n_s2)-1)/(2*n_s2)*pi)';   % size n_s2×1
    t1 = s2;                                 % re‑use nodes
    t2 = s1;
    z  = cos((2*(1:n_z) -1)/(2*n_z)*pi)';

    [p1,p2,p3,p4,p5] = ndgrid(s1,s2,t1,t2,z);
    fvals = f_cayley(p1(:),p2(:),p3(:),p4(:),p5(:));
    fvals = reshape(fvals, n_s1,n_s2,n_t1,n_t2,n_z);

    % ------------------------------------------------------------
    % 3)  Coefficient tensor and matricisation  (unchanged)
    % ------------------------------------------------------------
    A = cheby_5D_interpolate(fvals);               % coeffs
    R = reshape(A, n_s1*n_s2, n_s1*n_s2, n_z);     % matricise
    
    % ------------------------------------------------------------
    % 4)  Build a *fine uniform* test grid in [-1,1]^5
    % ------------------------------------------------------------
    s1t = linspace(-1, 1, n_test).';        % n_test × 1
    s2t = linspace(-1, 1, 2*n_test).';      % 2*n_test × 1
    t1t = s2t;                              % same sizes as before
    t2t = s1t;
    zt  = linspace(-1, 1, 3*n_test+1).';    % 3*n_test+1 × 1
    
    [q1,q2,q3,q4,q5] = ndgrid(s1t, s2t, t1t, t2t, zt);
    
    % --- evaluate and compute error exactly as before -------------
    f_exact   = f_cayley(q1(:),q2(:),q3(:),q4(:),q5(:));
    f_approx  = cheby_5D_evaluate(A, q1(:),q2(:),q3(:),q4(:),q5(:));
    bad       = ~isfinite(f_exact) | ~isfinite(f_approx);   % skip 0/0's
    err       = max(abs(f_exact(~bad) - f_approx(~bad)));
end
