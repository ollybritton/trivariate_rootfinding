function [R, n_s1, n_s2, n_t1, n_t2, n_z] = cayley_resultant(f1,f2,f3,n)

% The Cayley function
f_cayley = @(s1,s2,t1,t2,z) (f1(s1,s2,z).*f2(t1,s2,z).*f3(t1,t2,z) + f2(s1,s2,z).*f3(t1,s2,z).*f1(t1,t2,z) + f3(s1,s2,z).*f1(t1,s2,z).*f2(t1,t2,z) ...
    - f3(s1,s2,z).*f2(t1,s2,z).*f1(t1,t2,z) - f2(s1,s2,z).*f1(t1,s2,z).*f3(t1,t2,z) - f1(s1,s2,z).*f3(t1,s2,z).*f2(t1,t2,z)) ./ ((s1-t1).*(s2-t2));

% Amounts of interpolation points based on degrees of s1,s2,t1,t2,z
n_s1 = n;
n_s2 = 2*n;
n_t1 = 2*n;
n_t2 = n;
n_z = 3*n+1;

% Interpolation points
s1_vals = cos((2*(1:n_s1)-1)/(2*n_s1)*pi)';
s2_vals = cos((2*(1:n_s2)-1)/(2*n_s2)*pi)';
t1_vals = s2_vals;
t2_vals = s1_vals;
z_vals = cos((2*(1:n_z)-1)/(2*n_z)*pi)';

% 5D Chebyshev points
[a,b,c,d,e] = ndgrid(s1_vals,s2_vals,t1_vals,t2_vals,z_vals);

% Evaluate the function at the given points (a,b,c,d,e)
f = f_cayley(a(:), b(:), c(:), d(:), e(:));
% Express the values in a grid instead of a list
f = reshape(f,n_s1,n_s2,n_t1,n_t2,n_z);

A = cheby_5D_interpolate(f);

% Matricization of the tensor
R = reshape(A, n_s1*n_s2, n_s1*n_s2, n_z);

end