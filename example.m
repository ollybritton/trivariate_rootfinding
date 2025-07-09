n = 6;

% f1 = @(x,y,z) x+y;
% f2 = @(x,y,z) y+z+0.1;
% f3 = @(x,y,z) z;
% % 
% f1 = @(x,y,z) x;
% f2 = @(x,y,z) y;
% f3 = @(x,y,z) (z-0.1).*(z+0.1);
% 
% n = 2;
% q1 = @(x,y,z) ((x-0.5).^2 + (y-0.5).^2 + z.^2 - 0.5);
% q2 = @(x,y,z) ((x+0.5).^2 + (y-0.5).^2 + z.^2 - 0.5);
% q3 = @(x,y,z) (x.^2 + y.^2 + z.^2 - 0.5^2);
% 
% x0 = 0.1; %The other root is more inaccurate, and changing to +0.1 makes it even more so. Why?
% y0 = 0.0;
% z0 = 0.0;
% f1 = @(x,y,z) (x-1/sqrt(2)).^2 + (y-1/sqrt(2)).^2 + z.^2 - 1/2;
% f2 = @(x,y,z) (x+1/sqrt(2)).^2 + (y-1/sqrt(2)).^2 + z.^2 - 1/2;
% f3 = @(x,y,z) x.^2 + y.^2 + z.^2 - 1/2;

n  = 6;                                   % Chebyshev degree used there
f1 = @(x,y,z) cos(2*pi*x).*cos(2*pi*y).*cos(2*pi*z);
f2 = @(x,y,z) y;
f3 = @(x,y,z) x.^2 + y.^2 + z.^2 - 1;

% --- expected roots -------------------------------------------------
c15 = sqrt(15)/4;           % √15⁄4  ≈  0.9682
c07 = sqrt(7)/4;            % √7 ⁄4  ≈  0.6614
q1  = 1/4;                  %  0.25
q3  = 3/4;                  %  0.75

expected = [ ...
    %  (±√15/4, 0, ±1/4)
     c15  0  q1;
    -c15  0  q1;
     c15  0 -q1;
    -c15  0 -q1;
    %  (±√7/4 , 0, ±3/4)
     c07  0  q3;
    -c07  0  q3;
     c07  0 -q3;
    -c07  0 -q3;
    %  (±1/4 , 0, ±√15/4)
     q1   0  c15;
    -q1   0  c15;
     q1   0 -c15;
    -q1   0 -c15;
    %  (±3/4 , 0, ±√7/4)
     q3   0  c07;
    -q3   0  c07;
     q3   0 -c07;
    -q3   0 -c07 ];
% --------------------------------------------------------------------

actual = roots_xyz(f1,f2,f3,n);

r = roots_xyz(f1,f2,f3,n);

hold on

isosurface(chebfun3(f1), 0, 'g')
isosurface(chebfun3(f2), 0, 'b')
isosurface(chebfun3(f3), 0, 'r')

plot3(r(:, 1), r(:, 2), r(:, 3), 'k.', "MarkerSize", 50)

size(r)