hold on

dom = [-1 -0.95 -0.3 0.3 -0.3 0.3];
f1 = chebfun3(@(x,y,z) cos(2*pi*x).*cos(2*pi*y).*cos(2*pi*z), dom);
f2 = chebfun3(@(x,y,z) y, dom);
f3 = chebfun3(@(x,y,z) x.^2 + y.^2 + z.^2 - 1, dom);

c15 = sqrt(15)/4;           % √15⁄4  ≈  0.9682
c07 = sqrt(7)/4;            % √7 ⁄4  ≈  0.6614
q1  = 1/4;                  %  0.25
q3  = 3/4;                  %  0.75

expected = [ ...
    -c15  0  q1;
    -c15  0 -q1 ];
% --------------------------------------------------------------------

isosurface(chebfun3(f1), 0, 'g')
isosurface(chebfun3(f2), 0, 'b')
isosurface(chebfun3(f3), 0, 'r')

% 
r = roots_xyz(f1,f2,f3);
% 
disp(r);
% 

% 
plot3(r(:, 1), r(:, 2), r(:, 3), 'k.', "MarkerSize", 50)
% plot3(expected(:, 1), expected(:, 2), expected(:, 3), 'k.', "MarkerSize", 50)
% 


hold off
