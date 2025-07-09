n = 6;

% f1 = @(x,y,z) x+y;
% f2 = @(x,y,z) y+z+0.1;
% f3 = @(x,y,z) z;
 
f1 = chebfun3(@(x,y,z) x);
f2 = chebfun3(@(x,y,z) (y-0.1).*(y+0.1));
f3 = chebfun3(@(x,y,z) (z-0.2).*(z+0.2).*(z-0.3));

actual = roots_xyz(f1,f2,f3,n);

r = roots_xyz(f1,f2,f3,n);

disp(r);

isosurface(chebfun3(f1), 0, 'g')
hold on
isosurface(chebfun3(f2), 0, 'b')
isosurface(chebfun3(f3), 0, 'r')

plot3(r(:, 1), r(:, 2), r(:, 3), 'k.', "MarkerSize", 50)

hold off