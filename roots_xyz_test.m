classdef roots_xyz_test < matlab.unittest.TestCase
    methods(Test)
        function threeSpheres(testCase)
            f1 = chebfun3(@(x,y,z) (x-1/sqrt(2)).^2 + (y-1/sqrt(2)).^2 + z.^2 - 1);
            f2 = chebfun3(@(x,y,z) (x+1/sqrt(2)).^2 + (y-1/sqrt(2)).^2 + z.^2 - 1);
            f3 = chebfun3(@(x,y,z) x.^2 + y.^2 + z.^2 - 1/2);

            actual = roots_xyz(f1,f2,f3);
            expected = [...
                0, sqrt(2)/4, -sqrt(6)/4;
                0, sqrt(2)/4, sqrt(6)/4;
            ];
            
            assertSameRoots(testCase, actual, expected, 1e-10)
        end

        function threeCylinders(testCase)
            dom = [-1 1 -1 1 -1 1];
            f1 = chebfun3(@(x,y,z) x.^2 + y.^2 - 1, dom);
            f2 = chebfun3(@(x,y,z) y.^2 + z.^2 - 1, dom);
            f3 = chebfun3(@(x,y,z) z.^2 + x.^2 - 1, dom);
    
            s = 1/sqrt(2);
            expected = s * [ ...
                 1  1  1;
                -1  1  1;
                 1 -1  1;
                 1  1 -1;
                -1 -1  1;
                -1  1 -1;
                 1 -1 -1;
                -1 -1 -1 ];
    
            actual = roots_xyz(f1,f2,f3);
            assertSameRoots(testCase,actual,expected,1e-10)
        end

        function torusPlanes(testCase)
            R = 1; r = 0.5;
            dom = [-2 2 -2 2 -2 2];
            f1 = chebfun3(@(x,y,z) (x.^2 + y.^2 + z.^2 + R^2 - r^2).^2 - 4*R^2*(x.^2 + y.^2), dom);
            f2 = chebfun3(@(x,y,z) y, dom);
            f3 = chebfun3(@(x,y,z) z, dom);

            expected = [ ...
                -1.5 0 0;
                -0.5 0 0;
                0.5 0 0;
                1.5 0 0 ];

            actual = roots_xyz(f1,f2,f3);
            assertSameRoots(testCase,actual,expected,1e-6)
        end

        function simpleIntersect(testCase)
            f1 = chebfun3(@(x,y,z) x-0.1);
            f2 = chebfun3(@(x,y,z) y-0.1);
            f3 = chebfun3(@(x,y,z) z-0.1);

            actual = roots_xyz(f1,f2,f3);
            expected = [0.1 0.1 0.1;];
            
            assertSameRoots(testCase, actual, expected, 1e-15)
        end

        function severalIntersect(testCase)
            f1 = chebfun3(@(x,y,z) x-0.1);
            f2 = chebfun3(@(x,y,z) y-0.1);
            f3 = chebfun3(@(x,y,z) (z-0.1).*(z+0.1));

            actual = roots_xyz(f1,f2,f3);
            expected = [0.1 0.1 -0.1; 0.1 0.1 0.1];
            
            assertSameRoots(testCase, actual, expected, 1e-8)
        end

        function diagonalIntersect(testCase)
            f1 = chebfun3(@(x,y,z) x + y + z - 1);
            f2 = chebfun3(@(x,y,z) x - y);
            f3 = chebfun3(@(x,y,z) y - z);
    
            expected = [ 1/3  1/3  1/3 ];
    
            actual = roots_xyz(f1,f2,f3);
            assertSameRoots(testCase,actual,expected,1e-14)
        end

        function multipleRoot(testCase)
            f1 = chebfun3(@(x,y,z) (x - 0.2).^2);
            f2 = chebfun3(@(x,y,z)  y - 0.3);
            f3 = chebfun3(@(x,y,z)  z + 0.4);
    
            expected = [ 0.2  0.3  -0.4 ];
            actual   = roots_xyz(f1,f2,f3);
    
            assertSameRoots(testCase,actual,expected,1e-8)
        end

        function cubicGrid(testCase)
            dom = [-1 1 -1 1 -1 1];
            f1 = chebfun3(@(x,y,z) (x+1).*x.*(x-1), dom);
            f2 = chebfun3(@(x,y,z) (y+1).*y.*(y-1), dom);
            f3 = chebfun3(@(x,y,z) (z+1).*z.*(z-1), dom);
    
            vals = [-1; 0; 1];
            [X,Y,Z] = ndgrid(vals,vals,vals);
            expected = [X(:) Y(:) Z(:)];
    
            actual = roots_xyz(f1,f2,f3);
            assertSameRoots(testCase,actual,expected,1e-12)
        end

        function onEdgeOfDomain(testCase)
            f1 = chebfun3(@(x,y,z) x-1);
            f2 = chebfun3(@(x,y,z) y);
            f3 = chebfun3(@(x,y,z) z);

            actual = roots_xyz(f1,f2,f3);
            expected = [1 0 0];
            
            assertSameRoots(testCase, actual, expected, 1e-8)
        end

        function largerDomain(testCase)
            f1 = chebfun3(@(x,y,z) x-1.5, [-2 2 -2 2 -2 2]);
            f2 = chebfun3(@(x,y,z) y+1.2, [-2 2 -2 2 -2 2]);
            f3 = chebfun3(@(x,y,z) z-0.1, [-2 2 -2 2 -2 2]);

            actual = roots_xyz(f1,f2,f3);
            expected = [1.5 -1.2 0.1];
            
            assertSameRoots(testCase, actual, expected, 1e-8)
        end

        function smallerDomain(testCase)
            % Should have no roots in this domain
            dom = [-0.5 0.5 -0.5 0.5 -0.5 0.5];
            f1 = chebfun3(@(x,y,z) x-0.9, dom);
            f2 = chebfun3(@(x,y,z) y, dom);
            f3 = chebfun3(@(x,y,z) z, dom);

            actual = roots_xyz(f1,f2,f3);
            expected = [];

            assertSameRoots(testCase, actual, expected, 1e-8)
        end

        function trigPolynomial(testCase)
            f1 = chebfun3(@(x,y,z) cos(pi*x) - 0.5);
            f2 = chebfun3(@(x,y,z) sin(pi*y));
            f3 = chebfun3(@(x,y,z) z.^2 - 0.25);
    
            x = 1/3;
            z = 0.5;
            
            expected = [  x   0   z;
                         -x   0   z;
                          x   0  -z;
                         -x   0  -z ];
    
            actual = roots_xyz(f1,f2,f3);
            assertSameRoots(testCase,actual,expected,1e-12)
        end

        function cosineSphere(testCase)
            f1 = chebfun3(@(x,y,z) cos(2*pi*x).*cos(2*pi*y).*cos(2*pi*z));
            f2 = chebfun3(@(x,y,z) y);
            f3 = chebfun3(@(x,y,z) x.^2 + y.^2 + z.^2 - 1);
    
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
    
            actual = roots_xyz(f1,f2,f3);
    
            assertSameRoots(testCase,actual,expected,1e-12);
        end

        function cosineSphereIsolated(testCase)
            % Simple-ish example isolated from the above where it fails to
            % find the roots
            f1 = chebfun3(@(x,y,z) cos(2*pi*x).*cos(2*pi*y).*cos(2*pi*z));
            f2 = chebfun3(@(x,y,z) y);
            f3 = chebfun3(@(x,y,z) x.^2 + y.^2 + z.^2 - 1);

            expected = [-sqrt(7)/4  0  1/4; -sqrt(7)/4  0 -1/4 ];
            actual = roots_xyz(f1,f2,f3);
    
            assertSameRoots(testCase,actual,expected,1e-12);
        end
    end
end

% not sure if this works for checking minima since the components might
% look the same but differ slightly numerically
function assertSameRoots(testCase,actual,expected,tol)
    if nargin<4, tol = 1e-8; end
    
    if isempty(expected); testCase.verifyTrue(isempty(expected)); return; end

    testCase.assertEqual(size(actual,1),size(expected,1), ...
        "Actual and expected root arrays must have the same number of rows");

    testCase.assertEqual(size(actual,2),size(expected,2), ...
        "Actual and expected root arrays must have the same number of columns");

    actual = sortrows(round(actual/tol)*tol);
    expected = sortrows(round(expected/tol)*tol);

    testCase.verifyEqual(actual, expected, "AbsTol", tol);
end
