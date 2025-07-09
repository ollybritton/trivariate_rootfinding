classdef RootfindingTest < matlab.unittest.TestCase
    methods(Test)
        function threeSpheres(testCase)
            n = 2;
            f1 = chebfun3(@(x,y,z) (x-1/sqrt(2)).^2 + (y-1/sqrt(2)).^2 + z.^2 - 1);
            f2 = chebfun3(@(x,y,z) (x+1/sqrt(2)).^2 + (y-1/sqrt(2)).^2 + z.^2 - 1);
            f3 = chebfun3(@(x,y,z) x.^2 + y.^2 + z.^2 - 1/2);

            actual = roots_xyz(f1,f2,f3,n);
            expected = [...
                0, sqrt(2)/4, sqrt(6)/4;
                0, sqrt(2)/4, -sqrt(6)/4;
            ];
            
            assertSameRoots(testCase, actual, expected, 1e-10)
        end

        function simpleIntersect(testCase)
            n = 2;
            f1 = chebfun3(@(x,y,z) x-0.1);
            f2 = chebfun3(@(x,y,z) y-0.1);
            f3 = chebfun3(@(x,y,z) z-0.1);

            actual = roots_xyz(f1,f2,f3,n);
            expected = [0.1 0.1 0.1;];
            
            assertSameRoots(testCase, actual, expected, 1e-15)
        end

        function severalIntersect(testCase)
            n = 2;
            f1 = chebfun3(@(x,y,z) x-0.1);
            f2 = chebfun3(@(x,y,z) y-0.1);
            f3 = chebfun3(@(x,y,z) (z-0.1).*(z+0.1));

            actual = roots_xyz(f1,f2,f3,n);
            expected = [0.1 0.1 -0.1; 0.1 0.1 0.1];
            
            assertSameRoots(testCase, actual, expected, 1e-8)
        end

        function cosineSphere(testCase)
            n = 6;
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
    
            actual = roots_xyz(f1,f2,f3,n);
    
            assertSameRoots(testCase,actual,expected,1e-12);
        end
    end
end

% not sure if this works for checking minima since the components might
% look the same but differ slightly numerically
function assertSameRoots(testCase,actual,expected,tol)
    if nargin<4, tol = 1e-15; end
    
    testCase.assertEqual(size(actual,2),size(expected,2), ...
        "Actual and expected root arrays must have the same number of columns");

    actual = sortrows(actual);
    expected = sortrows(expected);

    testCase.verifyEqual(actual, expected, "AbsTol", tol);
end
