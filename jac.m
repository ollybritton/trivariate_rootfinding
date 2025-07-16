function J = jac(f1, f2, f3)
    [diffF1_1, diffF1_2, diffF1_3] = grad(f1);
    [diffF2_1, diffF2_2, diffF2_3] = grad(f2);
    [diffF3_1, diffF3_2, diffF3_3] = grad(f3);
    
    J = @(x,y,z) [feval(diffF1_1, x, y, z),  feval(diffF1_2, x, y, z),  feval(diffF1_3, x, y, z)
                    feval(diffF2_1, x, y, z),  feval(diffF2_2, x, y, z),  feval(diffF2_3, x, y, z)
                    feval(diffF3_1, x, y, z),  feval(diffF3_2, x, y, z),  feval(diffF3_3, x, y, z)];
end