function rts = newton_update(f1, f2, f3, rts)
    tol = 1e-12;
    
    [diffF1_1, diffF1_2, diffF1_3] = grad(f1);
    [diffF2_1, diffF2_2, diffF2_3] = grad(f2);
    [diffF3_1, diffF3_2, diffF3_3] = grad(f3);
    
    Jac = @(x,y,z) [feval(diffF1_1, x, y, z),  feval(diffF1_2, x, y, z),  feval(diffF1_3, x, y, z)
                    feval(diffF2_1, x, y, z),  feval(diffF2_2, x, y, z),  feval(diffF2_3, x, y, z)
                    feval(diffF3_1, x, y, z),  feval(diffF3_2, x, y, z),  feval(diffF3_3, x, y, z)];
            
    for ns=1:size(rts,1)
        r = rts(ns,:);
        update = 1; 
        iter = 1;
        while ( ( norm(update) > 10*tol ) && ( iter < 10 ) )
            update = Jac(r(1), r(2), r(3)) \ [ feval(f1, r(1), r(2), r(3)); 
                                               feval(f2, r(1), r(2), r(3)); 
                                               feval(f3, r(1), r(2), r(3)) ];
            r = r - update.'; 
            iter = iter + 1;
        end
    
        rts(ns,:) = r;
    end
end