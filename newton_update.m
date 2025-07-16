function rts = newton_update(f1, f2, f3, rts)
    tol = 1e-12;
    Jac = jac(f1,f2,f3);
            
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