function rts = roots_xyz(f1,f2,f3,n)

z_roots = roots_z(f1,f2,f3,n);
rts = [];

disp('Have solved for z, now finding x & y:')
tic


for i=1:size(z_roots,1)
    if (isinf(z_roots(i)) || ~isreal(z_roots(i)) ); continue; end
    h1 = f1(:,:,z_roots(i));
    h2 = f2(:,:,z_roots(i));

    roots_xy = roots(h1, h2);
    rts = [rts; [roots_xy (z_roots(i) * ones(size(roots_xy, 1),1))]];
end

toc

% Perform Newton update for each root as in root.m, can probably be
% vectorised
disp('Performing Newton update on each root:')
tic
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
toc

% Cluster nearby points
% TODO: roots.m does this before a Newton update; this probably saves work
% but leaves the possiblity that we introduce duplicates where nearby
% approximate roots converge to the same root
disp('Clustering nearby points:')
tic
tol = 10*sqrt(1e-12); % TODO: same as roots.m, should it be cube root instead?
rts = uniquetol(rts, tol, 'ByRows', true, 'DataScale', [1 1 1]); % TODO: differs from roots.m, not necessarily the "best" root in each cluster
toc

% Remove spurious solutions
disp('Removing spurios solutions')
tic
if size(rts,1) > 0
    threshold=1e-4;
    sols = (abs(f1(rts(:,1),rts(:,2),rts(:,3))) < threshold) & (abs(f2(rts(:,1),rts(:,2),rts(:,3))) < threshold) & (abs(f3(rts(:,1),rts(:,2),rts(:,3))) < threshold);
    roots_final = rts(sols,:);
    rts = roots_final;
end
toc

end