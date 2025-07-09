function rts = roots_xyz(f1,f2,f3,n)

z_roots = roots_z(f1,f2,f3,n);
rts = [];

disp('Finding all first and second components:')
tic

[l1,~] = size(z_roots);

% TODO: Choose in which order you solve z, y and x based on the degrees to
% minimize running time
for i=1:l1
    if (isinf(z_roots(i)) || ~isreal(z_roots(i)) ); continue; end
    h1 = f1(:,:,z_roots(i));
    h2 = f2(:,:,z_roots(i));

    roots_xy = roots(h1, h2);
    rts = [rts; [roots_xy (z_roots(i) * ones(size(roots_xy, 1),1))]];

    % rts = [rts; roots(h1,h2)];
    % y_roots = bivariate_rootfinder(h1,h2,n);
    % [l2,~] = size(y_roots);
    % 
    % for j=1:l2
    %     if (isinf(y_roots(j)) || ~isreal(y_roots(j))); continue; end
    %     g1 = @(x) h1(x,y_roots(j));
    %     x_roots = univariate_rootfinder(g1,n); %The case of complex roots for x is not handled
    %     threshold = 1e-2;
    %     sols = (abs(f1(x_roots,y_roots(j),z_roots(i))) < threshold) & (abs(f2(x_roots,y_roots(j),z_roots(i))) < threshold) & (abs(f3(x_roots,y_roots(j),z_roots(i))) < threshold);
    %     k = find(sols);
    %     if (size(k,1) ~= 0)
    %         rts = [rts; x_roots(k) repmat(y_roots(j),size(k,1),1) repmat(z_roots(i),size(k,1),1)];
    %     end
    % end
end

toc

% Perform Newton update for each root as in root.m, can probably be
% vectorised
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

% Cluster nearby points
% TODO: roots.m does this before a Newton update; this probably saves work
% but leaves the possiblity that we introduce duplicates where nearby
% approximate roots converge to the same root
tol = 10*sqrt(1e-12); % TODO: same as roots.m, should it be cube root instead?
rts = uniquetol(rts, tol, 'ByRows', true, 'DataScale', [1 1 1]); % TODO: differs from roots.m, not necessarily the "best" root in each cluster

% Remove spurious solutions
if size(rts,1) > 0
    threshold=1e-4;
    sols = (abs(f1(rts(:,1),rts(:,2),rts(:,3))) < threshold) & (abs(f2(rts(:,1),rts(:,2),rts(:,3))) < threshold) & (abs(f3(rts(:,1),rts(:,2),rts(:,3))) < threshold);
    k = find(sols);
    roots_final = rts(k,:);
    rts = roots_final;
end

end