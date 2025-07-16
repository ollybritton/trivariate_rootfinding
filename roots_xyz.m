function rts = roots_xyz(f1,f2,f3)
    domain = reshape(f1.domain, 2, 3);
    func_degree = max([length(f1) length(f2) length(f3)]); % Is this right?
    max_degree = min(5, func_degree); % TODO: chosen arbitrarily

    % TODO: also chosen arbitrarily here, should be calculated based on
    % analysing when it is best to subdivide
    % also obviously way too large
    subdivide_stop = 1.1; % this will subdivide at most once? 

    % TODO: also want to do domain overlook?
    % TODO: make newton polishing optional
    % TODO: multitol?

    a = domain(1,:);
    b = domain(2,:);

    % TODO: normalise to max value 1?
    
    % At the moment, the resultant is calculated by interpolating the
    % Cayley function, so subdividing this way doesn't make much sense. But
    % it means the scaffolding is there for when the resultant can be
    % calculated directly from the chebcoeff3 this will be better

    % TODO: another difference is that roots.m can subdivide one coordinate
    % direction at a time, this by default splits cubes into 8 which may be
    % wildly inefficient
    % TODO: roots.m also does this recursively but maybe it's better to do
    % iteratively?

    rts = roots_xyz_subregion(f1,f2,f3,a,b,max_degree,subdivide_stop);    
    
    % Perform Newton update for each root as in root.m, can probably be
    % vectorised
    % disp('Performing Newton update on each root:')
    % tic
    % rts = newton_update(f1,f2,f3,rts);
    % toc
    
    % Cluster nearby points
    % TODO: roots.m does this before a Newton update; this probably saves work
    % but leaves the possiblity that we introduce duplicates where nearby
    % approximate roots converge to the same root?
    disp('Clustering nearby points:')
    tic
    tol = 10*sqrt(1e-12); % TODO: same as roots.m, should it be cube root instead?
    rts = uniquetol(rts, tol, 'ByRows', true, 'DataScale', [1 1 1]); % TODO: differs from roots.m, not necessarily the "best" root in each cluster
    toc
    
    % Remove spurious solutions
    disp('Removing spurious solutions:')
    tic
    if size(rts,1) > 0
        threshold=1e-4;
        sols = (abs(f1(rts(:,1),rts(:,2),rts(:,3))) < threshold) & (abs(f2(rts(:,1),rts(:,2),rts(:,3))) < threshold) & (abs(f3(rts(:,1),rts(:,2),rts(:,3))) < threshold);
        roots_final = rts(sols,:);
        rts = roots_final;
    end
    toc

end