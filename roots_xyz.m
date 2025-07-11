function rts = roots_xyz(f1,f2,f3)
    domain = reshape(f1.domain, 2, 3);
    func_degree = max([length(f1) length(f2) length(f3)]); % Is this right?
    max_degree = min(6, func_degree); % TODO: chosen arbitrarily

    % TODO: also want to do domain overlook?
    % TODO: make newton polishing optional
    % TODO: multitol?

    a = domain(1,:);
    b = domain(2,:);

    % TODO: normalise to max value 1?
    
    [a_sub, b_sub] = split_subregion(a,b);
    rts = [];

    for i=1:size(a_sub,1)
        rts = [rts; roots_xyz_subregion(f1,f2,f3,a_sub(i,:),b_sub(i,:),max_degree)];
    end
    disp(rts);
    
    
    % Perform Newton update for each root as in root.m, can probably be
    % vectorised
    disp('Performing Newton update on each root:')
    tic
    rts = newton_update(f1,f2,f3,rts);
    toc
    
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


function rts = roots_xyz_subregion(f1,f2,f3,a,b,max_degree)
    z_roots = roots_z(f1,f2,f3,a,b,max_degree);
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

end