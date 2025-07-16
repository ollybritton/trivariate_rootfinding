
function rts = roots_xyz_subregion(f1,f2,f3,a,b,max_degree,subdivide_stop)
    approx_tol = 1e-15; % also from roots.m

    dom = [a(1) b(1) a(2) b(2) a(3) b(3)];

    % Find the highest degree of the Chebyshev approximation for each of
    % the functions
    f1_resampled = chebfun3(f1, dom, "eps", approx_tol);
    f2_resampled = chebfun3(f2, dom, "eps", approx_tol);
    f3_resampled = chebfun3(f3, dom, "eps", approx_tol);
    highest_degree = max([length(f1_resampled) length(f2_resampled) length(f3_resampled)]);

    disp("Subregion:")
    disp([a, b]);
    disp("Highest degree of interpolant over this region:")
    disp(highest_degree)
    rts = [];

    if highest_degree > max_degree && all(abs(b - a) > subdivide_stop)
        disp("Subdividing!");
        [a_sub, b_sub] = split_subregion(a,b);

        for i=1:size(a_sub,1)
            rts = [rts; roots_xyz_subregion(f1,f2,f3,a_sub(i,:),b_sub(i,:),max_degree,subdivide_stop)];
        end
    else
        % Base case, no more subdivisions to do

        % The degree of approximation used should be the smaller one out of
        % the higher degree and the max degree

        n = min([max_degree, highest_degree]);

        z_roots = roots_z(f1,f2,f3,a,b,approx_tol,n);
        
        disp('Have solved for z, now finding x & y:')
        
        tic
        for i=1:size(z_roots,1)
            if (isinf(z_roots(i)) || ~isreal(z_roots(i)) ); continue; end
            h1 = f1(:,:,z_roots(i));
            h2 = f2(:,:,z_roots(i));

            roots_xy = roots(h1, h2);

            if isempty(roots_xy); continue; end

            % Remove roots outside region under consideration, otherwise
            % get lots of duplicate roots
            % Roots outside region in z direction already removed by
            % roots_z
            % TODO: not sure if this works
            mask = (a(1) <= roots_xy(:,1) & roots_xy(:,1) <= b(1)) ...
                & (a(2) <= roots_xy(:,2) & roots_xy(:,2) <= b(2));
            roots_xy = roots_xy(mask,:);

            rts = [rts; [roots_xy (z_roots(i) * ones(size(roots_xy, 1),1))]];
        end
        toc
    end
end