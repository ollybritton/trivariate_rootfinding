function [rts, R, V, W, approx_err, eig_err] = roots_z(f1,f2,f3,a,b,max_degree)
    % TODO: is pertubation as follows necessary? It's done in Noferini-Nyman
    % f1 = @(x,y,z) f1(x,y,z) + eps*x.^n + eps*y.^n + eps*z.^n;
    % f2 = @(x,y,z) f2(x,y,z) + eps*x.^n + eps*y.^n + eps*z.^n;
    % f3 = @(x,y,z) f3(x,y,z) + eps*x.^n + eps*y.^n + eps*z.^n;

    scale = (b - a)/2;
    shift = (a + b)/2;
    
    remap = @(x,idx) scale(idx).*x + shift(idx);

    f1_hat = @(x,y,z) f1(remap(x,1),remap(y,2),remap(z,3));
    f2_hat = @(x,y,z) f2(remap(x,1),remap(y,2),remap(z,3));
    f3_hat = @(x,y,z) f3(remap(x,1),remap(y,2),remap(z,3));

    fprintf('Calculating Cayley resultant using degree: %d\n', max_degree);
    tic
    % Cayley resultant now requires the remapped functions
    [R, n_s1, n_s2, ~, ~, n_z, approx_err] = cayley_resultant(f1_hat,f2_hat,f3_hat,max_degree);
    toc

    % Scaling? This is what roots does
    % "scale as suggested as Van Dooren"
    R = R/norm(R, 'fro');
    
    disp('Finding where det(R(z)) = 0:')
    tic
    
    % TODO: the following is still a bit of a black box conceptually

    % If all the coefficients of the highest degree are zero, leave it out from
    % the linearization
    z_length = n_z;
    for i = flip(1:n_z)
        if (norm(R(:,:,i),'fro') == 0); z_length = i-1; else; break; end
    end
    
    if (z_length == 0)
        rts = [];
        V = []; W = [];
        lin_back_err = [];            % %% ADDED
    elseif (z_length <= 2)
        [V,D,W] = eig(R(:,:,1),-R(:,:,2));
        rts_unit = diag(D);           % %% ADDED: keep unit-z values
        % %% ADDED: residual-based backward error for degree-1 case
        R1 = R(:,:,1); R2 = R(:,:,2);
        m = length(rts_unit);
        lin_back_err_all = zeros(m,1);
        for i = 1:m
            vi = V(:,i);
            ri = (R1 + rts_unit(i)*R2) * vi;          % residual in unit coords
            nz_eff = z_length;                         % effective # z-slices used
            lin_back_err_all(i) = nz_eff * (norm(ri,2) / max(norm(vi,2), eps));
        end
    else
        n = n_s1*n_s2;
        % Compute the linearization matrices C1 and C2
        C1 = -2*eye(n*(z_length-1));
        C1(end-n+1:end,end-n+1:end) = 2*R(:,:,z_length);
        C1(1:n,1:n) = -eye(n);
    
        C2 = zeros(size(C1));
        C2(1:end-n,n+1:end) = eye(n*(z_length-2));
        C2(n+1:end,1:end-n) = C2(n+1:end,1:end-n)+eye(n*(z_length-2));
        C2(end-n+1:end,end-2*n+1:end-n) = -R(:,:,z_length);
    
        % Compute the last rows of the coefficient matrix C2
        D=[];
        for i = 1:z_length-1
            D = [D R(:,:,i)];
        end
        C2(end-n+1:end,:) = C2(end-n+1:end,:)+D;
        [V,D,W] = eig(C2,-C1);
        rts_unit = diag(D);           % %% ADDED: keep unit-z values
    
        % %% ADDED: residual-based backward error for higher degree
        m = length(rts_unit);
        lin_back_err_all = zeros(m,1);
        for i = 1:m
            vi = V(:,i);
            ri = (C2 - rts_unit(i)*C1) * vi;          % residual in unit coords
            nz_eff = z_length;                         % effective # z-slices used
            lin_back_err_all(i) = nz_eff * (norm(ri,2) / max(norm(vi,2), eps));
        end
    end
    
    % %% ADDED: map z to physical coords, then filter, and carry lin_back_err along
    if ~isempty(rts_unit)
        rts_phys = scale(3).*rts_unit + shift(3);     % physical z
        real_mask = (imag(rts_phys) == 0);
        interval_mask = a(3) <= real(rts_phys) & real(rts_phys) <= b(3);
        keep = real_mask & interval_mask;
    
        rts = real(rts_phys(keep));                   % returned roots (physical)
        V   = V(:, keep);
        W   = W(:, keep);
        lin_back_err = lin_back_err_all(keep);        % returned errors
    else
        rts = [];
        lin_back_err = [];
    end

    eig_err = lin_back_err;
    
    % Ignore the roots outside the required interval or those that are
    % imaginary

    rts = scale(3).*rts + shift(3);

    real_mask = imag(rts) == 0;
    rts = rts(real_mask);
    interval_mask = a(3) <= rts & rts <= b(3);
    rts = rts(interval_mask);

    if ~isempty(rts)
        V = V(real_mask,:);
        W = W(real_mask,:);
        V = V(interval_mask,:);
        W = W(interval_mask,:);
    end

    toc

end