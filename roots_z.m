function rts = roots_z(f1,f2,f3,n)

% Perturbation (is this necessary?)
% f1 = @(x,y,z) f1(x,y,z) + eps*x.^n + eps*y.^n + eps*z.^n;
% f2 = @(x,y,z) f2(x,y,z) + eps*x.^n + eps*y.^n + eps*z.^n;
% f3 = @(x,y,z) f3(x,y,z) + eps*x.^n + eps*y.^n + eps*z.^n;

disp('Calculating Cayley resultant:')
tic
[R, n_s1, n_s2, ~, ~, n_z] = cayley_resultant(f1,f2,f3,n);
toc

disp('Finding where det(R(z)) = 0:')
tic

% If all the coefficients of the highest degree are zero, leave it out from
% the linearization
z_length = n_z;
for i = flip(1:n_z)
    if (norm(R(:,:,i),'fro') < 1e-12); z_length = i-1; else; break; end
end

if (z_length == 0)
    rts = [];
elseif (z_length <= 2)
    [~,D] = eig(R(:,:,1),-R(:,:,2));
    rts = diag(D);
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

    % Solve the eigenproblem
    [~,D] = eig(C2,-C1);
    %     [~, D, C]=polyeig(C2,-C1);

    rts = diag(D);

    % Ignore the roots outside the interval [-1,1] (or the complex unit disk at
    % this point)
    rts = rts(abs(rts) < 1);

end
toc

end