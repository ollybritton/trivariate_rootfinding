function v = cheby_5D_evaluate(C, s1,s2,t1,t2,z)
% Evaluate a 5‑D Chebyshev series with coefficient tensor C (size
% n_s1 × n_s2 × n_t1 × n_t2 × n_z) at *vectorised* points
% (s1,s2,t1,t2,z) ∈ [‑1,1]^5  of equal length.
%
% Uses a tensor‑product of 1‑D Clenshaw recurrences; compact but
% perfectly vectorised.

    ns1 = size(C,1);   ns2 = size(C,2);
    nt1 = size(C,3);   nt2 = size(C,4);
    nz  = size(C,5);

    % 1‑D Chebyshev matrices  (degree × #pts)
    T1 = cos((0:ns1-1)'.*acos(s1'));     %  ns1 × N
    T2 = cos((0:ns2-1)'.*acos(s2'));
    T3 = cos((0:nt1-1)'.*acos(t1'));
    T4 = cos((0:nt2-1)'.*acos(t2'));
    T5 = cos((0:nz -1)'.*acos(z'));

    % Contract the tensor:   v = Σ_{k₁...k₅} Cₖ₁…ₖ₅  T₁ₖ₁ T₂ₖ₂ … T₅ₖ₅
    v = zeros(1,numel(s1));
    for k5 = 1:nz
        for k4 = 1:nt2
            for k3 = 1:nt1
                for k2 = 1:ns2
                    coeff_slice = squeeze(C(:,k2,k3,k4,k5)).'; % 1 × ns1
                    v = v + (coeff_slice * T1) .* T2(k2,:) .* T3(k3,:) ...
                                                  .* T4(k4,:) .* T5(k5,:);
                end
            end
        end
    end
    v = v.';          % column vector, like inputs
end
