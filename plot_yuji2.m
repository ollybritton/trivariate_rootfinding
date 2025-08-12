%% Empirical study of window size vs. accuracy in 2‑D  (random Q + manual rescale + yuji‑y)
% Yuji Nakatsukasa, Oxford, July 2025

warning off                         % suppress many warning messages
sig  = 1e-7;                        % σ as before

MS = 'Markersize'; LW = 'linewidth'; FS = 'fontsize';
lw = 2; ms = 12; fs = 14;

clf
Rs      = [0.0000001 0.000001 0.00001 0.0001 0.001 0.01 0.02 0.035 0.06 0.1 0.2 0.35 0.6 1 2 3.5 6 10];
errmax  = [];

%%  random 2×2 orthogonal matrix Q --------------------------------
theta = 2*pi*rand;         % uniform angle
Q = [cos(theta)  -sin(theta);
     sin(theta)   cos(theta)];

for R = Rs
    err = [];
    for ii = 1:30                  % 20 random trials per window size

        %%  random centre of the search square ---------------------------
        shiftx = randn/100;
        shifty = randn/100;

        %%  affine maps  u,v ∈ [-1,1]  ->  physical x,y ------------------
        xmap = @(u) shiftx + R*u;   % half‑width R ⇒ side length 2R
        ymap = @(v) shifty + R*v;

        %%  polynomials on reference square ------------------------------
        F = chebfun2(@(u,v) ...
                 (xmap(u)-shiftx).^2 + ...
                 sig.*( Q(1,1).*(xmap(u)-shiftx) + ...
                        Q(1,2).*(ymap(v)-shifty) ), ...
                 [-1 1 -1 1]);

        G = chebfun2(@(u,v) ...
                 (ymap(v)-shifty).^2 + ...
                 sig.*( Q(2,1).*(xmap(u)-shiftx) + ...
                        Q(2,2).*(ymap(v)-shifty) ), ...
                 [-1 1 -1 1]);

        %%  roots: yuji returns ONLY v‑coordinates in reference domain ----
        v_roots = yuji(F, G);                 % m×1 vector of v values
        y_roots = ymap(v_roots);              % physical y positions

        if ~isempty(y_roots)
            dy  = abs( y_roots - shifty );    % error in y only
            err = [err ; min(dy)];            % keep the nearest one
            loglog(R, min(dy), 'k.', MS,14), grid on; hold on
        end
    end
    errmax = [errmax ; max(err)];
end
shg
xlabel('window size',FS,fs)
ylabel('accuracy  (|y\_{root}-y\_{true}|)',FS,fs)

%%  guide‑lines:  O(w), O(w²), O(w³) -------------------------------------
const = errmax(1)/(Rs(1));
loglog(Rs, Rs * const, 'b--')
text(Rs(end), Rs(end) * const, 'O(w)', FS,fs,'color','b')

const = errmax(1)/(Rs(1)^2);
loglog(Rs, Rs.^2 * const, 'k--')
text(Rs(end), Rs(end).^2 * const, 'O(w^2)', FS,fs)

const = errmax(1)/(Rs(1)^3);
loglog(Rs, Rs.^3 * const, 'r--')
text(Rs(end), Rs(end).^3 * const, 'O(w^3)', FS,fs,'color','r')

set(gca,FS,fs)
shg

%%
% Each trial now has a fresh orientation given by Q, so the statistics
% capture both random shifts and random rotations of the coordinate axes.
