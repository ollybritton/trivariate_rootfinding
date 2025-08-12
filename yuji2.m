%% Empirical study of window size vs. accuracy in 2d
% Yuji Nakatsukasa, Oxford, July 2025

%%
% At last meeting with Olly and Mr. Director of Oxford Edge, we weren't
% sure if 'local refinement' of reducing the window size by a factor $C$ in
% $d$-dimensional rootfinding would improve the accuracy by $C$ or $C^d$,
% or something else. In the latter case, there would be significant room to
% overcome the instability described by Graf and Townsend. But Olly's
% experiments weren't indicating that, so here we try explore this
% empirically, in $d=2$ as the code there is impressively robust. 

%%
% Using Chebfun2's roots and Graf-Townsend's example, we test how many
% digits a resultant-based method is able to get. Since we want to explore
% the 'worst-case performance' that suffers the most from ill-conditioning,
% we'd like to run many examples with the 'same' conditioning and look at
% the worst-case performance. We do this by applying a random shift to x
% and y, solving the problem, and shifting back at the end. 

warning off % supress many warning messages
sig = 0.001; 
shiftx = rand/100; % shift in x
shifty = rand/100; % shift in y
f = chebfun2(@(x,y) (x-shiftx)^2-sig*(y-shifty),[-1 1 -1 1]); 
g = chebfun2(@(x,y) (y-shifty)^2-sig*(x-shiftx),[-1 1 -1 1]); 
r = roots(f,g);

%%
% That was a single run. We now run it many times and record the result, 
% also changing the window size. 

MS = 'Markersize'; LW = 'linewidth'; FS = 'fontsize'; CO = 'Color'; TEX = 'interpreter'; tex = 'latex';
lw = 2; ms = 12; fs = 14; 

clf
Rs = [0.01 0.02 0.035 0.06 0.1 0.2 0.35 0.6 1 2 3.5 6 10];
errmax = []; % record largest error
for R = Rs
    err = [];
    for ii = 1:10
        shiftx = randn/100; % shift in x
        shifty = randn/100; % shift in y

        f = chebfun2(@(x,y) (x-shiftx)^2-sig*(y-shifty) ,[shiftx-R shiftx+R shifty-R shifty+R]);
        g = chebfun2(@(x,y) (y-shifty)^2-sig*(x-shiftx) ,[shiftx-R shiftx+R shifty-R shifty+R]);
        r = roots(f,g);
        err = [err r(1,:)-[shiftx shifty]];
        loglog(R,err,'k.',MS,14), grid on
        hold on
    end
    errmax = [errmax max(err)]; 
end
shg
xlabel('window size',FS,fs)
ylabel('accuracy',FS,fs)

const = errmax(1)/(Rs(1)^2);
loglog(Rs,Rs.^2*const,'k--')
text(Rs(end),Rs(end).^2*const,'O(w^2)',FS,fs)

const = errmax(1)/(Rs(1)^3);
loglog(Rs,Rs.^3*const,'r--')
text(Rs(end),Rs(end).^3*const,'O(w^3)',FS,fs,'color','r')
shg
set(gca,FS,fs)

%%
% I think the trend is at least $error=(window^2)$, possibly higher. 
% I think this suggests that in higher dimensions, we can hope to get even
% better accuracy. It also suggests the somewhat hand-wavy argument in the
% N-Noferini-Townsend paper is not sharp and probably improvable. 
