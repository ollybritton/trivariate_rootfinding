
function [a_sub, b_sub] = split_subregion(a,b,point)
    if nargin < 3 || isempty(point)
        r_x = (b(1)-a(1))/2*0.004849834917525 + (b(1)+a(1))/2;
        r_y = (b(2)-a(2))/2*-0.0005194318842611 + (b(2)+a(2))/2;
        r_z = (b(3)-a(3))/2*0.000238764512987345 + (b(3)+a(3))/2;

        point = [r_x r_y r_z];
    end


    n = 3; % number of dimensions
    m = 2^3; % number of subregions

    a_sub = zeros(m, n);
    b_sub = zeros(m, n);

    for k = 0:m-1
        bits = bitget(k, 1:n);
        left = bits == 0;
        right = bits == 1;

        a_sub(k+1, left)  = a(left);
        b_sub(k+1, left)  = point(left);

        a_sub(k+1, right) = point(right);
        b_sub(k+1, right) = b(right);
    end
end