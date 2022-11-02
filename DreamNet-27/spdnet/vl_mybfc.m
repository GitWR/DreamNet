function [Y, Y_w] = vl_mybfc(X, W, count, res, dzdy)
%[DZDX, DZDF, DZDB] = VL_MYBFC(X, F, B, DZDY)
%BiMap layer  
Y = cell(length(X),1); % used to store the result data of bamap layer 

for ix = 1 : length(X)
    Y{ix} = W' * X{ix} * W ;
end


if nargin == 5
    [dim_ori, dim_tar] = size(W);
    Y_w = zeros(dim_ori,dim_tar);
    for ix = 1  : length(X)
        if iscell(dzdy)==1
            d_t = dzdy{ix};
        else
            d_t = dzdy; %(:,ix);
            % d_t = reshape(d_t,[dim_tar dim_tar]);
        end
        Y{ix} = W * d_t * W'; % partial derivative, dz/dx
        Y_w = Y_w + 2 * X{ix} * W * d_t; % partial derivative, dz/dw
    end
end
