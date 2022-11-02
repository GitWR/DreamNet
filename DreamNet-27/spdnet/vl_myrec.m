function Y = vl_myrec(X, epsilon, dzdy)
% Y = VL_MYREC (X, EPSILON, DZDY)
% ReEig layer

Us = cell(length(X),1);
Ss = cell(length(X),1);
Vs = cell(length(X),1);
% thres = 0.0; % we first try do not use thres for activation 1e-4
min_v = zeros(1,30); 
for ix = 1 : length(X)
    temp = X{ix}; % get each sample
    [Us{ix},Ss{ix},Vs{ix}] = svd(temp);
    min_v(ix) = min(min(diag(Ss{ix})));
end

D = size(Ss{1},2);
Y = cell(length(X),1);

if nargin < 3
    for ix = 1:length(X)
        [max_S, ~]=max_eig(Ss{ix},epsilon); % the equation try to perform relu in SPDNet
        Y{ix} = Us{ix}*max_S*Us{ix}'; % use the modified eig to re-build the data in this layer
    end
else
    for ix = 1:length(X)
        U = Us{ix}; S = Ss{ix}; V = Vs{ix};

        Dmin = D;
        
        dLdC = double(dzdy{ix}); dLdC = symmetric(dLdC); % the same as the operation in log layer
        
        [max_S, max_I] = max_eig(Ss{ix},epsilon); % the equation try to perform relu in SPDNet 
        dLdV = 2*dLdC*U*max_S;
        dLdS = (diag(not(max_I)))*U'*dLdC*U; % see eq.18 in SPDNet
        
        
        K = 1./(diag(S)*ones(1,Dmin)-(diag(S)*ones(1,Dmin))'); % the same as the operation in log layer
        K(eye(size(K,1))>0)=0; % eq.14 in spdnet
        K(find(isinf(K)==1))=0; 
        
        dzdx = U*(symmetric(K'.*(U'*dLdV))+dDiag(dLdS))*U'; % the same as the operation in log layer
        
        Y{ix} =  dzdx; % warning('no normalization');
    end
end
