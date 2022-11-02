function Y = vl_myadd(X, X_sc, dzdy)
   % VL_MYADD: defiend to implement the shortcut connection 
   % 此处显示详细说明
   Y = cell(length(X),1);
   for ix = 1 : length(X)
       temp = X{ix} + X_sc{ix};
       Y{ix} = temp;
   end
   if nargin == 3
       for ix = 1 : length(X)
           Y{ix} = dzdy{ix};
       end
   end
end

