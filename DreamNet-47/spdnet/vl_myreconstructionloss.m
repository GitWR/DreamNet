function Y = vl_myreconstructionloss(X, X_ori, gamma, dzdy)
  % this function is designed to implement the decode term with the reconstruction function
  % Date: 
  % Author: 
  % Coryright@
  
  % for m = 1 : length(X)
  %    dzdy{m} = single(zeros(size(X{1},1),size(X{1},1))); % 400 * 400
  % end

  dzdy_l3 = single(1);
  %gamma = 0.1; % needs to be adjusted 0.01
  dist_sum = zeros(1,length(X)); % save each pair dist
  Y = cell(length(X), 1); % save obj or dev
  dev_term = cell(1, length(X)); % save each pair' derivation 
  for i = 1 : length(X)
      temp = X{i} - X_ori{i};
      dev_term{i} = 2 * temp;
      dist_sum(i) = norm(temp,'fro') * norm(temp,'fro'); % the dist computed via LEM
  end
  if nargin < 4
      Y = gamma * (sum(dist_sum) / length(X)); % the obj og this loss function
  else
      for j = 1 : length(X)
          dev_l3 = bsxfun(@times, dev_term{j}, bsxfun(@times, ones(size(X{1},1)), dzdy_l3));
          Y{j} = gamma * dev_l3 + dzdy{j}; % the sum of rebuild term and softmax term, then push them into margin term
      end
  end
end

