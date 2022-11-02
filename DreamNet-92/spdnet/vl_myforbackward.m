function [res, su] = vl_myforbackward(net, x, dzdy, res, epoch, count1, varargin)
% vl_myforbackward  evaluates a simple SPDNet

opts.res = [] ;
opts.conserveMemory = false ;
opts.sync = false ;
opts.disableDropout = false ;
opts.freezeDropout = false ;
opts.accumulate = false ;
opts.cudnn = true ;
opts.skipForward = false;
opts.backPropDepth = +inf ;
opts.epsilon = 1e-4; % this parameter is worked in the RegLayer

mid = cell(length(x),1);
for k = 1 : length(x)
    mid{k} = zeros(43,43);
end

n = numel(net.layers) ; % count the number of layers

if (nargin <= 2) || isempty(dzdy)
  doder = false ;
else
  doder = true ; % this variable is used to control when to compute the derivative
end

if opts.cudnn
  cudnn = {'CuDNN'} ;
else
  cudnn = {'NoCuDNN'} ;
end

gpuMode = isa(x, 'gpuArray') ;

if nargin <= 3 || isempty(res)
  res = struct(...
    'x', cell(1,n+1), ...
    'dzdx', cell(1,n+1), ... % this gradient is necessary for computing the gradients in the layers below and updating their parameters  
    'dzdw', cell(1,n+1), ... % this gradient is required for updating W
    'aux', cell(1,n+1), ...
    'time', num2cell(zeros(1,n+1)), ...
    'backwardTime', num2cell(zeros(1,n+1))) ;
end
if ~opts.skipForward
  res(1).x = x ;
end

flag = zeros(1,length(x));
su = cell(1,n);
% -------------------------------------------------------------------------
%                                                              Forward pass
% -------------------------------------------------------------------------

for i=1:n
  if opts.skipForward
      break; 
  end
  l = net.layers{i} ; % each net layer store two contents: (1) layer type (2) weight
  res(i).time = tic ; % count the time spend on each layer
  switch l.type
    case 'bfc'
      temp = vl_mybfc(res(i).x, l.weight, i, res) ; % the output data of each layer is stored in the x part
      for j = 1 : length(temp)
          if isnan(temp{j})
              flag(j) = j;
          end
      end
      if (sum(flag)~=0)
          ind = find(flag~=0);
          temp(ind) = [];
          for k = 1 : i
              trans = res(k).x;
              if iscell(trans)
                 res(k).x(ind) = [];
              else
                  trans(:,ind) = [];
                  res(k).x = trans;
              end
          end
      end
      su{i} = flag;
      flag = zeros(1,length(x)); % after finished, set it to zero
      res(i+1).x = temp;
    
    case 'add'
      res(i+1).x = vl_myadd(res(i).x,res(i-7).x);
    
    case 'fc'
      res(i+1).x = vl_myfc(res(i).x, l.weight) ;
    case 'rec'
      res(i+1).x = vl_myrec(res(i).x, opts.epsilon) ;
    case 'marginloss'
      res(i+1).obj = 0.0; % vl_mymarginloss(res(i).x, l.class, epoch, count1,su) ;  % this is the new loss function
      res(i+1).x = res(i).x;
    case 'reconstructionloss'
      if i == 92
          data_recon = res(i-87).x; % 7  8
          gamma = 0.001; % 0.001
      elseif i == 11
          data_recon = res(i-6).x; 
          gamma = 0.0;
      else
          data_recon = res(i-7).x; % 6  7
          gamma = 0.0;
      end
      res(i+1).obj = vl_myreconstructionloss(res(i).x, data_recon, gamma); % may be adjusted    res(i-6).x
      res(i+1).x = res(i).x;
    case 'log'
      res(i+1).x = vl_mylog(res(i).x) ;
    case 'softmaxloss'
      res(i+1).obj = vl_mysoftmaxloss(res(i).x, l.class, su) ; % 0.0;
      res(i+1).x = res(i-2).x;
    case 'custom'
          res(i+1) = l.forward(l, res(i), res(i+1)) ;
    otherwise
      error('Unknown layer type %s', l.type) ;
  end
  % optionally forget intermediate results
  forget = opts.conserveMemory ;
  forget = forget & (~doder || strcmp(l.type, 'relu')) ;
  forget = forget & ~(strcmp(l.type, 'loss') || strcmp(l.type, 'softmaxloss')) ;
  forget = forget & (~isfield(l, 'rememberOutput') || ~l.rememberOutput) ;
  if forget
    res(i).x = [] ;
  end
  if gpuMode & opts.sync
    % This should make things slower, but on MATLAB 2014a it is necessary
    % for any decent performance.
    wait(gpuDevice) ;
  end
  res(i).time = toc(res(i).time) ;
end

% -------------------------------------------------------------------------
%                                                             Backward pass
% -------------------------------------------------------------------------

if doder
  %res(n+1).dzdx = dzdy ; % the right hand first part of eq.6 in SPDNet. Here, its value is 1
  for i = n:-1:max(1, n-opts.backPropDepth+1) % calculate the derivate in reversed order
    l = net.layers{i} ;
    res(i).backwardTime = tic ;
    switch l.type
      case 'bfc'
        [res(i).dzdx, res(i).dzdw] = ... % all the data in a given batch share the same weight
             vl_mybfc(res(i).x, l.weight, i, res, res(i+1).dzdx) ; % here, z seems like the loss function in the corresponding layer
                                                           % res(i+1).dzdx is try to express the Eq.(6) in SPDNet
      
      
      case 'add'
        res(i).dzdx = vl_myadd(res(i).x, res(i-7).x, res(i+1).dzdx);
        
      case 'fc'
        [res(i).dzdx, res(i).dzdw]  = ...
              vl_myfc(res(i).x, l.weight, res(i+1).dzdx) ; 
      case 'rec'
        res(i).dzdx = vl_myrec(res(i).x, opts.epsilon, res(i+1).dzdx) ;
      case 'marginloss'
        temp = res(i).x;
        mid_sc = cell(length(temp),1);
        for ii = 1 : length(x)
            mid_sc{ii} = zeros(size(temp{1},1),size(temp{1},2));
        end
        if i == 87
            dev_sc = mid_sc;
        else
            dev_sc = res(i+8).dzdx;
        end
        dev_log = res(i+1).dzdx;
        res(i).dzdx = vl_mymarginloss(res(i).x, l.class, epoch, count1, su, dev_log, res(i+4).dzdx, dev_sc) ;
      case 'reconstructionloss'
        %res(i+1).dzdx = mid;       
        if i == 92
            data_recon = res(i-87).x; % 88 
            gamma = 0.001;
            res(i+1).dzdx = mid;
        elseif i == 11
            data_recon = res(i-6).x; % 7
            gamma = 0.0;
        else
            data_recon = res(i-7).x; % 8
            gamma = 0.0;
        end
        res(i).dzdx = vl_myreconstructionloss(res(i).x, data_recon, gamma, res(i+1).dzdx) ; % res(i+1).dzdx 
      case 'log'
        res(i).dzdx = vl_mylog(res(i).x, res(i+1).dzdx) ;
      case 'softmaxloss'
        %res(i+1).dzdx = dzdy;
        res(i).dzdx = vl_mysoftmaxloss(res(i).x, l.class, su, 1) ; % res(i+1).dzdx
      case 'custom'
        res(i) = l.backward(l, res(i), res(i+1));
    end
    if opts.conserveMemory
      res(i+1).dzdx = [] ;
    end
    if gpuMode && opts.sync
      wait(gpuDevice) ;
    end
    res(i).backwardTime = toc(res(i).backwardTime) ;
  end
end

