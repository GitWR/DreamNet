function [net, info] = spdnet_train_afew(net, spd_train, opts)

opts.errorLabels = {'top1e'};
opts.train = find(spd_train.spd.set==1) ; % 1 represents the training samples
opts.val = find(spd_train.spd.set==2) ; % 2 indicates the testing samples
count = 0;
acc_test = [];

for epoch = 1 : opts.numEpochs 
    
    learningRate = opts.learningRate(epoch); % the lr rate is different in each epoch
    
    
    %% fast-forward to last checkpoint
     modelPath = @(ep) fullfile(opts.dataDir, sprintf('net-epoch-%d.mat', ep));
     modelFigPath = fullfile(opts.dataDir, 'net-train.pdf') ;
     if opts.continue
         if exist(modelPath(epoch),'file')
             if epoch == opts.numEpochs
                 load(modelPath(epoch), 'net', 'info') ;
             end
             continue ;
         end
         if epoch > 1 % > 1
             fprintf('resuming by loading epoch %d\n', epoch-1) ; % epoch-1
             load(modelPath(epoch-1), 'net', 'info') ; % epoch-1
         end   
     end
     
    train = opts.train(randperm(length(opts.train))) ; % data_label; shuffle, to make the training data feed into the net in dis order
    val = opts.val; % the train data is in order to pass the net
    [net,stats.train] = process_epoch(opts, epoch, spd_train, train, learningRate, net) ;
    [net,stats.val] = process_epoch(opts, epoch, spd_train, val, 0, net) ; 
    
   %% the following is to dynamicly draw the cost curve
     evaluateMode = 0;
     if evaluateMode
         sets = {'train'};
     else
         sets = {'train', 'val'};
     end
     for f = sets
         f = char(f);
         n = numel(eval(f));
         info.(f).objective(epoch) = (stats.(f)(2)+ stats.(f)(3)+ stats.(f)(4) + stats.(f)(5) + stats.(f)(6) + stats.(f)(7)) / n; % stats.(f)(2) / n;+ stats.(f)(3) + stats.(f)(4) + stats.(f)(5) + stats.(f)(6) + stats.(f)(7) + stats.(f)(8) + stats.(f)(9)
         info.(f).acc(:,epoch) = stats.(f)(8:end) / n; % stats.(f)(3:end) / n;
     end
     
     if ~evaluateMode, save(modelPath(epoch), 'net', 'info') ; end % save the learned objective and acc-error of the current epoch
    
     if epoch >= 1
     figure(1);
     clf;
     hasError = 1;
     subplot(1,hasError+1,1);
     if ~evaluateMode
         semilogy(1:epoch,info.train.objective,'.--','linewidth',2);
     end
     grid on;
     h = legend(sets);
     set(h,'color','none');
     xlabel('training epoch');
     ylabel('cost value');
     title('objective');
     if hasError
         subplot(1,2,2);
         leg={};
         plot(1:epoch,info.val.acc','.--','linewidth',2); % info.val.acc'
         leg = horzcat(leg,strcat('val')); % ,opts.errorLabels
         set(legend(leg{:}),'color','none');
         grid on;
         xlabel('training epoch');
         ylabel('error');
         title('error')
     end
     drawnow;
     print(1,modelFigPath,'-dpdf');
     end
end  
    
    
function [net,stats] = process_epoch(opts, epoch, spd_train, trainInd, learningRate, net)

training = learningRate > 0 ;
count1 = 0;

if training
    mode = 'training' ; 
else
    mode = 'validation' ; 
end

stats = [0 ; 0; 0; 0; 0; 0; 0; 0] ; % for softmax 

numGpus = numel(opts.gpus) ;
if numGpus >= 1
    one = gpuArray(single(1)) ;
else
    one = single(1) ;
end

batchSize = opts.batchSize;
errors = 0;
numDone = 0 ;
flag = 0;

for ib = 1 : batchSize : length(trainInd) % select the training samples. Here, 10 pairs of samples per group
    flag = flag + 1;
    fprintf('%s: epoch %02d: batch %3d/%3d:', mode, epoch, ib,length(trainInd)) ;
    batchTime = tic ; %
    res = [];
    if (ib+batchSize> length(trainInd))
        batchSize_r = length(trainInd)-ib+1;  %
    else
        batchSize_r = batchSize;
    end
    spd_data = cell(batchSize_r,1); % store the data in each batch    
    spd_label = zeros(batchSize_r,1); % store the label of the data in each batch
    for ib_r = 1 : batchSize_r
      
        spdPath = [spd_train.SpdDir '\' spd_train.spd.name{trainInd(ib+ib_r-1)}];
        %spdPath = [spd_train.spd.name{trainInd(ib+ib_r-1)}];
        load(spdPath);
        spd_data{ib_r} = temp_2;
        spd_label(ib_r) = spd_train.spd.label(trainInd(ib+ib_r-1));
        
    end
    net.layers{6}.class = spd_label; % the label of the data in the log-layer
    net.layers{9}.class = spd_label; % (first softmax) one-hot vector is used to justify your algorithm generate a right label or not 
    net.layers{15}.class = spd_label;
    net.layers{18}.class = spd_label; % (second softmax) 
    net.layers{24}.class = spd_label; 
    net.layers{27}.class = spd_label; 
    
    %forward/backward spdnet
    if training
        dzdy = one; 
    else
        dzdy = [] ;
    end
    
    [res, su] = vl_myforbackward(net, spd_data, dzdy, res, epoch, count1) ; % Currently, I have almostly know its for-and-back-ward process
    for mm = 1 : length(su)
        temp_label = su{mm};
        if (sum(temp_label) ~= 0)
            ind = find(temp_label~=0);
            spd_label(ind) = [];
        end
    end
    
    %% accumulating graidents
    if numGpus <= 1
      net = accumulate_gradients(opts, learningRate, batchSize_r, net, res) ;
    else
      if isempty(mmap)
        mmap = map_gradients(opts.memoryMapFile, net, res, numGpus) ;
      end
      write_gradients(mmap, net, res) ;
      labBarrier() ;
      net = accumulate_gradients(opts, learningRate, batchSize_r, net, res, mmap) ;
    end
          
    %% classification
    predictions = gather(res(end-3).x) ;
    [~,pre_label] = sort(predictions, 'descend') ;
    error1 = sum(~bsxfun(@eq, pre_label(1,:)', spd_label)) ;
     
    predictions = gather(res(end-12).x) ;
    [~,pre_label] = sort(predictions, 'descend') ;
    error2 = sum(~bsxfun(@eq, pre_label(1,:)', spd_label)) ;
    
    predictions = gather(res(end-21).x) ;
    [~,pre_label] = sort(predictions, 'descend') ;
    error3 = sum(~bsxfun(@eq, pre_label(1,:)', spd_label)) ;
    
    error = min([error1,error2,error3]); %
    
    errors = errors + error;
    batchTime = toc(batchTime) ;
    speed = batchSize/batchTime ;
    numDone = numDone + batchSize_r ;
    
    stats = stats+[batchTime ; res(10).obj; res(19).obj; res(28).obj; res(12).obj; res(21).obj; res(30).obj; error]; 
    fprintf(' %.2f s (%.1f data/s)', batchTime, speed) ;
    fprintf(' sm1: %.5f', stats(2)/numDone) ; 
    fprintf(' sm2: %.5f', stats(3)/numDone) ; 
    fprintf(' sm3: %.5f', stats(4)/numDone) ; 
    fprintf(' re-1: %.5f', stats(5)/numDone) ;
    fprintf(' re-2: %.5f', stats(6)/numDone) ;
    fprintf(' re-3: %.5f', stats(7)/numDone) ;
    fprintf(' error: %.5f', stats(8)/numDone) ;
    fprintf(' [%d/%d]', numDone, batchSize_r);
    fprintf(' lr: %.6f',learningRate);
    fprintf('\n') ; 
    
end


% -------------------------------------------------------------------------
function [net,res] = accumulate_gradients(~, lr, batchSize, net, res, ~) % opti.
% -------------------------------------------------------------------------
for l = numel(net.layers):-1:1 % reverse order
  if isempty(res(l).dzdw)==0 % when the layer is defined on the SPD manifold, we must optimize W on Steifiel maniold
    if ~isfield(net.layers{l}, 'learningRate')
       net.layers{l}.learningRate = 1 ;
    end
    if ~isfield(net.layers{l}, 'weightDecay')
       net.layers{l}.weightDecay = 1;
    end
    thisLR = lr * net.layers{l}.learningRate ;

    if isfield(net.layers{l}, 'weight')
        if strcmp(net.layers{l}.type,'bfc')==1
            W1 = net.layers{l}.weight;
            W1grad  = (1/batchSize)*res(l).dzdw;
            % gradient update on Stiefel manifolds
            problemW1.M = stiefelfactory(size(W1,1), size(W1,2));
            W1Rgrad = (problemW1.M.egrad2rgrad(W1, W1grad)); % this sentence has a problem
            net.layers{l}.weight = (problemW1.M.retr(W1, -thisLR*W1Rgrad)); % retr indicates retraction (back to manifold)
        else
            net.layers{l}.weight = net.layers{l}.weight - thisLR * (1/batchSize) * res(l).dzdw ;% update W, here just for fc layer
        end
    
    end
  end
end

