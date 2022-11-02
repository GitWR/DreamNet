function [net, info] = spdnet_train_afew(net, spd_train, opts)

opts.errorLabels = {'top1e'};
opts.train = find(spd_train.spd.set==1) ; % 1 represents the training samples
opts.val = find(spd_train.spd.set==2) ; % 2 indicates the testing samples
count = 0;

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
         if epoch > 1
             fprintf('resuming by loading epoch %d\n', epoch-1) ;
             load(modelPath(epoch-1), 'net', 'info') ;
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
         info.(f).objective(epoch) = (stats.(f)(2)+ stats.(f)(3)+ stats.(f)(4) + stats.(f)(5) + stats.(f)(6) + stats.(f)(7) + stats.(f)(8) + stats.(f)(9) ...
         + stats.(f)(10) + stats.(f)(11) + stats.(f)(12) + stats.(f)(13) + stats.(f)(14) + stats.(f)(15) + stats.(f)(16) + stats.(f)(17) ...
         + stats.(f)(18) + stats.(f)(19) + stats.(f)(20) + stats.(f)(21)) / n; 
         
         info.(f).acc(:,epoch) = stats.(f)(22:end) / n; % stats.(f)(3:end) / n;
     end
     if ~evaluateMode, save(modelPath(epoch), 'net', 'info') ; end
     
     %if epoch >= 1100
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
         plot(1:epoch,info.val.acc','.--','linewidth',2);
         leg = horzcat(leg,strcat('val')); % ,opts.errorLabels
         set(legend(leg{:}),'color','none');
         grid on;
         xlabel('training epoch');
         ylabel('error');
         title('error')
     end
     drawnow;
     print(1,modelFigPath,'-dpdf');
     %end
end  
    
    
function [net,stats] = process_epoch(opts, epoch, spd_train, trainInd, learningRate, net)

training = learningRate > 0 ;
count1 = 0;

if training
    mode = 'training' ; 
else
    mode = 'validation' ; 
end

% stats = [0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0 ; 0; 0; 0] ;
stats = [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0] ; % for softmax 

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

for ib = 1 : batchSize : length(trainInd) % select the training samples. 
    flag = flag + 1;
    fprintf('%s: epoch %02d: batch %3d/%3d:', mode, epoch, ib,length(trainInd)) ;
    batchTime = tic ; 
    res = [];
    if (ib+batchSize> length(trainInd))
        batchSize_r = length(trainInd)-ib+1;  
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
    net.layers{33}.class = spd_label; 
    net.layers{36}.class = spd_label; 
    net.layers{42}.class = spd_label; % (second softmax) 
    net.layers{45}.class = spd_label; 
    net.layers{51}.class = spd_label; 
    net.layers{54}.class = spd_label; 
    net.layers{60}.class = spd_label; 
    net.layers{63}.class = spd_label; 
    net.layers{69}.class = spd_label; 
    net.layers{72}.class = spd_label; 
    net.layers{78}.class = spd_label; 
    net.layers{81}.class = spd_label; 
    net.layers{87}.class = spd_label; 
    net.layers{90}.class = spd_label; 
    
    
    %forward/backward spdnet
    if training
        dzdy = one; 
    else
        dzdy = [] ;
    end
    
    [res, su] = vl_myforbackward(net, spd_data, dzdy, res, epoch, count1) ; % Currently, i have almostly know its for-and-back-ward process
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
          
    % accumulate training errors
    predictions = gather(res(end-3).x) ;
    [~,pre_label] = sort(predictions, 'descend') ;
    error1 = sum(~bsxfun(@eq, pre_label(1,:)', spd_label)) ;
    
    predictions = gather(res(end-12).x) ;
    [~,pre_label] = sort(predictions, 'descend') ;
    error2 = sum(~bsxfun(@eq, pre_label(1,:)', spd_label)) ;
    
    predictions = gather(res(end-21).x) ;
    [~,pre_label] = sort(predictions, 'descend') ;
    error3 = sum(~bsxfun(@eq, pre_label(1,:)', spd_label)) ;
    
    predictions = gather(res(end-30).x) ;
    [~,pre_label] = sort(predictions, 'descend') ;
    error4 = sum(~bsxfun(@eq, pre_label(1,:)', spd_label)) ;
    
    predictions = gather(res(end-39).x) ;
    [~,pre_label] = sort(predictions, 'descend') ;
    error5 = sum(~bsxfun(@eq, pre_label(1,:)', spd_label)) ;
    
    predictions = gather(res(end-48).x) ;
    [~,pre_label] = sort(predictions, 'descend') ;
    error6 = sum(~bsxfun(@eq, pre_label(1,:)', spd_label)) ;
    
    predictions = gather(res(end-57).x) ;
    [~,pre_label] = sort(predictions, 'descend') ;
    error7 = sum(~bsxfun(@eq, pre_label(1,:)', spd_label)) ;
    
    predictions = gather(res(end-66).x) ;
    [~,pre_label] = sort(predictions, 'descend') ;
    error8 = sum(~bsxfun(@eq, pre_label(1,:)', spd_label)) ;
    
    predictions = gather(res(end-75).x) ;
    [~,pre_label] = sort(predictions, 'descend') ;
    error9 = sum(~bsxfun(@eq, pre_label(1,:)', spd_label)) ;
    
    predictions = gather(res(end-84).x) ;
    [~,pre_label] = sort(predictions, 'descend') ;
    error10 = sum(~bsxfun(@eq, pre_label(1,:)', spd_label)) ;
    
    error = min([error1,error2,error3,error4,error5,error6,error7,error8,error9,error10]); % ,error3
    
    errors = errors + error;
    batchTime = toc(batchTime) ;
    speed = batchSize/batchTime ;
    numDone = numDone + batchSize_r ;
    
    stats = stats+[batchTime ; res(10).obj; res(19).obj; res(28).obj; res(37).obj; res(46).obj; res(55).obj; res(64).obj; res(73).obj; res(82).obj; res(91).obj; ...
        res(12).obj; res(21).obj; res(30).obj; res(39).obj; res(48).obj; res(57).obj; res(66).obj; res(75).obj; res(84).obj; res(93).obj; error];
    
    fprintf(' %.2f s (%.1f data/s)', batchTime, speed) ;
    fprintf(' sm1: %.5f', stats(2)/numDone) ; 
    fprintf(' sm2: %.5f', stats(3)/numDone) ; 
    fprintf(' sm3: %.5f', stats(4)/numDone) ; 
    fprintf(' sm4: %.5f', stats(5)/numDone) ; 
    fprintf(' sm5: %.5f', stats(6)/numDone) ; 
    fprintf(' sm6: %.5f', stats(7)/numDone) ; 
    fprintf(' sm7: %.5f', stats(8)/numDone) ; 
    fprintf(' sm8: %.5f', stats(9)/numDone) ; 
    fprintf(' sm9: %.5f', stats(10)/numDone) ;
    fprintf(' sm10: %.5f', stats(11)/numDone) ;
    
    fprintf(' re-1: %.5f', stats(12)/numDone) ;
    fprintf(' re-2: %.5f', stats(13)/numDone) ;
    fprintf(' re-3: %.5f', stats(14)/numDone) ;
    fprintf(' re-4: %.5f', stats(15)/numDone) ;
    fprintf(' re-5: %.5f', stats(16)/numDone) ;
    fprintf(' re-6: %.5f', stats(17)/numDone) ;
    fprintf(' re-7: %.5f', stats(18)/numDone) ;
    fprintf(' re-8: %.5f', stats(19)/numDone) ;
    fprintf(' re-9: %.5f', stats(20)/numDone) ;
    fprintf(' re-10: %.5f', stats(21)/numDone);
    
    fprintf(' error: %.5f', stats(22)/numDone) ;
    fprintf(' [%d/%d]', numDone, batchSize_r);
    fprintf(' lr: %.6f',learningRate);
    fprintf('\n') ; 
    
end


% -------------------------------------------------------------------------
function [net,res] = accumulate_gradients(~, lr, batchSize, net, res, ~) % opti.
% -------------------------------------------------------------------------
for l = numel(net.layers):-1:1 % reverse order
  if isempty(res(l).dzdw)==0 % when the layer is defined on the SPD manifold, we should optimize W on the Steifiel maniold
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

