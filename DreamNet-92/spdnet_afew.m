function [net, info] = spdnet_afew(varargin)
%set up the path

confPath; % upload the path of some toolkits to the workspace
%parameter setting
opts.dataDir = fullfile('./data/afew') ;
opts.imdbPathtrain = fullfile(opts.dataDir, 'SPD_info.mat'); % get the data
opts.batchSize = 30; % original is 30 
opts.test.batchSize = 1;
opts.numEpochs = 1200; % maximum number of epoches is 500, this can be adjusted
opts.gpus = [] ;
opts.learningRate = 0.01 * ones(1,opts.numEpochs); % original lr is 0.01 
opts.weightDecay = 0.0005; 
opts.continue = 1;
%spdnet initialization
net = spdnet_init_afew_deep_v1() ; % 
%loading metadata 
load(opts.imdbPathtrain) ;
%spdnet training
[net, info] = spdnet_train_afew(net, spd_train, opts);