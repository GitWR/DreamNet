function net = spdnet_init_afew_deep_v1(varargin)
% spdnet_init initializes a spdnet

rng('default');
rng(0) ;

opts.layernum = 22;

Winit = cell(opts.layernum,1);
%opts.datadim = [63,53,43,33,43,53,63]; % the dimensionality of each bimap layer, which also indicates the kernel size of each layer
opts.datadim = [63,53,43,33,43,33,43,33,43,33,43,33,43,33,43,33,43,33,43,33,43,33,43]; % the dimensionality of each bimap layer, which also indicates the kernel size of each layer
% 45,25,10

for iw = 1 : opts.layernum % designed to initialize each cov kernel 
    if iw < 4
       A = rand(opts.datadim(iw));
       [U1, ~, ~] = svd(A * A');
       Winit{iw} = U1(:,1:opts.datadim(iw+1)); % the initialized filters are all satisfy column orthogonality
    elseif iw==4 || iw == 6 || iw == 8 || iw == 10 || iw == 12 || iw == 14 || iw == 16 || iw == 18 || iw == 20 || iw == 22
       A = rand(opts.datadim(iw+1));
       [U1, ~, ~] = svd(A * A');
       temp = U1(:,1:opts.datadim(iw));
       Winit{iw} = temp';
    else
       A = rand(opts.datadim(iw));
       [U1, ~, ~] = svd(A * A');
       temp = U1(:,1:opts.datadim(iw+1)); % the initialized filters are all satisfy column orthogonality
       Winit{iw} = temp; 
    end
end

f=1/100 ;
classNum = 45; % categories
fdim = size(Winit{iw},1) * size(Winit{iw},1); % iw-3
theta = f * randn(fdim, classNum, 'single'); % here is 2500 \times 8, may be fc layer
Winit{end+1} = theta; % the fully connected layer

net.layers = {} ; % use to construct each layer of the proposed SPDNet
net.layers{end+1} = struct('type', 'bfc','weight', Winit{1}) ;
net.layers{end+1} = struct('type', 'rec') ; 
net.layers{end+1} = struct('type', 'bfc','weight', Winit{2}) ;
net.layers{end+1} = struct('type', 'rec') ; 
net.layers{end+1} = struct('type', 'bfc','weight', Winit{3}) ;
net.layers{end+1} = struct('type', 'marginloss') ;

net.layers{end+1} = struct('type', 'log') ; 
net.layers{end+1} = struct('type', 'fc','weight', Winit{end}) ;
net.layers{end+1} = struct('type', 'softmaxloss') ;

net.layers{end+1} = struct('type', 'bfc','weight', Winit{4}) ;
net.layers{end+1} = struct('type', 'reconstructionloss') ;
net.layers{end+1} = struct('type', 'rec') ; 
net.layers{end+1} = struct('type', 'bfc','weight', Winit{5}) ;

net.layers{end+1} = struct('type', 'add') ;  % this layer used to perform skip connection

net.layers{end+1} = struct('type', 'marginloss') ;

net.layers{end+1} = struct('type', 'log') ; 
net.layers{end+1} = struct('type', 'fc','weight', Winit{end}) ;  % assume that the initial weight values of fc layer are same 
net.layers{end+1} = struct('type', 'softmaxloss') ;

net.layers{end+1} = struct('type', 'bfc','weight', Winit{6}) ;
net.layers{end+1} = struct('type', 'reconstructionloss') ;
net.layers{end+1} = struct('type', 'rec') ; 
net.layers{end+1} = struct('type', 'bfc','weight', Winit{7}) ;

net.layers{end+1} = struct('type', 'add') ;  % this layer used to perform skip connection

net.layers{end+1} = struct('type', 'marginloss') ;

net.layers{end+1} = struct('type', 'log') ;
net.layers{end+1} = struct('type', 'fc','weight', Winit{end}) ;
net.layers{end+1} = struct('type', 'softmaxloss') ;

net.layers{end+1} = struct('type', 'bfc','weight', Winit{8}) ;
net.layers{end+1} = struct('type', 'reconstructionloss') ;
net.layers{end+1} = struct('type', 'rec') ; 
net.layers{end+1} = struct('type', 'bfc','weight', Winit{9}) ;

net.layers{end+1} = struct('type', 'add') ;  % this layer used to perform skip connection

net.layers{end+1} = struct('type', 'marginloss') ;

net.layers{end+1} = struct('type', 'log') ;
net.layers{end+1} = struct('type', 'fc','weight', Winit{end}) ;
net.layers{end+1} = struct('type', 'softmaxloss') ;

net.layers{end+1} = struct('type', 'bfc','weight', Winit{10}) ;
net.layers{end+1} = struct('type', 'reconstructionloss') ;
net.layers{end+1} = struct('type', 'rec') ; 
net.layers{end+1} = struct('type', 'bfc','weight', Winit{11}) ;

net.layers{end+1} = struct('type', 'add') ;  % this layer used to perform skip connection

net.layers{end+1} = struct('type', 'marginloss') ;

net.layers{end+1} = struct('type', 'log') ;
net.layers{end+1} = struct('type', 'fc','weight', Winit{end}) ;
net.layers{end+1} = struct('type', 'softmaxloss') ;

net.layers{end+1} = struct('type', 'bfc','weight', Winit{12}) ;
net.layers{end+1} = struct('type', 'reconstructionloss') ;
net.layers{end+1} = struct('type', 'rec') ; 
net.layers{end+1} = struct('type', 'bfc','weight', Winit{13}) ;

net.layers{end+1} = struct('type', 'add') ;  % this layer used to perform skip connection

net.layers{end+1} = struct('type', 'marginloss') ;

net.layers{end+1} = struct('type', 'log') ;
net.layers{end+1} = struct('type', 'fc','weight', Winit{end}) ;
net.layers{end+1} = struct('type', 'softmaxloss') ;

net.layers{end+1} = struct('type', 'bfc','weight', Winit{14}) ;
net.layers{end+1} = struct('type', 'reconstructionloss') ;
net.layers{end+1} = struct('type', 'rec') ; 
net.layers{end+1} = struct('type', 'bfc','weight', Winit{15}) ;

net.layers{end+1} = struct('type', 'add') ;  % this layer used to perform skip connection

net.layers{end+1} = struct('type', 'marginloss') ;

net.layers{end+1} = struct('type', 'log') ;
net.layers{end+1} = struct('type', 'fc','weight', Winit{end}) ;
net.layers{end+1} = struct('type', 'softmaxloss') ;

net.layers{end+1} = struct('type', 'bfc','weight', Winit{16}) ;
net.layers{end+1} = struct('type', 'reconstructionloss') ;
net.layers{end+1} = struct('type', 'rec') ; 
net.layers{end+1} = struct('type', 'bfc','weight', Winit{17}) ;

net.layers{end+1} = struct('type', 'add') ;  % this layer used to perform skip connection

net.layers{end+1} = struct('type', 'marginloss') ;

net.layers{end+1} = struct('type', 'log') ;
net.layers{end+1} = struct('type', 'fc','weight', Winit{end}) ;
net.layers{end+1} = struct('type', 'softmaxloss') ;

net.layers{end+1} = struct('type', 'bfc','weight', Winit{18}) ;
net.layers{end+1} = struct('type', 'reconstructionloss') ;
net.layers{end+1} = struct('type', 'rec') ; 
net.layers{end+1} = struct('type', 'bfc','weight', Winit{19}) ;

net.layers{end+1} = struct('type', 'add') ;  % this layer used to perform skip connection

net.layers{end+1} = struct('type', 'marginloss') ;

net.layers{end+1} = struct('type', 'log') ;
net.layers{end+1} = struct('type', 'fc','weight', Winit{end}) ;
net.layers{end+1} = struct('type', 'softmaxloss') ;

net.layers{end+1} = struct('type', 'bfc','weight', Winit{20}) ;
net.layers{end+1} = struct('type', 'reconstructionloss') ;
net.layers{end+1} = struct('type', 'rec') ; 
net.layers{end+1} = struct('type', 'bfc','weight', Winit{21}) ;

net.layers{end+1} = struct('type', 'add') ;  % this layer used to perform skip connection

net.layers{end+1} = struct('type', 'marginloss') ;

net.layers{end+1} = struct('type', 'log') ;
net.layers{end+1} = struct('type', 'fc','weight', Winit{end}) ;
net.layers{end+1} = struct('type', 'softmaxloss') ;

net.layers{end+1} = struct('type', 'bfc','weight', Winit{22}) ;
net.layers{end+1} = struct('type', 'reconstructionloss') ;






