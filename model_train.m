function model_train(varargin)

% Copyright (C) 2015 Tsung-Yu Lin, Aruni RoyChowdhury, Subhransu Maji.
% All rights reserved.
%
% This file is part of the BCNN and is made available under
% the terms of the BSD license (see the COPYING file).

[opts, imdb] = model_setup(varargin{:}) ;

% -------------------------------------------------------------------------
%                                          Train encoders and compute codes
% -------------------------------------------------------------------------
if opts.useGpu
    gpuDevice(opts.useGpu(1));
end

if ~exist(opts.resultPath)
  psi = {} ;
  for i = 1:numel(opts.encoders)
    if exist(opts.encoders{i}.codePath)
      load(opts.encoders{i}.codePath, 'code', 'area') ;
    else
      if exist(opts.encoders{i}.path)
        encoder = load(opts.encoders{i}.path) ;
        if isa(encoder.net, 'dagnn.DagNN'), encoder.net = dagnn.DagNN.loadobj(encoder.net); end
        if isfield(encoder, 'net')
            if opts.useGpu, device = 'gpu'; else device = 'cpu'; end
            encoder.net = net_move_to_device(encoder.net, device);
        end
      else
        opts.encoders{i}.opts = horzcat(opts.encoders{i}.opts);
        train = find(ismember(imdb.images.set, [1 2])) ;
        train = vl_colsubset(train, 1000, 'uniform') ;
        encoder = encoder_train_from_images(...
          imdb, imdb.images.id(train), ...
          opts.encoders{i}.opts{:}, ...
          'useGpu', opts.useGpu, ...
          'scale', opts.imgScale, ...
          'keepAspect', opts.keepAspect, ...
          'border', opts.border) ;
        encoder_save(encoder, opts.encoders{i}.path) ;
      end
      code = encoder_extract_for_images(encoder, imdb, imdb.images.id, 'dataAugmentation', opts.dataAugmentation, 'scale', opts.imgScale) ;
      % savefast(opts.encoders{i}.codePath, 'code') ;
    end
    psi{i} = code ;
    clear code ;
  end
  psi = cat(1, psi{:}) ;
end

% -------------------------------------------------------------------------
%                                                            Train and test
% -------------------------------------------------------------------------

if exist(opts.resultPath)
  info = load(opts.resultPath) ;
else
  info = traintest(opts, imdb, psi) ;
  save(opts.resultPath, '-struct', 'info') ;
  vl_printsize(1) ;
  [a,b,c] = fileparts(opts.resultPath) ;
  print('-dpdf', fullfile(a, [b '.pdf'])) ;
end

str = {} ;
str{end+1} = sprintf('data: %s', opts.expDir) ;
str{end+1} = sprintf(' setup: %10s', opts.suffix) ;
str{end+1} = sprintf(' mAP: %.1f', info.test.map*100) ;
if isfield(info.test, 'acc')
  str{end+1} = sprintf(' acc: %6.1f ', info.test.acc*100);
end
if isfield(info.test, 'im_acc')
  str{end+1} = sprintf(' acc wo normlization: %6.1f ', info.test.im_acc*100);
end
str{end+1} = sprintf('\n') ;
str = cat(2, str{:}) ;
fprintf('%s', str) ;

[a,b,c] = fileparts(opts.resultPath) ;
txtPath = fullfile(a, [b '.txt']) ;
f=fopen(txtPath, 'w') ;
fprintf(f, '%s', str) ;
fclose(f) ;



% -------------------------------------------------------------------------
function info = traintest(opts, imdb, psi)
% -------------------------------------------------------------------------

% Train using verification or not
verificationTask = isfield(imdb, 'pairs');

switch opts.dataAugmentation
    case 'none'
        ts =1 ;
    case 'f1'
        ts = 2;
    otherwise
        error('not supported data augmentation')
end

if verificationTask, 
    train = ismember(imdb.pairs.set, [1 2]) ;
    test = ismember(imdb.pairs.set, 3) ;
else % classification task
    multiLabel = (size(imdb.images.label,1) > 1) ; % e.g. PASCAL VOC cls
    train = ismember(imdb.images.set, [1 2]) ;
    train = repmat(train, ts, []);
    train = train(:)';
    test = ismember(imdb.images.set, 3) ;
    test = repmat(test, ts, []);
    test = test(:)';
    info.classes = find(imdb.meta.inUse) ;
    
    % Train classifiers
    C = 1 ;
    w = {} ;
    b = {} ;
    
    for c=1:numel(info.classes)
      fprintf('\n-------------------------------------- ');
      fprintf('OVA-classifier: class: %d\n', c) ;
      if ~multiLabel
        y = 2*(imdb.images.label == info.classes(c)) - 1 ;
      else
        y = imdb.images.label(c,:) ;
      end
      y_test = y(test(1:ts:end));
      y = repmat(y, ts, []);
      y = y(:)';
      np = sum(y(train) > 0) ;
      nn = sum(y(train) < 0) ;
      n = np + nn ;

      [w{c},b{c}] = vl_svmtrain(psi(:,train & y ~= 0), y(train & y ~= 0), 1/(n* C), ...
        'epsilon', 0.001, 'verbose', 'biasMultiplier', 1, ...
        'maxNumIterations', n * 200) ;

      pred = w{c}'*psi + b{c} ;

      % try cheap calibration
      mp = median(pred(train & y > 0)) ;
      mn = median(pred(train & y < 0)) ;
      b{c} = (b{c} - mn) / (mp - mn) ;
      w{c} = w{c} / (mp - mn) ;
      pred = w{c}'*psi + b{c} ;

      scores{c} = pred ;
     
      pred_test = reshape(pred(test), ts, []);
      pred_test = mean(pred_test, 1);

      [~,~,i]= vl_pr(y(train), pred(train)) ; ap(c) = i.ap ; ap11(c) = i.ap_interp_11 ;
      [~,~,i]= vl_pr(y_test, pred_test) ; tap(c) = i.ap ; tap11(c) = i.ap_interp_11 ;
      [~,~,i]= vl_pr(y(train), pred(train), 'normalizeprior', 0.01) ; nap(c) = i.ap ;
      [~,~,i]= vl_pr(y_test, pred_test, 'normalizeprior', 0.01) ; tnap(c) = i.ap ;
    end
    
    % Book keeping
    info.w = cat(2,w{:}) ;
    info.b = cat(2,b{:}) ;
    info.scores = cat(1, scores{:}) ;
    info.train.ap = ap ;
    info.train.ap11 = ap11 ;
    info.train.nap = nap ;
    info.train.map = mean(ap) ;
    info.train.map11 = mean(ap11) ;
    info.train.mnap = mean(nap) ;
    info.test.ap = tap ;
    info.test.ap11 = tap11 ;
    info.test.nap = tnap ;
    info.test.map = mean(tap) ;
    info.test.map11 = mean(tap11) ;
    info.test.mnap = mean(tnap) ;
    clear ap nap tap tnap scores ;
    fprintf('mAP train: %.1f, test: %.1f\n', ...
      mean(info.train.ap)*100, ...
      mean(info.test.ap)*100);

    % Compute predictions, confusion and accuracy
    [~,preds] = max(info.scores,[],1) ;
    info.testScores = reshape(info.scores(:,test), size(info.scores,1), ts, []);
    info.testScores = reshape(mean(info.testScores, 2), size(info.testScores,1), []);
    [~,pred_test] = max(info.testScores, [], 1);
    [~,gts] = ismember(imdb.images.label, info.classes) ;
    gts_test = gts(test(1:ts:end));
    gts = repmat(gts, ts, []);
    gts = gts(:)';

    [info.train.confusion, info.train.acc] = compute_confusion(numel(info.classes), gts(train), preds(train)) ;
    [info.test.confusion, info.test.acc] = compute_confusion(numel(info.classes), gts_test, pred_test) ;
    
    
    [~, info.train.im_acc] = compute_confusion(numel(info.classes), gts(train), preds(train), ones(size(gts(train))), true) ;
    [~, info.test.im_acc] = compute_confusion(numel(info.classes), gts_test, pred_test, ones(size(gts_test)), true) ;
%     [info.test.confusion, info.test.acc] = compute_confusion(numel(info.classes), gts(test), preds(test)) ;
end

% -------------------------------------------------------------------------
function code = encoder_extract_for_images(encoder, imdb, imageIds, varargin)
% -------------------------------------------------------------------------
opts.batchSize = 64 ;
opts.maxNumLocalDescriptorsReturned = 500 ;
opts.concatenateCode = true;
opts.dataAugmentation = 'none';
opts.scale = 1;
opts = vl_argparse(opts, varargin) ;

[~,imageSel] = ismember(imageIds, imdb.images.id) ;
imageIds = unique(imdb.images.id(imageSel)) ;
n = numel(imageIds) ;

% prepare batches
n = ceil(numel(imageIds)/opts.batchSize) ;
batches = mat2cell(1:numel(imageIds), 1, [opts.batchSize * ones(1, n-1), numel(imageIds) - opts.batchSize*(n-1)]) ;
batchResults = cell(1, numel(batches)) ;

% just use as many workers as are already available
% numWorkers = matlabpool('size') ;
%parfor (b = 1:numel(batches), numWorkers)
for b = numel(batches):-1:1
  batchResults{b} = get_batch_results(imdb, imageIds, batches{b}, ...
                        encoder, opts.maxNumLocalDescriptorsReturned, opts.dataAugmentation, opts.scale) ;
end


switch opts.dataAugmentation
    case 'none'
        ts = 1;
    case 'f1'
        ts = 2;
    otherwise
        error('not supported data augmentation')
end

code = cell(1, numel(imageIds)*ts) ;
for b = 1:numel(batches)
  m = numel(batches{b});
  for j = 1:m
      k = batches{b}(j) ;
      for aa=1:ts
        code{(k-1)*ts+aa} = batchResults{b}.code{(j-1)*ts+aa};
      end
  end
end
if opts.concatenateCode
   code = cat(2, code{:}) ;
end
% code is either:
% - a cell array, each cell containing an array of local features for a
%   segment
% - an array of FV descriptors, one per segment

% -------------------------------------------------------------------------
function result = get_batch_results(imdb, imageIds, batch, encoder, maxn, dataAugmentation, scale)
% -------------------------------------------------------------------------
m = numel(batch) ;
im = cell(1, m) ;
task = getCurrentTask() ;
if ~isempty(task), tid = task.ID ; else tid = 1 ; end

switch dataAugmentation
    case 'none'
        tfs = [0 ; 0 ; 0 ];
    case 'f1'
        tfs = [...
            0   0 ;
            0   0 ;
            0   1];
    otherwise
        error('not supported data augmentation')
end

ts = size(tfs,2);
im = cell(1, m*ts);
for i = 1:m
    fprintf('Task: %03d: encoder: extract features: image %d of %d\n', tid, batch(i), numel(imageIds)) ;
    for j=1:ts
        idx = (i-1)*ts+j;
        im{idx} = imread(fullfile(imdb.imageDir, imdb.images.name{imdb.images.id == imageIds(batch(i))}));
        if size(im{idx}, 3) == 1, im{idx} = repmat(im{idx}, [1 1 3]); end; %grayscale image
        
        tf = tfs(:,j) ;
        if tf(3), sx = fliplr(1:size(im{idx}, 2)) ;
            im{idx} = im{idx}(:,sx,:);
        end
    end
end

if ~isfield(encoder, 'numSpatialSubdivisions')
  encoder.numSpatialSubdivisions = 1 ;
end
switch encoder.type
    case 'rcnn'
        net = vl_simplenn_tidy(encoder.net);
        net.useGpu = encoder.net.useGpu;
        code_ = get_rcnn_features(encoder.net, ...
            im, ...
            'regionBorder', encoder.regionBorder) ;
    case 'dcnn'
        gmm = [] ;
        if isfield(encoder, 'covariances'), gmm = encoder ; end
        code_ = get_dcnn_features(encoder.net, ...
            im, ...
            'encoder', gmm, ...
            'numSpatialSubdivisions', encoder.numSpatialSubdivisions, ...
            'maxNumLocalDescriptorsReturned', maxn, 'scales', scale) ;
    case 'dsift'
        gmm = [] ;
        if isfield(encoder, 'covariances'), gmm = encoder ; end
        code_ = get_dcnn_features([], im, ...
            'useSIFT', true, ...
            'encoder', gmm, ...
            'numSpatialSubdivisions', encoder.numSpatialSubdivisions, ...
            'maxNumLocalDescriptorsReturned', maxn) ;
    case 'bcnn'
        code_ = get_bcnn_features(encoder.net, im, 'scales', scale);
    case {'impbcnn', 'o2dp', 'sketcho2dp', 'sketch_bcnn'}
        code_ = get_rcnn_features(encoder.net, im, ...
            'regionBorder', encoder.regionBorder, 'scales', scale) ;
end
result.code = code_ ;

% -------------------------------------------------------------------------
function encoder = encoder_train_from_images(imdb, imageIds, varargin)
% -------------------------------------------------------------------------
opts.type = 'rcnn' ;
opts.model = '' ;
opts.modela = '';
opts.modelb = '';
opts.layer = 0 ;
opts.layera = 0 ;
opts.layerb = 0 ;
opts.useGpu = false ;
opts.regionBorder = 0.05 ;
opts.numPcaDimensions = +inf ;
opts.numSamplesPerWord = 1000 ;
opts.whitening = false ;
opts.whiteningRegul = 0 ;
opts.renormalize = false ;
opts.numWords = 64 ;
opts.numSpatialSubdivisions = 1 ;
opts.normalization = 'sqrt_L2';
opts.scale = 1;
opts.method = 'schulz';
opts.bpMethod = 'svd';
opts.sigma = 1;
opts.pow = 0.5;
opts.border = [0, 0];
opts.keepAspect = [];
opts.maxIter = 5;
opts.iter = 10;
opts.reg = 0.5;
opts.p = 0.5;
opts.dout = 4096;
opts = vl_argparse(opts, varargin) ;

encoder.type = opts.type ;
encoder.regionBorder = opts.regionBorder ;
switch opts.type
  case {'dcnn', 'dsift'}
    encoder.numWords = opts.numWords ;
    encoder.renormalize = opts.renormalize ;
    encoder.numSpatialSubdivisions = opts.numSpatialSubdivisions ;
end

switch opts.type
    case {'rcnn', 'dcnn'}
        encoder.net = load(opts.model) ;
        if ~isempty(opts.layer)
            encoder.net.layers = encoder.net.layers(1:opts.layer) ;
        end
        if opts.useGpu
            encoder.net = vl_simplenn_tidy(encoder.net);
            encoder.net = vl_simplenn_move(encoder.net, 'gpu') ;
            encoder.net.useGpu = true ;
        else
            encoder.net = vl_simplenn_tidy(encoder.net);
            encoder.net = vl_simplenn_move(encoder.net, 'cpu') ;
            encoder.net.useGpu = false ;
        end
   case 'bcnn'
        encoder.normalization = opts.normalization;
        encoder.neta = load(opts.modela);
        if isfield(encoder.neta, 'net')
            encoder.neta = encoder.neta.net;
        end
        
        if ~isempty(opts.modelb)
            assert(~isempty(opts.layerb), 'layerb is not specified')
            encoder.netb = load(opts.modelb);
            if isfield(encoder.netb, 'net')
                encoder.netb = encoder.netb.net;
            end
            encoder.netb.layers = encoder.netb.layers(1:opts.layerb);
        end
        
        if ~isempty(opts.layera)
            encoder.layera = opts.layera;
            maxLayer = opts.layera;
            if ~isempty(opts.layerb) && isempty(opts.modelb)
                maxLayer = max(maxLayer, opts.layerb);
                encoder.layerb = opts.layerb;
            end
            encoder.neta.layers = encoder.neta.layers(1:maxLayer);
        end
        
        if opts.useGpu, device = 'gpu'; else device = 'cpu'; end
        
        encoder.neta = net_move_to_device(encoder.neta, device);
        if isfield(encoder, 'netb')
            encoder.netb = net_move_to_device(encoder.netb, device);
        end
        
        encoder.net = initializeNetFromEncoder(encoder);
        rmFields = {'neta', 'netb', 'layera', 'layerb'};
        rmIdx = find(ismember(rmFields, fieldnames(encoder)));
        for i=1:numel(rmIdx)
            encoder = rmfield(encoder, rmFields{rmIdx(i)});
        end
        if isa(encoder.net, 'dagnn.DagNN')
            encoder.net.mode = 'test';
        else
            encoder.net = vl_simplenn_tidy(encoder.net);
            if opts.useGpu
                encoder.net.useGpu = true;
            end
        end        
    case 'impbcnn'
        encoder.net = load(opts.model);
        encoder.sigma = opts.sigma;
        encoder.maxIter = opts.maxIter; 
        % resolve the inconsistent saving format of network
        if isfield(encoder.net, 'net')
            encoder.net = encoder.net.net;
        end
        
        % resolve the inconsistent saving format of meta
        if isa(encoder.net, 'dagnn.DagNN')
            if ~isprop(encoder.net, 'meta')
                encoder.net.meta.normalization = encoder.net.normalization;
            end
        else
            if ~isfield(encoder.net, 'meta')
                encoder.net.meta.normalization = encoder.net.normalization;
                encoder.net = rmfield(encoder.net, 'normalization');
            end
        end

        % resolve the inconsistent saving format of averageImage
        if size(encoder.net.meta.normalization.averageImage, 3) ~= 3
            encoder.net.meta.normalization.averageImage = reshape(...
                    encoder.net.meta.normalization.averageImage, [1, 1, 3]);
        end


        % create the Dag object if the network is in Dag format
        if isfield(encoder.net, 'params')
            encoder.net = dagnn.DagNN.loadobj(encoder.net);
            inputName = encoder.net.getInputs();
            if ~strcmp(inputName, 'input')
                encoder.net.renameVar(inputName, 'input');
            end
        end

        % check if network is a Dag
        isDag = isa(encoder.net, 'dagnn.DagNN');

        % covert the network to a Dag if it is not
        if ~isDag
            encoder.net.layers = encoder.net.layers(1:opts.layer);
            encoder.net = dagnn.DagNN.fromSimpleNN(encoder.net, 'canonicalNames', true);
        end

        % Truncate the neta at layera
        if ~isempty(opts.layer)
            maxLayer = opts.layer;

            % remove the layers not required for computing the output of
            % layera
            executeOrder = encoder.net.getLayerExecutionOrder();
            maxIndex = find(executeOrder == maxLayer);
            removeIdx = executeOrder(maxIndex+1:end);
            removeName = {encoder.net.layers(removeIdx).name};
            encoder.net.removeLayer(removeName);

            encoder.net = net_deploy(encoder.net);
            encoder.net.removeLayer('prob')
        end

        % move to the device
        if opts.useGpu, device = 'gpu'; else device = 'cpu'; end
        encoder.net = net_move_to_device(encoder.net, device);

        input = encoder.net.getOutputs{1};

        % initialize the network
        if isnan(encoder.net.getLayerIndex('bilr_1'))
            
            encoder.method = opts.method;
            encoder.bpMethod = opts.bpMethod;
            encoder.pow = opts.pow;
            myBlock = SqrtmPooling('normalizeGradients', false, ...
                'method', encoder.method, ...
                'bpMethod', encoder.bpMethod, ...
                'sigma', encoder.sigma, ...
                'pow', encoder.pow, ...
                'maxIter', encoder.maxIter);
            output = {'b_1', 'svd_u', 'svd_d'};
            
            % Add bilinearpool layer
            layerName = 'bilr_1';
            encoder.net.addLayer(layerName, myBlock, input, output);
            
            % power normalization layer
            layerName = sprintf('sqrt_1');
            input = output;
            output = 's_1';
            encoder.net.addLayer(layerName, SilenceWrapper('blockType', ...
                'PowerNorm', 'fanIn', 1, 'params', {'pow', 0.5}), input, output);
            
            
            % L2 normalization layer
            layerName = 'l2_1';
            input = output;
            bpoutput = 'l_1';
            encoder.net.addLayer(layerName, L2Norm(), {input}, bpoutput);
        
        end

        if isa(encoder.net, 'dagnn.DagNN')
            encoder.net.mode = 'test';
            if opts.useGpu
                encoder.net.move('gpu');
            end
        else
            encoder.net = vl_simplenn_tidy(encoder.net);
            if opts.useGpu
                encoder.net.useGpu = true;
            end
        end
    case 'o2dp'
        encoder.net = load(opts.model);
        encoder.iter = opts.iter;
        encoder.reg = opts.reg; 
        encoder.p = opts.p;
        encoder.sigma = opts.sigma;
        % resolve the inconsistent saving format of network
        if isfield(encoder.net, 'net')
            encoder.net = encoder.net.net;
        end
        
        % resolve the inconsistent saving format of meta
        if isa(encoder.net, 'dagnn.DagNN')
            if ~isprop(encoder.net, 'meta')
                encoder.net.meta.normalization = encoder.net.normalization;
            end
        else
            if ~isfield(encoder.net, 'meta')
                encoder.net.meta.normalization = encoder.net.normalization;
                encoder.net = rmfield(encoder.net, 'normalization');
            end
        end

        % resolve the inconsistent saving format of averageImage
        if size(encoder.net.meta.normalization.averageImage, 3) ~= 3
            encoder.net.meta.normalization.averageImage = reshape(...
                    encoder.net.meta.normalization.averageImage, [1, 1, 3]);
        end


        % create the Dag object if the network is in Dag format
        if isfield(encoder.net, 'params')
            encoder.net = dagnn.DagNN.loadobj(encoder.net);
            inputName = encoder.net.getInputs();
            if ~strcmp(inputName, 'input')
                encoder.net.renameVar(inputName, 'input');
            end
        end

        % check if network is a Dag
        isDag = isa(encoder.net, 'dagnn.DagNN');

        % covert the network to a Dag if it is not
        if ~isDag
            encoder.net.layers = encoder.net.layers(1:opts.layer);
            encoder.net = dagnn.DagNN.fromSimpleNN(encoder.net, 'canonicalNames', true);
        end

        % Truncate the neta at layera
        if ~isempty(opts.layer)
            maxLayer = opts.layer;

            % remove the layers not required for computing the output of
            % layera
            executeOrder = encoder.net.getLayerExecutionOrder();
            maxIndex = find(executeOrder == maxLayer);
            removeIdx = executeOrder(maxIndex+1:end);
            removeName = {encoder.net.layers(removeIdx).name};
            encoder.net.removeLayer(removeName);

            encoder.net = net_deploy(encoder.net);
            encoder.net.removeLayer('prob')
        end

        % move to the device
        if opts.useGpu, device = 'gpu'; else device = 'cpu'; end
        encoder.net = net_move_to_device(encoder.net, device);

        input = encoder.net.getOutputs{1};

        % initialize the network
        if isnan(encoder.net.getLayerIndex('bilr_1'))
            myBlock = GammaDemocraticO2dpPool('p', opts.p, ...
                'iter', opts.iter, 'reg', opts.reg); 

            output = {'b_1'};
            
            % Add bilinearpool layer
            layerName = 'bilr_1';
            encoder.net.addLayer(layerName, myBlock, input, output);
            
            % power normalization layer
            layerName = sprintf('sqrt_1');
            input = output;
            output = 's_1';
            encoder.net.addLayer(layerName, SilenceWrapper('blockType', ...
                'PowerNorm', 'fanIn', 1, 'params', {'pow', 0.5}), input, output);
            
            % L2 normalization layer
            layerName = 'l2_1';
            input = output;
            bpoutput = 'l_1';
            encoder.net.addLayer(layerName, L2Norm(), {input}, bpoutput);
        
        end

        if isa(encoder.net, 'dagnn.DagNN')
            encoder.net.mode = 'test';
            if opts.useGpu
                encoder.net.move('gpu');
            end
        else
            encoder.net = vl_simplenn_tidy(encoder.net);
            if opts.useGpu
                encoder.net.useGpu = true;
            end
        end

    case 'sketcho2dp'
        encoder.net = load(opts.model);
        encoder.dout = opts.dout;
        encoder.iter = opts.iter;
        encoder.reg = opts.reg; 
        encoder.p = opts.p;
        % resolve the inconsistent saving format of network
        if isfield(encoder.net, 'net')
            encoder.net = encoder.net.net;
        end
        
        % resolve the inconsistent saving format of meta
        if isa(encoder.net, 'dagnn.DagNN')
            if ~isprop(encoder.net, 'meta')
                encoder.net.meta.normalization = encoder.net.normalization;
            end
        else
            if ~isfield(encoder.net, 'meta')
                encoder.net.meta.normalization = encoder.net.normalization;
                encoder.net = rmfield(encoder.net, 'normalization');
            end
        end

        % resolve the inconsistent saving format of averageImage
        if size(encoder.net.meta.normalization.averageImage, 3) ~= 3
            encoder.net.meta.normalization.averageImage = reshape(...
                    encoder.net.meta.normalization.averageImage, [1, 1, 3]);
        end


        % create the Dag object if the network is in Dag format
        if isfield(encoder.net, 'params')
            encoder.net = dagnn.DagNN.loadobj(encoder.net);
            inputName = encoder.net.getInputs();
            if ~strcmp(inputName, 'input')
                encoder.net.renameVar(inputName, 'input');
            end
        end

        % check if network is a Dag
        isDag = isa(encoder.net, 'dagnn.DagNN');

        % covert the network to a Dag if it is not
        if ~isDag
            encoder.net.layers = encoder.net.layers(1:opts.layer);
            encoder.net = dagnn.DagNN.fromSimpleNN(encoder.net, 'canonicalNames', true);
        end

        % Truncate the neta at layera
        if ~isempty(opts.layer)
            maxLayer = opts.layer;

            % remove the layers not required for computing the output of
            % layera
            executeOrder = encoder.net.getLayerExecutionOrder();
            maxIndex = find(executeOrder == maxLayer);
            removeIdx = executeOrder(maxIndex+1:end);
            removeName = {encoder.net.layers(removeIdx).name};
            encoder.net.removeLayer(removeName);

            encoder.net = net_deploy(encoder.net);
            encoder.net.removeLayer('prob')
        end

        % move to the device
        if opts.useGpu, device = 'gpu'; else device = 'cpu'; end
        encoder.net = net_move_to_device(encoder.net, device);

        input = encoder.net.getOutputs{1};

        % initialize the network
        if isnan(encoder.net.getLayerIndex('bilr_1'))
            output = {'sk_1'};
            % a cheap way to determine if it is resnet or vgg
            if numel(encoder.net.layers) > 100
                prev_ch = [2048, 2048];
            else
                prev_ch = [512, 512];
            end
            myBlock = CompactBilinearTSLayer('outDim', encoder.dout, ...
                            'previousChannels', prev_ch, ...
                            'learnW', 0);
            params = myBlock.weights_;
            paramNames = {'cp_proj1', 'cp_proj2'};
            encoder.net.addLayer('sketch', myBlock, ...
                        input, output, paramNames);
            %{
            fidx = encoder.net.getParamIndex('sk_w1');
            factor = 1.0 / sqrt(encoder.dout);
            init_w = factor * (randi(2, 512, encoder.dout)*2-3);
            encoder.net.params(fidx).value = init_w; 
            encoder.net.params(fidx).learningRate = 0;
            
            fidx = encoder.net.getParamIndex('sk_w2');
            init_w = factor * (randi(2, 512, encoder.dout)*2-3);
            encoder.net.params(fidx).value = init_w;
            encoder.net.params(fidx).learningRate = 0;
            %}

            for f = 1:numel(paramNames)
                varId = encoder.net.getParamIndex(paramNames{f});
                encoder.net.params(varId).value = params{f};

                encoder. net.params(varId).learningRate = 1;
                encoder. net.params(varId).weightDecay = 1;
            end

            input = output; 
            output = {'b_1'};
            
            % Add bilinearpool layer
            layerName = 'bilr_1';
            encoder.net.addLayer(layerName, GammaDemocraticPool('p', opts.p, ...
                    'iter', opts.iter, 'reg', opts.reg), input, output);
            
            % power normalization layer
            layerName = sprintf('sqrt_1');
            input = output;
            output = 's_1';
            encoder.net.addLayer(layerName, SilenceWrapper('blockType', ...
                'PowerNorm', 'fanIn', 1, 'params', {'pow', 0.5}), input, output);
            
            % L2 normalization layer
            layerName = 'l2_1';
            input = output;
            bpoutput = 'l_1';
            encoder.net.addLayer(layerName, L2Norm(), {input}, bpoutput);
        
        end

        if isa(encoder.net, 'dagnn.DagNN')
            encoder.net.mode = 'test';
            if opts.useGpu
                encoder.net.move('gpu');
            end
        else
            encoder.net = vl_simplenn_tidy(encoder.net);
            if opts.useGpu
                encoder.net.useGpu = true;
            end
        end

    case 'sketcho2dp'
        encoder.net = load(opts.model);
        encoder.dout = opts.dout;
        encoder.iter = opts.iter;
        encoder.reg = opts.reg; 
        encoder.p = opts.p;
        % resolve the inconsistent saving format of network
        if isfield(encoder.net, 'net')
            encoder.net = encoder.net.net;
        end
        
        % resolve the inconsistent saving format of meta
        if isa(encoder.net, 'dagnn.DagNN')
            if ~isprop(encoder.net, 'meta')
                encoder.net.meta.normalization = encoder.net.normalization;
            end
        else
            if ~isfield(encoder.net, 'meta')
                encoder.net.meta.normalization = encoder.net.normalization;
                encoder.net = rmfield(encoder.net, 'normalization');
            end
        end

        % resolve the inconsistent saving format of averageImage
        if size(encoder.net.meta.normalization.averageImage, 3) ~= 3
            encoder.net.meta.normalization.averageImage = reshape(...
                    encoder.net.meta.normalization.averageImage, [1, 1, 3]);
        end


        % create the Dag object if the network is in Dag format
        if isfield(encoder.net, 'params')
            encoder.net = dagnn.DagNN.loadobj(encoder.net);
            inputName = encoder.net.getInputs();
            if ~strcmp(inputName, 'input')
                encoder.net.renameVar(inputName, 'input');
            end
        end

        % check if network is a Dag
        isDag = isa(encoder.net, 'dagnn.DagNN');

        % covert the network to a Dag if it is not
        if ~isDag
            encoder.net.layers = encoder.net.layers(1:opts.layer);
            encoder.net = dagnn.DagNN.fromSimpleNN(encoder.net, 'canonicalNames', true);
        end

        % Truncate the neta at layera
        if ~isempty(opts.layer)
            maxLayer = opts.layer;

            % remove the layers not required for computing the output of
            % layera
            executeOrder = encoder.net.getLayerExecutionOrder();
            maxIndex = find(executeOrder == maxLayer);
            removeIdx = executeOrder(maxIndex+1:end);
            removeName = {encoder.net.layers(removeIdx).name};
            encoder.net.removeLayer(removeName);

            encoder.net = net_deploy(encoder.net);
            encoder.net.removeLayer('prob')
        end

        % move to the device
        if opts.useGpu, device = 'gpu'; else device = 'cpu'; end
        encoder.net = net_move_to_device(encoder.net, device);

        input = encoder.net.getOutputs{1};

        % initialize the network
        if isnan(encoder.net.getLayerIndex('bilr_1'))
            output = {'b_1'};
            layerName = 'bilr_1';
            % a cheap way to determine if it is resnet or vgg
            if numel(encoder.net.layers) > 100
                prev_ch = [2048, 2048];
            else
                prev_ch = [512, 512];
            end
            myBlock = CompactBilinearTSLayer('outDim', encoder.dout, ...
                            'previousChannels', prev_ch, ...
                            'learnW', 0, 'dopool', true);
            params = myBlock.weights_;
            paramNames = {'cp_proj1', 'cp_proj2'};
            encoder.net.addLayer(layerName, myBlock, ...
                        input, output, paramNames);

            for f = 1:numel(paramNames)
                varId = encoder.net.getParamIndex(paramNames{f});
                encoder.net.params(varId).value = params{f};

                encoder. net.params(varId).learningRate = 1;
                encoder. net.params(varId).weightDecay = 1;
            end

            % power normalization layer
            layerName = sprintf('sqrt_1');
            input = output;
            output = 's_1';
            encoder.net.addLayer(layerName, SilenceWrapper('blockType', ...
                'PowerNorm', 'fanIn', 1, 'params', {'pow', 0.5}), input, output);
            
            % L2 normalization layer
            layerName = 'l2_1';
            input = output;
            bpoutput = 'l_1';
            encoder.net.addLayer(layerName, L2Norm(), {input}, bpoutput);
        
        end

        if isa(encoder.net, 'dagnn.DagNN')
            encoder.net.mode = 'test';
            if opts.useGpu
                encoder.net.move('gpu');
            end
        else
            encoder.net = vl_simplenn_tidy(encoder.net);
            if opts.useGpu
                encoder.net.useGpu = true;
            end
        end
end

encoder.net.meta.normalization.border = opts.border;
if ~isempty(opts.keepAspect)
    encoder.net.meta.normalization.keepAspect = opts.keepAspect;
end

switch opts.type
  case {'rcnn', 'bcnn', 'impbcnn', 'o2dp', 'sketcho2dp', 'sketch_bcnn'}
    return ;
end

% Step 0: sample descriptors
fprintf('%s: getting local descriptors to train GMM\n', mfilename) ;
code = encoder_extract_for_images(encoder, imdb, imageIds, 'concatenateCode', false, 'scale', opts.scale) ;
descrs = cell(1, numel(code)) ;
numImages = numel(code);
numDescrsPerImage = floor(encoder.numWords * opts.numSamplesPerWord / numImages);
for i=1:numel(code)
  descrs{i} = vl_colsubset(code{i}, numDescrsPerImage) ;
end
descrs = cat(2, descrs{:}) ;
fprintf('%s: obtained %d local descriptors to train GMM\n', ...
  mfilename, size(descrs,2)) ;


% Step 1 (optional): learn PCA projection
if opts.numPcaDimensions < inf || opts.whitening
  fprintf('%s: learning PCA rotation/projection\n', mfilename) ;
  encoder.projectionCenter = mean(descrs,2) ;
  x = bsxfun(@minus, descrs, encoder.projectionCenter) ;
  X = x*x' / size(x,2) ;
  [V,D] = eig(X) ;
  d = diag(D) ;
  [d,perm] = sort(d,'descend') ;
  d = d + opts.whiteningRegul * max(d) ;
  m = min(opts.numPcaDimensions, size(descrs,1)) ;
  V = V(:,perm) ;
  if opts.whitening
    encoder.projection = diag(1./sqrt(d(1:m))) * V(:,1:m)' ;
  else
    encoder.projection = V(:,1:m)' ;
  end
  clear X V D d ;
else
  encoder.projection = 1 ;
  encoder.projectionCenter = 0 ;
end
descrs = encoder.projection * bsxfun(@minus, descrs, encoder.projectionCenter) ;
if encoder.renormalize
  descrs = bsxfun(@times, descrs, 1./max(1e-12, sqrt(sum(descrs.^2)))) ;
end

% Step 2: train GMM
v = var(descrs')' ;
[encoder.means, encoder.covariances, encoder.priors] = ...
  vl_gmm(descrs, opts.numWords, 'verbose', ...
  'Initialization', 'kmeans', ...
  'CovarianceBound', double(max(v)*0.0001), ...
  'NumRepetitions', 1) ;
