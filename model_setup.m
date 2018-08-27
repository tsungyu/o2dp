function [opts, imdb] = model_setup(varargin)


% Copyright (C) 2015 Tsung-Yu Lin, Aruni RoyChowdhury, Subhransu Maji.
% All rights reserved.
%
% This file is part of the BCNN and is made available under
% the terms of the BSD license (see the COPYING file).

setup ;

opts.numFetchThreads = 12;
opts.seed = 1 ;
opts.batchSize = 128 ;
opts.numEpochs = 100;
opts.momentum = 0.9;
opts.learningRate = 0.001;
opts.numSubBatches = 1;
opts.keepAspect = [];
opts.useVal = false;
opts.fromScratch = false;
opts.useGpu = 1 ;
opts.border = [0, 0];
opts.regionBorder = 0.05 ;
opts.numDCNNWords = 64 ;
opts.numDSIFTWords = 256 ;
opts.numSamplesPerWord = 1000 ;
opts.printDatasetInfo = false ;
opts.excludeDifficult = true ;
opts.datasetSize = inf;
% opts.encoders = {struct('name', 'rcnn', 'opts', {})} ;
opts.encoders = {} ;
opts.dataset = 'cub' ;
opts.carsDir = 'data/cars';
opts.cubDir = 'data/cub';
opts.fmdDir = 'data/fmd';
opts.mit_indoor = 'data/mit_indoor';
opts.aircraftDir = 'data/fgvc-aircraft-2013b';
opts.ilsvrcDir = '/home/tsungyulin/dataset/ILSVRC2014/CLS-LOC/';
opts.ilsvrcDir_224 = '/home/tsungyulin/dataset/ILSVRC2014/CLS-LOC-224/';
opts.dtdDir = 'data/dtd';
opts.suffix = 'baseline' ;
opts.prefix = 'v1' ;
opts.model  = 'imagenet-vgg-m.mat';
opts.modela = 'imagenet-vgg-m.mat';
opts.modelb = [];
opts.layer  = 14;
opts.layera = [];
opts.layerb = [];
opts.imgScale = 1;
opts.bcnnLRinit = false;
opts.bcnnLayer = 14;
opts.rgbJitter = false;
opts.dataAugmentation = {'none', 'none', 'none'};
opts.cudnn = true;
opts.nonftbcnnDir = 'nonftbcnn';
opts.batchNormalization = false;
opts.cudnnWorkspaceLimit = 1024*1024*1204; 
opts.plotStatistics = true;
opts.classifier = 'svm';
opts.doValidation = false;
opts.memoryMapFile = fullfile(tempdir, 'matconvnet.bin') ;

[opts, varargin] = vl_argparse(opts,varargin) ;

opts.expDir = sprintf('data/%s/%s-seed-%02d', opts.prefix, opts.dataset, opts.seed) ;
opts.nonftbcnnDir = fullfile(opts.expDir, opts.nonftbcnnDir);
opts.imdbDir = fullfile(opts.expDir, 'imdb') ;
opts.resultPath = fullfile(opts.expDir, sprintf('result-%s.mat', opts.suffix)) ;

opts = vl_argparse(opts,varargin) ;

if nargout <= 1, return ; end

% % Setup GPU if needed
% if opts.useGpu
%   gpuDevice(opts.useGpu) ;
% end

% -------------------------------------------------------------------------
%                                                            Setup encoders
% -------------------------------------------------------------------------

vl_xmkdir(opts.expDir) ;
vl_xmkdir(opts.imdbDir) ;

models = {} ;
modelPath = {};
for i = 1:numel(opts.encoders)
  if isstruct(opts.encoders{i})
    name = opts.encoders{i}.name ;
    opts.encoders{i}.path = fullfile(opts.expDir, [name '-encoder.mat']) ;
    opts.encoders{i}.codePath = fullfile(opts.expDir, [name '-codes.mat']) ;
    [md, mdpath] = get_cnn_model_from_encoder_opts(opts.encoders{i});
    models = horzcat(models, md) ;
    modelPath = horzcat(modelPath, mdpath);
  else
    for j = 1:numel(opts.encoders{i})
      name = opts.encoders{i}{j}.name ;
      opts.encoders{i}{j}.path = fullfile(opts.expDir, [name '-encoder.mat']) ;
      opts.encoders{i}{j}.codePath = fullfile(opts.expDir, [name '-codes.mat']) ;
      [md, mdpath] = get_cnn_model_from_encoder_opts(opts.encoders{i}{j});      
      models = horzcat(models, md) ;
      modelPath = horzcat(modelPath, mdpath);
    end
  end
  save(fullfile(opts.expDir, [name, '-options.mat']), 'opts');
end

% -------------------------------------------------------------------------
%                                                       Download CNN models
% -------------------------------------------------------------------------

for i = 1:numel(models)
    if ~exist(modelPath{i})
        error(['cannot find model ', models{i}]) ;
    end
end

% -------------------------------------------------------------------------
%                                                              Load dataset
% -------------------------------------------------------------------------


imdbPath = fullfile(opts.imdbDir, sprintf('imdb-seed-%d.mat', opts.seed)) ;
if exist(imdbPath)
  imdb = load(imdbPath) ;
  if(opts.rgbJitter)
      opts.pca = imdb_compute_pca(imdb, opts.expDir);
  end
  return ;
end

switch opts.dataset
    case 'cubcrop'
        imdb = cub_get_database(opts.cubDir, true, false);
    case 'cub'
        imdb = cub_get_database(opts.cubDir, false, opts.useVal);
    case 'aircraft-variant'
        imdb = aircraft_get_database(opts.aircraftDir, 'variant');
    case 'cars'
        imdb = cars_get_database(opts.carsDir, false, opts.useVal);
    case 'imagenet'
        imdb = cnn_imagenet_setup_data('dataDir', opts.ilsvrcDir);
    case 'imagenet-224'
        imdb = cnn_imagenet_setup_data('dataDir', opts.ilsvrcDir_224);
    case 'dtd'
        imdb = dtd_get_database(opts.dtdDir, 'seed', opts.seed);
    case 'fmd'
        imdb = fmd_get_database(opts.fmdDir, 'seed', opts.seed);
    case 'mit_indoor'
        imdb = mit_indoor_get_database(opts.mit_indoor);
    otherwise
        error('Unknown dataset %s', opts.dataset) ;
end

save(imdbPath, '-struct', 'imdb') ;

if(opts.rgbJitter)
   opts.pca = imdb_compute_pca(imdb, opts.expDir);
end

if opts.printDatasetInfo
  print_dataset_info(imdb) ;
end

% -------------------------------------------------------------------------
function [model, modelPath] = get_cnn_model_from_encoder_opts(encoder)
% -------------------------------------------------------------------------
p = find(strcmp('model', encoder.opts)) ;
if ~isempty(p)
  [~,m,e] = fileparts(encoder.opts{p+1}) ;
  model = {[m e]} ;
  modelPath = encoder.opts{p+1};
else
  model = {} ;
  modelPath = {};
end

% bilinear cnn models
p = find(strcmp('modela', encoder.opts)) ;
if ~isempty(p)
  [~,m,e] = fileparts(encoder.opts{p+1}) ;
  model = horzcat(model,{[m e]}) ;
  modelPath = horzcat(modelPath, encoder.opts{p+1});
end
p = find(strcmp('modelb', encoder.opts)) ;
if ~isempty(p) && ~isempty(encoder.opts{p+1})
  [~,m,e] = fileparts(encoder.opts{p+1}) ;
  model = horzcat(model,{[m e]}) ;
  modelPath = horzcat(modelPath, encoder.opts{p+1});
end


