function run_experiments_o2dp(dataset, p, gpuidx)


% Copyright (C) 2018 Tsung-Yu Lin and Subhransu Maji.
% All rights reserved.
%
% This file is part of the BCNN and is made available under
% the terms of the BSD license (see the COPYING file).

% input:
% dataset: the name of the dataset. See the code for the options.
% p: gamma value for the gamma-democratic pooling
% gpuidx: the index of the gpu ob which you would like to run the
% experiment. Use an empty array [] for running on cpu.

  if nargin < 2
      p = 0.5; 
  end
  if nargin < 3
      gpuidx = [];
  end

  if ischar(p)
      p = str2num(p);
  end

  if ischar(gpuidx)
      gpuidx = str2num(gpuidx);
  end

  model_path = 'data/models/imagenet-vgg-verydeep-16.mat';

  o2dp.name = 'o2dp' ;
  o2dp.opts = {...
    'type', 'o2dp', ...
    'model', model_path, ...
    'layer', 30,...
    'p', p, ...
    'iter', 10, ...
    'reg', 0.5, ...
    } ;

  setupNameList = {'o2dp'};   % list of models to train and test
  encoderList = {{o2dp}};
  datasetList = {{dataset, 1}};
  
  scales = [2];

  for ii = 1 : numel(datasetList)
    dataset = datasetList{ii} ;
    if iscell(dataset)
      numSplits = dataset{2} ;
      dataset = dataset{1} ;
    else
      numSplits = 1 ;
    end
    switch dataset
        case {'cub', 'cars'}
            border = [0, 0];
            datasetName = dataset;
        case 'aircrafts'
            border = [32, 32];
            datasetName = 'aircraft-variant';
        case {'dtd', 'fmd', 'mit_indoor'}
            border = [0, 0];
            datasetName = dataset;
    end
    switch dataset
        case 'cub'
            keepAspect = true;
        case {'cars', 'aircrafts'}
            keepAspect = false;
        case {'dtd', 'fmd', 'mit_indoor'}
            keepAspect  = false;
    end
    for jj = 1 : numSplits
      for ee = 1: numel(encoderList)
        % train and test the model
        model_train(...
            'dataset', datasetName, ...
            'seed', jj, ...
            'encoders', encoderList{ee}, ...
            'prefix', ['o2dp', num2str(p, '%.0e')], ...              % name of the output folder
            'suffix', setupNameList{ee}, ...
            'printDatasetInfo', ee == 1, ...
            'useGpu', gpuidx, ...
            'imgScale', scales(ee), ...  
            'dataAugmentation', 'f1', ...
            'keepAspect', keepAspect, ...
            'border', border) ;   
      end
    end
  end
end
