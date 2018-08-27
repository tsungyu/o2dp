function run_experiments_sketcho2dp_resnet(dataset, p, dout, gpuidx)

% Copyright (C) 2018 Tsung-Yu Lin and Subhransu Maji.
% All rights reserved.
%
% This file is part of the BCNN and is made available under
% the terms of the BSD license (see the COPYING file).

% input:
% dataset: the name of the dataset. See the code for the options.
% p: gamma value for the gamma-democratic pooling.
% dout: the dimension of the sketch space.
% gpuidx: the index of the gpu ob which you would like to run the
% experiment. Use an empty array [] for running on cpu.

  if nargin < 2
      p = 0.5;
  end

  if nargin < 3
      dout = 8192;
  end
  
  if nargin < 4
      gpuidx = [];
  end

  if ischar(p)
      p = str2num(p);
  end
  if ischar(dout)
      dout = str2num(dout);
  end
  if ischar(gpuidx)
      gpuidx = str2num(gpuidx);
  end


  model_path = 'data/models/imagenet-resnet-101-dag.mat';
  sketcho2dp.name = 'sketcho2dp' ;
  sketcho2dp.opts = {...
    'type', 'sketcho2dp', ...
    'model', model_path, ...
    'layer', 342, ...      % the layer from which the features are aggregated
    'p', p, ...            % the hyper-parameter gamma 
    'iter', 10, ...        % the number of iteration for sinkhorn
    'reg', 0.5, ...        % the dampened factor for sinkhorn
    'dout', dout, ...      % the dimension for the sketch space
    } ;

  switch dataset
      case {'dtd', 'fmd'}
          splits = 10;
      otherwise
          splits = 1;
  end

  setupNameList = {'sketcho2dp'};   % list of models to train and test
  encoderList = {{sketcho2dp}};
  datasetList = {{dataset, splits}};
  
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
            'prefix', ['sketcho2dp', num2str(p, '%.0e'), '_dout_', num2str(dout, '%d')], ...              % name of the output folder
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
