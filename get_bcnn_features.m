function code= get_bcnn_features(net, im, varargin)
% GET_BCNN_FEATURES  Get bilinear cnn features for an image
%   This function extracts the binlinear combination of CNN features
%   extracted from two different networks.

% Copyright (C) 2015 Tsung-Yu Lin, Aruni RoyChowdhury, Subhransu Maji.
% All rights reserved.
%
% This file is part of the BCNN and is made available under
% the terms of the BSD license (see the COPYING file).

nVargOut = max(nargout,1)-1;

if nVargOut==1 
    assert(true, 'Number of output should not be two.')
end

opts.crop = true ;
%opts.scales = 2.^(1.5:-.5:-3); % try a bunch of scales
opts.scales = 2;
opts.encoder = [] ;
opts.regionBorder = 0.05;
opts.border = [0, 0];
opts.normalization = 'sqrt_L2';
opts = vl_argparse(opts, varargin) ;

% % get parameters of the network
isDag = isa(net, 'dagnn.DagNN');
isTwoNet = false;
if isDag
    if isfield(net.meta, 'meta2')
        isTwoNet = true;
        averageColourB = mean(mean(net.meta.meta2.normalization.averageImage,1),2) ;
        imageSizeB = net.meta.meta2.normalization.imageSize;
    end
    if isTwoNet
        keepAspect = net.meta.meta1.normalization.keepAspect;
        averageColourA = mean(mean(net.meta.meta1.normalization.averageImage,1),2) ;
        imageSizeA = net.meta.meta1.normalization.imageSize;
    else
        keepAspect = net.meta.normalization.keepAspect;
        averageColourA = mean(mean(net.meta.normalization.averageImage,1),2) ;
        imageSizeA = net.meta.normalization.imageSize;
    end
else
    keepAspect = net.meta.normalization.keepAspect;
    averageColourA = mean(mean(net.meta.normalization.averageImage,1),2) ;
    imageSizeA = net.meta.normalization.imageSize;
end

% assert(all(imageSizeA == imageSizeB));

if ~iscell(im)
  im = {im} ;
end

code = cell(1, numel(im));

if nVargOut==2
    im_resA = cell(numel(im), 1);
    im_resB = cell(numel(im), 1);
end
% for each image
for k=1:numel(im)
    im_croppedA = imresize(single(im{k}), imageSizeA([2 1]), 'bilinear');
    crop_hA = size(im_croppedA,1) ;    crop_wA = size(im_croppedA,2) ;
    if isTwoNet
        im_croppedB = imresize(single(im{k}), imageSizeB([2 1]), 'bilinear');
        crop_hB = size(im_croppedB,1) ;    crop_wB = size(im_croppedB,2) ;
    end
    
    psi = cell(1, numel(opts.scales));
    
    % for each scale
    for s=1:numel(opts.scales)
        
        im_resizedA = preprocess_image(im{k}, keepAspect, imageSizeA, averageColourA, opts.scales(s), opts.border);
        if isTwoNet
            im_resizedB = preprocess_image(im{k}, keepAspect, imageSizeB, averageColourB, opts.scales(s), opts.border);
        end
        
        if isDag
            if strcmp(net.device, 'gpu')
                im_resizedA = gpuArray(im_resizedA);
            end
            inputNames = net.getInputs();
            inputs = {inputNames{1}, im_resizedA};
            if isTwoNet
                if strcmp(net.device, 'gpu')
                    im_resizedB = gpuArray(im_resizedB);
                end
                inputs{end+1} = 'netb_input';
                inputs{end+1} = im_resizedB;
            end
            net.eval(inputs);
            feat = net.vars(net.getVarIndex('l_1')).value;
        else
            if net.useGpu
                im_resizedA = gpuArray(im_resizedA);
            end
            res = [];
            res = vl_bilinearnn(net, im_resizedA, [], res, ...
                            'conserveMemory', true, 'sync', true);
            feat = res(end).x;
        end
        psi{s} = squeeze(gather(feat));
        feat_dim = max(cellfun(@length,psi));
    end
    code{k} = zeros(feat_dim, 1);
    % pool across scales
    for s=1:numel(opts.scales),
        if ~isempty(psi{s}),
            code{k} = code{k} + psi{s};
        end
    end
    assert(~isempty(code{k}));
end



function im_resized = preprocess_image(im, keepAspect, imageSize, averageColour, scale, border)

w = size(im,2) ;
h = size(im,1) ;
factor = [(imageSize(1)+border(1))/h, (imageSize(2)+border(2))/w]*scale;

if keepAspect
    factor = max(factor);
end

im_resized = imresize(single(im), ...
    'scale', factor, ...
    'method', 'bilinear') ;

w = size(im_resized,2) ;
h = size(im_resized,1) ;

im_resized = imcrop(im_resized, [fix((w-imageSize(1)*scale)/2)+1, ...
            fix((h-imageSize(2)*scale)/2)+1, ...
            round(imageSize(1)*scale)-1, round(imageSize(2)*scale)-1]);

im_resized = bsxfun(@minus, im_resized, averageColour) ;
