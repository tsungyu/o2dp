function code = get_rcnn_features(net, im, varargin)
% GET_RCNN_FEATURES
%    This function gets the fc7 features for an image region,
%    extracted from the provided mask.

opts.batchSize = 96 ;
opts.regionBorder = 0.05;
opts.scales = 1;
opts = vl_argparse(opts, varargin) ;

if ~iscell(im)
  im = {im} ;
end

res = [] ;
cache = struct() ;
resetCache() ;

    % for each image
    function resetCache()
        cache.images = cell(1,opts.batchSize) ;
        cache.indexes = zeros(1, opts.batchSize) ;
        cache.numCached = 0 ;
    end

    function flushCache()
        if cache.numCached == 0, return ; end
        images = cat(4, cache.images{:}) ;
        % images = bsxfun(@minus, images, net.meta.normalization.averageImage) ;
        images = bsxfun(@minus, images, averageColor) ;
        
        isDag = isa(net, 'dagnn.DagNN');
        if isDag
            if strcmp(net.device, 'gpu')
                images = gpuArray(images);
            end
            inputName = net.getInputs();
            outputNames = net.getOutputs();
            input = {inputName{1}, images};
            outIdx = net.getVarIndex(outputNames{1});

            net.eval(input);
            code_ = squeeze(gather(net.vars(outIdx).value));
        else
            if net.useGpu
                images = gpuArray(images) ;
            end
            res = vl_simplenn(net, images, ...
                            [], res, ...
                            'conserveMemory', true, ...
                            'sync', true) ;
            code_ = squeeze(gather(res(end).x)) ;
        end
        code_ = bsxfun(@times, 1./sqrt(sum(code_.^2)), code_) ;
        for q=1:cache.numCached
            code{cache.indexes(q)} = code_(:,q) ;
        end
        resetCache() ;
    end

    function appendCache(i,im)
        cache.numCached = cache.numCached + 1 ;
        cache.images{cache.numCached} = im ;
        cache.indexes(cache.numCached) = i;
        if cache.numCached >= opts.batchSize
            flushCache() ;
        end
    end

    if opts.scales == 1
        averageColor = net.meta.normalization.averageImage ;
    else
        averageColor = mean(mean(net.meta.normalization.averageImage, 1), 2);
    end
    code = {} ;
    for k=1:numel(im)
        appendCache(k, getImage(opts, single(im{k}), ...
                net.meta.normalization.imageSize, ...
                net.meta.normalization.keepAspect, opts.scales, ...
                net.meta.normalization.border));
    end
    flushCache() ;
end

% -------------------------------------------------------------------------
function reg = getImage(opts, im, regionSize, keepAspect, scale, border)
% -------------------------------------------------------------------------

w = size(im,2) ;
h = size(im,1) ;
factor = [(regionSize(1)+border(1))/h, (regionSize(2)+border(2))/w]*scale;

if keepAspect
    factor = max(factor);
end

reg = imresize(single(im), ...
    'scale', factor, ...
    'method', 'bilinear') ;

w = size(reg, 2) ;
h = size(reg, 1) ;

reg = imcrop(reg, [fix((w-regionSize(1)*scale)/2)+1, fix((h-regionSize(2)*scale)/2)+1,...
    round(regionSize(1)*scale)-1, round(regionSize(2)*scale)-1]);
end

