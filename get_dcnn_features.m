function code = get_dcnn_features(net, im, varargin)
% GET_DCNN_FEATURES  Get convolutional features for an image region
%   This function extracts the DCNN (CNN+FV) for an image.
%   These can be used as SIFT replacement in e.g. a Fisher Vector.
%

opts.useSIFT = false ;
opts.crop = true ;
%opts.scales = 2.^(1.5:-.5:-3); % as in CVPR14 submission
opts.scales = 2;

opts.encoder = [] ;
opts.numSpatialSubdivisions = 1 ;
opts.maxNumLocalDescriptorsReturned = +inf ;
opts = vl_argparse(opts, varargin) ;

% Find geometric parameters of the representation. x is set to the
% leftmost pixel of the receptive field of the lefmost feature at
% the last level of the network. This is computed by backtracking
% from the last layer. Then we obtain a map
%
%   x(u) = offset + stride * u
%
% from a feature index u to pixel coordinate v of the center of the
% receptive field.

if isempty(net)
    keepAspect = true;
else
    keepAspect = net.meta.normalization.keepAspect;
end

if opts.useSIFT
  binSize = 8;
  offset = 1 + 3/2 * binSize ;
  stride = 4;
  border = binSize*2 ;
  imageSize = [224 224];
else
  info = vl_simplenn_display(net) ;
  x=1 ;
  for l=numel(net.layers):-1:1
    x=(x-1)*info.stride(2,l)-info.pad(2,l)+1 ;
  end
  offset = round(x + info.receptiveFieldSize(end)/2 - 0.5);
  stride = prod(info.stride(1,:)) ;
  border = round(info.receptiveFieldSize(end)/2+1) ;
  averageColour = mean(mean(net.meta.normalization.averageImage,1),2) ;
  imageSize = net.meta.normalization.imageSize;
end

if ~iscell(im)
  im = {im} ;
end

numNull = 0 ;
numReg = 0 ;

% for each image
for k=1:numel(im)
  im_cropped = imresize(single(im{k}), imageSize([2 1]), 'bilinear');
  crop_h = size(im_cropped,1) ;
  crop_w = size(im_cropped,2) ;
  psi = cell(1, numel(opts.scales)) ;
  loc = cell(1, numel(opts.scales)) ;
  res = [] ;

  % for each scale
  for s=1:numel(opts.scales)
    if min(crop_h,crop_w) * opts.scales(s) < border, continue ; end
    if sqrt(crop_h*crop_w) * opts.scales(s) > 1024, continue ; end

    % resize the cropped image and extract features everywhere
    
    if keepAspect
        w = size(im{k},2) ;
        h = size(im{k},1) ;
        factor = [imageSize(1)/h,imageSize(2)/w];
        
        
        factor = max(factor)*opts.scales(s) ;
        %if any(abs(factor - 1) > 0.0001)
        
        im_resized = imresize(single(im{k}), ...
            'scale', factor, ...
            'method', 'bilinear') ;
        %end
        
        w = size(im_resized,2) ;
        h = size(im_resized,1) ;
        
        im_resized = imcrop(im_resized, [fix((w-imageSize(1)*opts.scales(s))/2)+1, fix((h-imageSize(2)*opts.scales(s))/2)+1,...
            round(imageSize(1)*opts.scales(s))-1, round(imageSize(2)*opts.scales(s))-1]);
    else
        im_resized = imresize(single(im{k}), round(imageSize([2 1])*opts.scales(s)), 'bilinear');
    end
    
%     im_resized = imresize(im_cropped, opts.scales(s)) ;
    if opts.useSIFT
      [frames,descrs] = vl_dsift(mean(im_resized,3), ...
        'size', binSize, ...
        'step', stride, ...
        'fast', 'floatdescriptors') ;
      ur = unique(frames(1,:)) ;
      vr = unique(frames(2,:)) ;
      [u,v] = meshgrid(ur,vr) ;
      %assert(isequal([u(:)';v(:)'], frames)) ;
    else
      im_resized = bsxfun(@minus, im_resized, averageColour) ;
      if net.useGpu
        im_resized = gpuArray(im_resized) ;
      end
      res = vl_simplenn(net, im_resized, [], res, ...
        'conserveMemory', true, 'sync', true) ;
      w = size(res(end).x,2) ;
      h = size(res(end).x,1) ;
      descrs = permute(gather(res(end).x), [3 1 2]) ;
      descrs = reshape(descrs, size(descrs,1), []) ;
      [u,v] = meshgrid(...
        offset + (0:w-1) * stride, ...
        offset + (0:h-1) * stride) ;
    end

    u_ = (u - 1) / opts.scales(s) + 1 ;
    v_ = (v - 1) / opts.scales(s) + 1 ;
    loc_ = [u_(:)';v_(:)'] ;

    psi{s} = descrs;
    loc{s} = loc_;
  end
  
  % Concatenate features from all scales
  for r = 1:numel(psi)
    code{k} = cat(2, psi{:}) ;
    codeLoc{k} = cat(2, zeros(2,0), loc{:}) ;
    numReg = numReg + 1 ;
    numNull = numNull + isempty(code{k}) ;
  end
end

if numNull > 0
  fprintf('%s: %d out of %d regions with null DCNN descriptor\n', ...
    mfilename, numNull, numReg) ;
end

% at this point code{i} contains all local featrues for image i
if isempty(opts.encoder)
  % no gmm: return the local descriptors, but not too many!
  rng(0) ;
  for k=1:numel(code)
      code{k} = vl_colsubset(code{k}, opts.maxNumLocalDescriptorsReturned) ;
  end
else
  % FV encoding
  for k=1:numel(code)
      descrs = opts.encoder.projection * bsxfun(@minus, code{k}, ...
        opts.encoder.projectionCenter) ;
    
      if opts.encoder.renormalize
        descrs = bsxfun(@times, descrs, 1./max(1e-12, sqrt(sum(descrs.^2)))) ;
      end
      tmp = {} ;
      break_u = get_intervals(codeLoc{k}(1,:), opts.numSpatialSubdivisions) ;
      break_v = get_intervals(codeLoc{k}(2,:), opts.numSpatialSubdivisions) ;
      for spu = 1:opts.numSpatialSubdivisions
        for spv = 1:opts.numSpatialSubdivisions
          sel = ...
            break_u(spu) <= codeLoc{k}(1,:) & codeLoc{k}(1,:) < break_u(spu+1) & ...
            break_v(spv) <= codeLoc{k}(2,:) & codeLoc{k}(2,:) < break_v(spv+1);
          tmp{end+1}= vl_fisher(descrs(:, sel), ...
            opts.encoder.means, ...
            opts.encoder.covariances, ...
            opts.encoder.priors, ...
            'Improved') ;
        end
      end
      % normalization keeps norm = 1
      code{k} = cat(1, tmp{:}) / opts.numSpatialSubdivisions ;
  end
end

function breaks = get_intervals(x,n)
if isempty(x)
  breaks = ones(1,n+1) ;
else
  x = sort(x(:)') ;
  breaks = x(round(linspace(1, numel(x), n+1))) ;
end
breaks(end) = +inf ;
