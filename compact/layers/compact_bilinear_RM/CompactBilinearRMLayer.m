classdef CompactBilinearRMLayer < dagnn.Filter
  properties      
        learnW = true;
  end
    
  
  methods
    function outputs = forward(obj, inputs, params)
        
        layer.weights = params;
        pre.x = inputs{1};
        now.x = [];
        now = yang_compact_bilinear_RM_forward(layer, pre, now);
        outputs{1} = now.x;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
        now.dzdx = derOutputs{1};
        pre.x = inputs{1};
        layer.weights = params;
        layer.learnW = obj.learnW;
        pre = yang_compact_bilinear_RM_backward(layer, pre, now);
        
        derInputs{1} = pre.dzdx;
        
        derParams = pre.dzdw;      
    end
    
    function rfs = getReceptiveFields(obj)
      rfs(1,1).size = [NaN NaN] ;
      rfs(1,1).stride = [NaN NaN] ;
      rfs(1,1).offset = [NaN NaN] ;
      rfs(2,1) = rfs(1,1) ;
    end

    function obj = CompactBilinearRMLayer(varargin)
      obj.load(varargin) ;
    end    
    
  end
end
