classdef Convolution2DHostStridedConvStrategy < nnet.internal.cnn.layer.util.ExecutionStrategy
    % Convolution2DHostStridedConvStrategy   Execution strategy for running the convolution on the host
    
    %   Copyright 2017-2018 The MathWorks, Inc.
    
    methods
        function [Z, memory] = forward(~, X, ...
                weights, bias, ...
                topPad, leftPad, ...
                bottomPad, rightPad, ...
                verticalStride, horizontalStride, ...
                verticalDilation, horizontalDilation)

            Z = nnet.internal.cnnhost.stridedConv( ...
                X, weights, ...
                topPad, leftPad, ...
                bottomPad, rightPad, ...
                verticalStride, horizontalStride, ...
                verticalDilation, horizontalDilation, bias);

            memory = [];
        end
        
        function [dX,dW] = backward( ~, ...
                X, weights, dZ, ...
                topPad, leftPad, ...
                bottomPad, rightPad, ...
                strideHeight, strideWidth, ...
                verticalDilation, horizontalDilation)
            
            needsWeightGradients = nargout > 1;
            dX = nnet.internal.cnnhost.convolveBackwardData2D( ...
                X, weights, dZ, ...
                topPad, leftPad, ...
                bottomPad, rightPad, ...
                strideHeight, strideWidth, ...
                verticalDilation, horizontalDilation);
            if needsWeightGradients
                dW{1} = nnet.internal.cnnhost.convolveBackwardFilter2D( ...
                    X, weights, dZ, ...
                    topPad, leftPad, ...
                    bottomPad, rightPad, ...
                    strideHeight, strideWidth, ...
                    verticalDilation, horizontalDilation);
                dW{2} = nnet.internal.cnnhost.convolveBackwardBias2D(dZ);
            end
        end
    end
end
