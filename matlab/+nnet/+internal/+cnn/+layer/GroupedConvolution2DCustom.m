classdef GroupedConvolution2DCustom < nnet.internal.cnn.layer.Layer
    % GroupedConvolution2D   Implementation of the 2D convolution layer
    % with groups.
    
    %   Copyright 2018-2019 The MathWorks, Inc.
    
    properties
        % LearnableParameters   Learnable parameters for the layer
        % (Vector of nnet.internal.cnn.layer.learnable.PredictionLearnableParameter)
        LearnableParameters = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.empty();
        
        % Name (char array)   A name for the layer
        Name
        
        % Stride (vector of int) Stride for each dimension
        Stride
        
        % DilationFactor (vector of int) Dilation factor for each dimension
        DilationFactor
        
        % PaddingSize   The padding applied to the input along the edges
        %   The padding that is applied along the edges. This is a row
        %   vector [t b l r] where t is the padding to the top, b is the
        %   padding applied to the bottom, l is the padding applied to the
        %   left, and r is the padding applied to the right.
        PaddingSize
                        
        % NumGroups (int)  The number of groups. Public since it is
        % inferred when setting weights.
        NumGroups 

        ChannelMean
    end
    
    properties (Constant)
        % DefaultName   Default layer's name.
        DefaultName = 'grouped_conv'
    end
    
    properties (SetAccess = private)
        % InputNames   This layer has a single input
        InputNames = {'in'}
        
        % OutputNames   This layer has a single output
        OutputNames = {'out'}
        
        % HasSizeDetermined   True for layers with size determined.
        HasSizeDetermined
        
        % Hyper-parameters
        
        % FilterSize  (1x2 int vector)  Size of each filter expressed in
        % height x width
        FilterSize
        
        % NumChannelsPerGroup (int)   The number of channels per group that
        % the input to the layer will have. [] if it has to be inferred
        % later
        NumChannelsPerGroup
        
        % NumFiltersPerGroup (int)  The number of filters per group in the layer
        NumFiltersPerGroup
        
        % PaddingMode   The mode used to determine the padding
        %   The mode used to calculate the PaddingSize property. This can
        %   be:
        %       'manual'    - PaddingSize is specified manually.
        %       'same'      - PaddingSize will be calculated so that the
        %                     output size is the same size as the input
        %                     when the stride is 1. More generally, the
        %                     output size will be ceil(inputSize/stride),
        %                     where inputSize is the height and width of
        %                     the input.
        PaddingMode
    end 
    
    properties(Access = private)
        ExecutionStrategy
    end
    
    properties (Dependent)
        % Learnable Parameters (nnet.internal.cnn.layer.LearnableParameter)
        Weights
        Bias
    end
    
    properties (Dependent, SetAccess = private)
        % Effective filter size which takes into account dilation
        EffectiveFilterSize
    end
    
    properties (Constant, Access = private)
        % WeightsIndex  Index of the Weights into the LearnableParameter
        %               vector
        WeightsIndex = 1;
        
        % BiasIndex     Index of the Bias into the LearnableParameter
        %               vector
        BiasIndex = 2;
    end
    
    methods
        function this = GroupedConvolution2DCustom( ...
                name, filterSize, numChannelsPerGroup, numFiltersPerGroup, numGroups, ...
                stride, dilationFactor, paddingMode, paddingSize)
            % GroupedConvolution2D   Constructor for a GroupedConvolution2D layer
            %
            %   Create a 2D grouped convolutional layer with the
            %   following compulsory parameters:
            %
            %       name             - Name for the layer
            %       filterSize       - Size of the filters [height x width]
            %       numChannelsPerGroup - The number of channels per group that 
            %                          the input to the layer will have. 
            %                          [] if it has to be determined later.
            %       numGroups        - The number of groups
            %       numFiltersPerGroup  - The number of filters per group in 
            %                          the layer
            %       dilationFactor   - A vector specifying the dilation factor for
            %                          each dimension [height width]
            %       stride           - A vector specifying the stride for
            %                          each dimension [height width]
            %       paddingMode      - A string, 'manual' or 'same'.
            %       paddingSize      - A vector specifying the padding for
            %                          each dimension [top bottom left right]
            
            this.Name = name;
            
            % Set Hyper-parameters
            this.FilterSize = filterSize;
            this.NumChannelsPerGroup = numChannelsPerGroup;
            this.HasSizeDetermined = ~isempty( numChannelsPerGroup ) && ...
                ~iIsTheStringSame(paddingMode) && ~isempty( numGroups );
            this.NumFiltersPerGroup = numFiltersPerGroup;
            this.NumGroups = numGroups;
            this.Stride = stride;
            this.DilationFactor = dilationFactor;
            this.PaddingMode = paddingMode;
            this.PaddingSize = paddingSize;
            
            % Set weights and bias to be LearnableParameter
            this.Weights = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter();
            this.Bias = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter();
            % Set default initializers. The external layer constructor
            % overwrites these values, which are set only for internal code
            % that bypasses the casual API.
            this.Weights.Initializer = iInternalInitializer('narrow-normal');
            this.Bias.Initializer = iInternalInitializer('zeros');

            this = this.setHostStrategy();
        end
        
        function Z = predict( this, X )
            % Note that padding is stored as [top bottom left right] but
            % the function expects [top left bottom right].
            inputSize = [size(X,1) size(X,2)];
            paddingSize = iCalculatePaddingSizeFromInputSize( ...
                this.PaddingMode, this.PaddingSize, ...
                this.EffectiveFilterSize, this.Stride, inputSize );
            Z = this.ExecutionStrategy.forward(X, this.Weights.Value, ...
                this.Bias.Value, ...
                paddingSize(1), paddingSize(3), ...
                paddingSize(2), paddingSize(4), ...
                this.Stride(1), this.Stride(2), ...
                this.DilationFactor(1), this.DilationFactor(2), ...
                this.NumGroups);
        end
              
        function varargout = backward( this, X, ~, dZ, ~ )
            % Return values: [dX, dW]
            % Note that padding is stored as [top bottom left right] but
            % the function expects [top left bottom right].
            inputSize = [size(X,1) size(X,2)];
            paddingSize = iCalculatePaddingSizeFromInputSize( ...
                this.PaddingMode, this.PaddingSize, ...
                this.EffectiveFilterSize, this.Stride, inputSize );
            [varargout{1:nargout}] = this.ExecutionStrategy.backward( ...
                X, this.Weights.Value, dZ, ...
                paddingSize(1), paddingSize(3), ...
                paddingSize(2), paddingSize(4), ...
                this.Stride(1), this.Stride(2), ...
                this.DilationFactor(1), this.DilationFactor(2), ...
                this.NumGroups);
        end
        
        function this = inferSize(this, inputSize)
            % inferSize     Infer the number of channels, the number of
            % groups and padding if 'same', based on the input size
            totChannels = iChannelsFromInputSize(inputSize);
            if isempty(this.NumGroups) && isempty(this.NumChannelsPerGroup)
                % Channelwise convolution
                this.NumChannelsPerGroup = 1;
                this.NumGroups = totChannels;
            elseif isempty(this.NumChannelsPerGroup)
                numChannelsPerGroup = totChannels/this.NumGroups;
                if iIsInteger(numChannelsPerGroup)
                    this.NumChannelsPerGroup = numChannelsPerGroup;
                else
                    error(message('nnet_cnn:layer:GroupedConvolution2DLayer:IncompatibleNumGroups',...
                        totChannels, this.NumGroups))
                end
            end            
            % If channels per group and number of groups are also set, do
            % nothing. A size mismatch will be caught by isValidInputSize.
            
            if iIsTheStringSame(this.PaddingMode)
                this.PaddingSize = iCalculateSamePadding( ...
                    this.EffectiveFilterSize, this.Stride, inputSize(1:2));
                
                % If the padding is set to 'same', the size will always
                % need to be determined again because we will need to
                % recalculate the padding.
                this.HasSizeDetermined = false;
            else
                this.HasSizeDetermined = true;
            end
        end
        
        function tf = isValidInputSize(this, inputSize)
            % isValidInputSize   Check if the layer can accept an input of
            % a certain size. Assumes inferSize has been called.
            tf = numel(inputSize)==3 && ...
                this.isFilterSizeSmallerThanImage( inputSize ) && ...
                this.numFilterChannelsMatchesNumImageChannels( inputSize );
        end
        
        function outputSize = forwardPropagateSize(this, inputSize)
            % forwardPropagateSize    Output the size of the layer based on
            % the input size. Assumes inferSize has been called.
            assert(~isempty(this.NumGroups))
            
            paddingSize = iCalculatePaddingSizeFromInputSize( ...
                this.PaddingMode, this.PaddingSize, this.EffectiveFilterSize, ...
                this.Stride, inputSize(1:2));
            heightAndWidthPadding = iCalculateHeightAndWidthPadding(paddingSize);
            outputHeightAndWidth = floor((inputSize(1:2) + ...
                heightAndWidthPadding - this.EffectiveFilterSize)./this.Stride) + 1;

            outputSize = [outputHeightAndWidth this.NumFiltersPerGroup*this.NumGroups];
        end
        
        function this = initializeLearnableParameters(this, precision)
            % initializeLearnableParameters    Initialize learnable
            % parameters using their initializer

            % Assumes inferSize has been called
            assert(~isempty(this.NumGroups) && ~isempty(this.NumChannelsPerGroup)) 
            
            if isempty(this.Weights.Value)
                % Initialize only if it is empty
                weightsSize = [this.FilterSize, this.NumChannelsPerGroup, ...
                    this.NumFiltersPerGroup*this.NumGroups ];
                % Initialize using the user visible size and reshape
                userVisibleSize = [this.FilterSize, this.NumChannelsPerGroup, ...
                    this.NumFiltersPerGroup, this.NumGroups ];
                weights = this.Weights.Initializer.initialize(...
                    userVisibleSize, 'Weights');
                weights = reshape(weights, weightsSize);
                this.Weights.Value = precision.cast(weights);
            else
                % Cast to desired precision
                this.Weights.Value = precision.cast(this.Weights.Value);
            end
            
            if isempty(this.Bias.Value)
                % Initialize only if it is empty  
                biasSize = [1, 1, this.NumFiltersPerGroup*this.NumGroups];
                % Initialize using the user visible size and reshape
                userVisibleSize = [1, 1, this.NumFiltersPerGroup, this.NumGroups];
                bias = this.Bias.Initializer.initialize(...
                    userVisibleSize, 'Bias');
                this.Bias.Value = reshape(bias, biasSize);
                bias = reshape(bias, biasSize);
                this.Bias.Value = precision.cast(bias);
            else
                % Cast to desired precision
                this.Bias.Value = precision.cast(this.Bias.Value);
            end
        end
        
        function this = prepareForTraining(this)
            this.LearnableParameters = nnet.internal.cnn.layer.learnable.convert2training(this.LearnableParameters);
        end
        
        function this = prepareForPrediction(this)
            this.LearnableParameters = nnet.internal.cnn.layer.learnable.convert2prediction(this.LearnableParameters);
        end
        
        function this = setupForHostPrediction(this)
            this = this.setHostStrategy();
            this.LearnableParameters(1).UseGPU = false;
            this.LearnableParameters(2).UseGPU = false;
        end
        
        function this = setupForGPUPrediction(this)
            this = this.setGPUStrategy();
            this.LearnableParameters(1).UseGPU = true;
            this.LearnableParameters(2).UseGPU = true;
        end
        
        function this = setupForHostTraining(this)
            this = this.setHostStrategy();
        end
        
        function this = setupForGPUTraining(this)
            this = this.setGPUStrategy();
        end
        
        % Setter and getter for Weights and Bias
        % These make easier to address into the vector of LearnableParameters
        % giving a name to each index of the vector
        function weights = get.Weights(this)
            weights = this.LearnableParameters(this.WeightsIndex);
        end
        
        function this = set.Weights(this, weights)
            this.LearnableParameters(this.WeightsIndex) = weights;
        end
        
        function bias = get.Bias(this)
            bias = this.LearnableParameters(this.BiasIndex);
        end
        
        function this = set.Bias(this, bias)
            this.LearnableParameters(this.BiasIndex) = bias;
        end

        function channelmean = get.ChannelMean(this)
            channelmean = this.ChannelMean;
        end

        function this = set.ChannelMean(this, channelmean)
            this.ChannelMean = channelmean;
        end

        function dilatedFilterSize = get.EffectiveFilterSize(this)
            % Dilation is equivalent to adding extra zeros in between the
            % elements of the filter so that it leads to the following
            % effective filter size:
            % dilatedFilterSize = filterSize +
            % (filterSize - 1) * (dilationFactor - 1)
            % or, simplifying:
            dilatedFilterSize = (this.FilterSize - 1) .* this.DilationFactor + 1;
        end               
    end
    
    methods(Access = private)        
        function tf = isFilterSizeSmallerThanImage( this, inputSize )
            % The size of the image is given by the first two dimensions of the input size
            imageSize = inputSize(1:2);
            
            % Need to take padding as well as dilation factor into account when comparing
            % image size and filter size
            heightAndWidthPadding = iCalculateHeightAndWidthPadding(this.PaddingSize);
            tf = all( this.EffectiveFilterSize <= imageSize + heightAndWidthPadding );
        end
        
        function tf = numFilterChannelsMatchesNumImageChannels( this, inputSize )
            % Assumes inferSize has been called
            assert(~isempty(this.NumChannelsPerGroup) && ~isempty(this.NumGroups))
            numImageChannels = inputSize(3);
            totalNumChannels = this.NumGroups*this.NumChannelsPerGroup;
            tf = totalNumChannels == numImageChannels;
        end
        
        function this = setHostStrategy(this)
            % setHostStrategy   Since Mkldnn does not support groups, use 
            % only strided conv strategy.
            this.ExecutionStrategy = nnet.internal.cnn.layer.util.GroupedConvolution2DHostStridedConvStrategy();
        end
        
        function this = setGPUStrategy(this)
            this.ExecutionStrategy = nnet.internal.cnn.layer.util.GroupedConvolution2DGPUStrategy();
        end
    end
end

function tf = iIsInteger(x)
tf = ~mod(x,1);
end

function numChannels = iChannelsFromInputSize(inputSize)
% iChannelsFromInputSize   The number of channels is the third element in
% inputSize. If inputSize doesn't have three elements, then it is
% implicitly 1.
if numel(inputSize)<3
    numChannels = 1;
else
    numChannels = inputSize(3);
end
end

function tf = iIsTheStringSame(x)
tf = nnet.internal.cnn.layer.padding.isTheStringSame(x);
end

function paddingSize = iCalculatePaddingSizeFromInputSize( ...
    paddingMode, paddingSize, filterOrPoolSize, stride, spatialInputSize)
paddingSize = nnet.internal.cnn.layer.padding.calculatePaddingSizeFromInputSize( ...
    paddingMode, paddingSize, filterOrPoolSize, stride, spatialInputSize);
end

function heightAndWidthPadding = iCalculateHeightAndWidthPadding(paddingSize)
heightAndWidthPadding = nnet.internal.cnn.layer.padding.calculateHeightAndWidthPadding(paddingSize);
end

function paddingSize = iCalculateSamePadding(filterSize, stride, inputSize)
paddingSize = nnet.internal.cnn.layer.padding.calculateSamePadding(filterSize, stride, inputSize);
end

function initializer = iInternalInitializer(name)
initializer = nnet.internal.cnn.layer.learnable.initializer...
    .initializerFactory(name);
end

% LocalWords:  Learnable nnet cnn learnable convolutional Backpropagation
% LocalWords:  Mkldnn Rnd
