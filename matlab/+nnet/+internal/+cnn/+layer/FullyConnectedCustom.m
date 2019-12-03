classdef FullyConnectedCustom < nnet.internal.cnn.layer.Layer
    % FullyConnected   Implementation of the fully connected layer
    
    %   Copyright 2015-2018 The MathWorks, Inc.
    
    properties
        % LearnableParameters   The learnable parameters for this layer
        %   This layer has two learnable parameters, which are the weights
        %   and the bias.
        LearnableParameters = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.empty()
        
        % Name (char array)   A name for the layer
        Name


        ChannelMean

    end
    
    properties (Constant)
        % DefaultName   Default layer's name.
        DefaultName = 'fc'
    end
    
    properties(Access = public)
        QuantizationMethod
    end
    
    properties(SetAccess = private)
        % InputNames   This layer has a single input
        InputNames = {'in'}
        
        % OutputNames   This layer has a single output
        OutputNames = {'out'}
        
        % InputSize   Input size of the layer
        %   The input size of the fully connected layer. Note that the
        %   internal layers deal with 3D observations, and so this input
        %   size will be 3D. This will be empty if the input size has not
        %   been set yet.
        InputSize
        
        % NumNeurons  (scalar integer)   Number of neurons for the layer
        NumNeurons
        
        % Execution strategy   Execution strategy of the layer
        %   The execution strategy determines where (host/GPU) and how
        %   forward and backward operations are performed.
        ExecutionStrategy
        
        % ObservationDimension for the input data
        ObservationDim
    end
    
    properties (Dependent)
        % Weights   The weights for the layer
        Weights
        
        % Bias   The bias vector for the layer
        Bias
    end
    
    properties (Dependent, SetAccess = private)
        % HasSizeDetermined   Specifies if all size parameters are set
        %   If the input size has not been determined, then this will be
        %   set to false, otherwise it will be true.
        HasSizeDetermined
    end
    
    properties (Constant, Access = private)
        % WeightsIndex   Index of the Weights in the LearnableParameter vector
        WeightsIndex = 1;
        
        % BiasIndex   Index of the Bias in the LearnableParameter vector
        BiasIndex = 2;
    end
    
    methods
        function this = FullyConnectedCustom(name, inputSize, numNeurons)
            this.Name = name;
            
            % Set hyperparameters
            this.NumNeurons = numNeurons;
            this.InputSize = inputSize;
            
            if ~isempty(this.InputSize)
                if isempty(this.ObservationDim)
                    this.ObservationDim = numel(this.InputSize)+1;
                end
            end
            
            
            % Set learnable parameters
            this.Weights = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter();
            this.Bias = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter();
            % Set default initializers. The external layer constructor
            % overwrites these values, which are set only for internal code
            % that bypasses the casual API.
            this.Weights.Initializer = iInternalInitializer('narrow-normal');
            this.Bias.Initializer = iInternalInitializer('zeros');
            
            % Set execution strategy
            this.ExecutionStrategy = this.getHostStrategy();
            
            % Initialize QuantizationMethod
            this.QuantizationMethod = nnet.internal.cnn.layer.NoQuantization; 

            % FullyConnected layer needs X but not Z for the backward pass
            this.NeedsXForBackward = true;
            this.NeedsZForBackward = false;
        end
        
        function Z = predict(this, X)
            [wieghts0, bias0] = this.QuantizationMethod.remapped(this.Weights.Value, this.Bias.Value);
            [h,w,p,~] = size(X);
            Z = this.ExecutionStrategy.forward(X - reshape(this.ChannelMean,[h,w,p]), wieghts0, bias0, this.ObservationDim);
        end
        
        function [Z, memory] = forward(this, X)
            [h,w,p,~] = size(X);
            Z = this.ExecutionStrategy.forward(X - reshape(this.ChannelMean,[h,w,p]), this.Weights.Value,this.Bias.Value, this.ObservationDim);
            memory = [];
        end
        
        function varargout = backward(this, X, ~, dZ, ~)
            [varargout{1:nargout}] = this.ExecutionStrategy.backward(X, this.Weights.Value, dZ, this.ObservationDim);
        end
        
        function this = inferSize(this, inputSize)
            if ~this.HasSizeDetermined
                this.InputSize = inputSize;
                this.ObservationDim = numel(inputSize) + 1;
                % Match weights and bias to layer size
                weights = this.matchWeightsToLayerSize(this.Weights);
                this.LearnableParameters(this.WeightsIndex) = weights;
                bias = this.matchBiasToLayerSize(this.Bias);
                this.LearnableParameters(this.BiasIndex) = bias;
                % Set execution strategy
                this.ExecutionStrategy = this.getHostStrategy();
                
            end
        end
        
        function outputSize = forwardPropagateSize(this, inputSize)
            if ~this.HasSizeDetermined
                error(message('nnet_cnn:internal:cnn:layer:FullyConnected:ForwardPropagateSizeWithNoInputSize'));
            else
                if numel(this.InputSize) ~= 1
                    % spatialDims does not include channel and ObservationDim
                    spatialDims = 1:this.ObservationDim-2;
                    filterSize = this.InputSize(spatialDims);
                    outputSpatialDims = floor(inputSize(spatialDims) - filterSize) + 1;
                    outputSize = [ outputSpatialDims this.NumNeurons ];
                else
                    outputSize = this.NumNeurons;
                end
            end
        end
        
        function tf = isValidInputSize(this, inputSize)
            % isValidInputSize   Check if the layer can accept an input of
            % a certain size
            tf = ( ~this.HasSizeDetermined &&...
                 (numel(inputSize) == 4 || numel(inputSize) == 3 || numel(inputSize) == 1) ||...
                 isequal(this.InputSize, inputSize) );
        end
        
        function outputSeqLen = forwardPropagateSequenceLength(this, inputSeqLen, ~)
            % forwardPropagateSequenceLength   The sequence length of the
            % output of the layer given an input sequence length
            
            if ~isscalar( this.InputSize )
                % For non-scalar input sizes, the layer does not
                % support time distribution
                assert( isnumeric(inputSeqLen{:}) && (inputSeqLen{:} == 1) );
            end
            % A fully connected layer with scalar input size is
            % time-distribtued, and can propagate an arbitrary sequence
            % length.
            outputSeqLen = inputSeqLen;
        end
        
        function this = initializeLearnableParameters(this, precision)
            % Initialize weights
            if isempty(this.Weights.Value)
                % Initialize only if it is empty                
                weightsSize = [this.InputSize this.NumNeurons];
                % Initialize using the user visible size and reshape
                userVisibleSize = [this.NumNeurons prod(this.InputSize)];
                weights = this.Weights.Initializer.initialize(...
                    userVisibleSize, 'Weights');
                weights = reshape(weights, weightsSize);
                this.Weights.Value = precision.cast(weights);
            else
                % Cast to desired precision
                this.Weights.Value = precision.cast(this.Weights.Value);
            end
            
            % Initialize bias
            if isempty(this.Bias.Value)
                % Initialize only if it is empty
                if isscalar(this.InputSize)
                    biasSize = [1 this.NumNeurons];
                else
                    biasSize = ones(1, this.ObservationDim-1);
                    biasSize(end) = this.NumNeurons;
                end
                userVisibleSize = [this.NumNeurons 1];
                % Initialize using the user visible size and reshape                
                bias = this.Bias.Initializer.initialize(...
                    userVisibleSize, 'Bias');
                bias = reshape(bias, biasSize);
                this.Bias.Value = precision.cast(bias);
            else
                % Cast to desired precision
                this.Bias.Value = precision.cast(this.Bias.Value);
            end
        end
        
        function this = prepareForTraining(this)
            % prepareForTraining   Prepare this layer for training
            %   Before this layer can be used for training, we need to
            %   convert the learnable parameters to use the class
            %   TrainingLearnableParameter.
            this.LearnableParameters = nnet.internal.cnn.layer.learnable.convert2training(this.LearnableParameters);
        end
        
        function this = prepareForPrediction(this)
            % prepareForPrediction   Prepare this layer for prediction
            %   Before this layer can be used for prediction, we need to
            %   convert the learnable parameters to use the class
            %   PredictionLearnableParameter.
            this.LearnableParameters = nnet.internal.cnn.layer.learnable.convert2prediction(this.LearnableParameters);
        end
        
        function this = setupForHostPrediction(this)
            this.ExecutionStrategy = this.getHostStrategy();
            this.LearnableParameters(1).UseGPU = false;
            this.LearnableParameters(2).UseGPU = false;
        end
        
        function this = setupForGPUPrediction(this)
            this.ExecutionStrategy = this.getGPUStrategy();
            this.LearnableParameters(1).UseGPU = true;
            this.LearnableParameters(2).UseGPU = true;
        end
        
        function this = setupForHostTraining(this)
            this.ExecutionStrategy = this.getHostStrategy();
        end
        
        function this = setupForGPUTraining(this)
            this.ExecutionStrategy = this.getGPUStrategy();
        end
        
        function weights = get.Weights(this)
            weights = this.LearnableParameters(this.WeightsIndex);
        end
        
        function this = set.Weights(this, weights)
            if this.HasSizeDetermined
                weights = this.matchWeightsToLayerSize(weights);
            end
            this.LearnableParameters(this.WeightsIndex) = weights;
        end
        
        function bias = get.Bias(this)
            bias = this.LearnableParameters(this.BiasIndex);
        end
        
        function this = set.Bias(this, bias)
            if this.HasSizeDetermined
                bias = this.matchBiasToLayerSize(bias);
            end
            this.LearnableParameters(this.BiasIndex) = bias;
        end
        
        function tf = get.HasSizeDetermined(this)
            tf = ~isempty( this.InputSize );
        end 
    end
    
    methods (Access = private)
        function executionStrategy = getHostStrategy(this)
            switch numel(this.InputSize)
                case 1
                    executionStrategy = nnet.internal.cnn.layer.util.FullyConnectedHostVectorStrategy();
                case 3
                    executionStrategy = nnet.internal.cnn.layer.util.FullyConnectedHostImageStrategy();
                case 4
                    executionStrategy = nnet.internal.cnn.layer.util.FullyConnectedHost3DImageStrategy();
                otherwise
                   executionStrategy = nnet.internal.cnn.layer.util.FullyConnectedHostImageStrategy(); 
            end
        end
        
        function executionStrategy = getGPUStrategy(this)
            switch numel(this.InputSize)
                case 1
                    executionStrategy = nnet.internal.cnn.layer.util.FullyConnectedGPUVectorStrategy();
                case 3
                    executionStrategy = nnet.internal.cnn.layer.util.FullyConnectedGPUImageStrategy();
                case 4
                    executionStrategy = nnet.internal.cnn.layer.util.FullyConnectedGPU3DImageStrategy();
                otherwise
                   executionStrategy = nnet.internal.cnn.layer.util.FullyConnectedGPUImageStrategy(); 
            end
        end
        
        function weightsParameter = matchWeightsToLayerSize(this, weightsParameter)
            % matchWeightsToLayerSize    Reshape weights from a matrix into
            % a 4-D array.
            weights = weightsParameter.Value;
            if (numel(this.InputSize) == 3 || numel(this.InputSize) == 4)
                requiredSize = [this.InputSize this.NumNeurons];
                if isequal( iGetNDSize(weights, numel(requiredSize)), requiredSize )
                    % Weights are the right size -- nothing to do
                elseif isempty( weights )
                    % Weights are empty -- nothing we can do
                elseif ismatrix( weights ) 
                    % Weights are 2D -- need to transpose and reshape to 4D
                    % Transpose is needed since the user would set
                    % it as [output input] instead of [input output].
                    weights = reshape( weights', requiredSize );
                elseif numel(this.InputSize) == 3 && ndims(weights)== 5 % 2D input, 3D weight
                    weights = reshape( weights, requiredSize );
                elseif numel(this.InputSize) == 4 && ndims(weights)== 4 % 3D input, 2D weight
                    weights = reshape( weights, requiredSize );
                else
                    % There are three possibilities and this is a fourth state
                    % therefore something has gone wrong
                    warning( message('nnet_cnn:internal:cnn:layer:FullyConnected:InvalidState') );
                end
            elseif isscalar(this.InputSize)
                requiredSize = [this.NumNeurons this.InputSize];
                currentSize = iGetNDSize(weights,this.ObservationDim);
                if isequal( currentSize, requiredSize )
                    % Weights are the right size -- nothing to do
                elseif isempty( weights )
                    % Weights are empty -- nothing we can do
                elseif isequal( currentSize, fliplr(requiredSize) )
                    % Weights need to be transposed
                    weights = weights';
                elseif isequal( [currentSize(end) prod(currentSize(1:(end-1)))] , requiredSize )
                    % currentSize(end) corresponds to the numNeurons in the
                    % FC layer and currentSize(1:(end-1)) corresponds to
                    % InputSize
                    weights = reshape( weights, [prod(currentSize(1:(end-1))) currentSize(end)] );
                    weights = weights';
                else
                    % Weights are incorrect size
                    warning( message('nnet_cnn:internal:cnn:layer:FullyConnected:InvalidState') );
                end
            end
            weightsParameter.Value = weights;
        end
        
        function biasParameter = matchBiasToLayerSize(this, biasParameter)
            % matchBiasToLayerSize   Reshape biases from a matrix into a
            % 3-D array.
            bias = biasParameter.Value;
            if (numel(this.InputSize) == 3 || numel(this.InputSize) == 4)
                requiredSize = ones(1,this.ObservationDim-1);
                requiredSize(end) = this.NumNeurons;
                if isequal( iGetNDSize( bias, numel(requiredSize) ), requiredSize )
                    % Biases are the right size -- nothing to do
                elseif isempty( bias )
                    % Biases are empty -- nothing we can do
                elseif ismatrix( bias) || (ndims(bias) == 3 && numel(requiredSize) == 4) || (ndims(bias) == 4 && numel(requiredSize) == 3)
                    % Biases are 2d -- need to reshaped to 3d or 4d
                    % Biases are 3d -- need to be reshaped to 4d
                    % Biases are 4d -- need to be reshaped to 3d
                    bias = reshape(bias, requiredSize);
                end
            elseif isscalar(this.InputSize)
                requiredSize = [this.NumNeurons 1];
                if isequal( size(bias), requiredSize )
                    % Biases are the right size -- nothing to do
                elseif isempty( bias )
                    % Biases are empty -- nothing we can do
                elseif isequal ( size(bias), fliplr(requiredSize) )
                    % Transpose the bias
                    bias = bias';
                else
%                     if isequal( size(bias), expWrongSize )
                    % Biases are 3d or 4d -- need to reshape to 2d
                    bias = reshape(bias, requiredSize);
                end
            end
            biasParameter.Value = bias;
        end
    end
end

function sz = iGetNDSize(X, N)
sz = ones(1,N);
sz(1:ndims(X)) = size(X);
end

function initializer = iInternalInitializer(name)
initializer = nnet.internal.cnn.layer.learnable.initializer...
    .initializerFactory(name);
end

% LocalWords:  Learnable learnable fc hyperparameters nnet cnn Rnd
