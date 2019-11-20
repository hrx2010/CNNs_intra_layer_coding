classdef GroupedConvolution2DLayerCustom < nnet.cnn.layer.Layer & nnet.internal.cnn.layer.Externalizable
    % GroupedConvolution2DLayer   2-D grouped convolution layer
    %
    %   To create a grouped convolution layer, use groupedConvolution2dLayer
    %
    %   GroupedConvolution2DLayer properties:
    %       Name                        - A name for the layer.
    %       FilterSize                  - The height and width of the
    %                                     filters.
    %       NumChannelsPerGroup         - The number of channels for each
    %                                     filter per group.
    %       NumFiltersPerGroup          - The number of filters per group.
    %       NumGroups                   - The number of groups
    %       Stride                      - The step size for traversing the
    %                                     input vertically and
    %                                     horizontally.
    %       DilationFactor              - The step size for sampling the
    %                                     input vertically and
    %                                     horizontally.
    %       PaddingMode                 - The mode used to determine the
    %                                     padding.
    %       PaddingSize                 - The padding applied to the input 
    %                                     along the edges.
    %       Weights                     - Weights of the layer.
    %       Bias                        - Biases of the layer.
    %       WeightsInitializer          - The function for initializing the 
    %                                     weights.    
    %       WeightLearnRateFactor       - A number that specifies
    %                                     multiplier for the learning rate
    %                                     of the weights.
    %       BiasLearnRateFactor         - A number that specifies a
    %                                     multiplier for the learning rate
    %                                     for the biases.
    %       WeightL2Factor              - A number that specifies a
    %                                     multiplier for the L2 weight
    %                                     regulariser for the weights.
    %       BiasInitializer             - The function for initializing the
    %                                     bias.    
    %       BiasL2Factor                - A number that specifies a
    %                                     multiplier for the L2 weight
    %                                     regulariser for the biases.
    %       NumInputs                   - The number of inputs for the 
    %                                     layer.
    %       InputNames                  - The names of the inputs of the 
    %                                     layer.
    %       NumOutputs                  - The number of outputs of the 
    %                                     layer.
    %       OutputNames                 - The names of the outputs of the 
    %                                     layer.
    %
    %   Example:
    %       Create a grouped convolution layer with 10 groups of 5 filters of size 10-by-10.
    %
    %       layer = convolution2dLayer(10, 5, 'NumGroups', 10);
    %
    %   See also groupedConvolution2dLayer.
    
    %   Copyright 2015-2018 The MathWorks, Inc.
    
    properties(Dependent)
        % Name   A name for the layer
        %   The name for the layer. If this is set to '', then a name will
        %   be automatically set at training time.
        Name         
    end
    
    properties(SetAccess = private, Dependent)
        % FilterSize   The height and width of the filters
        %   The height and width of the filters. This is a row vector [h w]
        %   where h is the filter height and w is the filter width.
        FilterSize
        
        % NumGroups   The number of groups. If 'auto', it will be determined 
        % automatically at training time,
        NumGroups 
        
        % NumChannelsPerGroup   The number of channels in the input per group
        %   The number of channels in the input per group. This cannot be
        %   set and is determined at training time.
        NumChannelsPerGroup
        
        % NumFiltersPerGroup   The number of filters per group
        %   The number of filters for this layer. This also determines how
        %   many maps there will be in the output.
        NumFiltersPerGroup               
        
        % PaddingMode   The mode used to determine the padding
        %   The mode used to calculate the PaddingSize property. This can
        %   be:
        %       'manual'    - PaddingSize is specified manually. 
        %       'same'      - PaddingSize is calculated so that the output
        %                     is the same size as the input when the stride
        %                     is 1. More generally, the output size will be
        %                     ceil(inputSize/stride), where inputSize is 
        %                     the height and width of the input.
        PaddingMode
    end
     
    properties(Dependent)
        % Stride   The vertical and horizontal stride
        %   The step size for traversing the input vertically and
        %   horizontally. This is a row vector [u v] where u is the
        %   vertical stride, and v is the horizontal stride.
        Stride
        
        % DilationFactor   The step size for sampling the input or 
        % equivalently the up-sampling factor of the filter.  
        % It corresponds to an effective filter size of 
        % filterSize + (filterSize-1) * (DilationFactor-1), but the size
        % of the weights does not depend on the dilation factor.
        % This can be a scalar, in which case the same value is used for 
        % both dimensions, or it can be a vector [d_h d_w] where d_h is 
        % the vertical dilation, and d_w is the horizontal dilation.
        DilationFactor
                
        % PaddingSize   The padding applied to the input along the edges
        %   The padding that is applied along the edges. This is a row 
        %   vector [t b l r] where t is the padding to the top, b is the 
        %   padding applied to the bottom, l is the padding applied to the 
        %   left, and r is the padding applied to the right.
        PaddingSize
        
        % Weights   The weights for the layer
        %   The filters for the convolutional layer. An array with size
        %   FilterSize-by-NumChannelsPerGroup-by-NumFiltersPerGroup-by-NumGroups.
        Weights
        
        % Bias   The bias vector for the layer
        %   The bias for the convolutional layer. The size will be
        %   1-by-1-by-NumFiltersPerGroups-by-NumGroups.
        Bias
        
        % ChannelMean   Channel means to be subtracted
        %   An array with size NumChannels-by-1
        ChannelMean

        % WeightsInitializer   The function to initialize the weights.
        WeightsInitializer        
        
        % WeightLearnRateFactor   The learning rate factor for the weights
        %   The learning rate factor for the weights. This factor is
        %   multiplied with the global learning rate to determine the
        %   learning rate for the weights in this layer. For example, if it
        %   is set to 2, then the learning rate for the weights in this
        %   layer will be twice the current global learning rate.
        WeightLearnRateFactor
        
        % WeightL2Factor   The L2 regularization factor for the weights
        %   The L2 regularization factor for the weights. This factor is
        %   multiplied with the global L2 regularization setting to
        %   determine the L2 regularization setting for the weights in this
        %   layer. For example, if it is set to 2, then the L2
        %   regularization for the weights in this layer will be twice the
        %   global L2 regularization setting.
        WeightL2Factor
        
        % BiasInitializer   The function to initialize the bias.
        BiasInitializer        
        
        % BiasLearnRateFactor   The learning rate factor for the biases
        %   The learning rate factor for the bias. This factor is
        %   multiplied with the global learning rate to determine the
        %   learning rate for the bias in this layer. For example, if it
        %   is set to 2, then the learning rate for the bias in this layer
        %   will be twice the current global learning rate.
        BiasLearnRateFactor
        
        % BiasL2Factor   The L2 regularization factor for the biases
        %   The L2 regularization factor for the biases. This factor is
        %   multiplied with the global L2 regularization setting to
        %   determine the L2 regularization setting for the biases in this
        %   layer. For example, if it is set to 2, then the L2
        %   regularization for the biases in this layer will be twice the
        %   global L2 regularization setting.
        BiasL2Factor
    end
    
    methods
        function this = GroupedConvolution2DLayerCustom(privateLayer)
            this.PrivateLayer = privateLayer;
        end
        
        function out = saveobj(this)
            privateLayer = this.PrivateLayer;
            out.Version = 1.0;
            out.Name = privateLayer.Name;
            out.FilterSize = privateLayer.FilterSize;
            out.NumGroups = privateLayer.NumGroups;            
            out.NumChannelsPerGroup = privateLayer.NumChannelsPerGroup;
            out.NumFiltersPerGroup = privateLayer.NumFiltersPerGroup;
            out.Stride = privateLayer.Stride;
            out.DilationFactor = privateLayer.DilationFactor;
            out.PaddingMode = privateLayer.PaddingMode;
            out.PaddingSize = privateLayer.PaddingSize;
            out.Weights = toStruct(privateLayer.Weights);
            out.Bias = toStruct(privateLayer.Bias);
            out.ChannelMean = privateLayer.ChannelMean;
        end

        function val = get.Name(this)
            val = this.PrivateLayer.Name;
        end
        
        function this = set.Name(this, val)
            iAssertValidLayerName(val);
            this.PrivateLayer.Name = char(val);
        end        
                        
        function this = set.NumGroups(this, val)
            val = gather(val);
            % Setting method private, used to update after setting weights.
            this.PrivateLayer.NumGroups = val;
        end        
        
        function this = set.Stride(this, value)
            value = gather(value);
            iAssertValidStride(value);
            % Convert to canonical form
            this.PrivateLayer.Stride = double(iMakeIntoRowVectorOfTwo(value));
        end
        
        function this = set.DilationFactor(this, value)
            value = gather(value);
            iAssertValidDilationFactor(value);
            % Convert to canonical form
            this.PrivateLayer.DilationFactor = double(iMakeIntoRowVectorOfTwo(value));
        end
        
        function this = set.PaddingSize(this, value)
            if isequal(this.PaddingMode,'same')
                error(message('nnet_cnn:layer:Layer:PaddingSizeCanOnlyBeSetInManualMode'))
            end
            value = gather(value);
            iAssertValidPaddingSize(value);
            % Convert to canonical form
            this.PrivateLayer.PaddingSize = double(iCalculatePaddingSize(value));
        end
        
        function val = get.Weights(this)
            val = this.PrivateLayer.Weights.HostValue;
            if ~isempty(val)
                % Reshape to external format
                val = reshape(val, [this.FilterSize this.NumChannelsPerGroup ...
                    this.NumFiltersPerGroup this.NumGroups]);
            end
        end
        
        function this = set.Weights(this, value)
            expectedNumChannelsPerGroup = iExpectedValue(this.NumChannelsPerGroup);
            expectedNumGroups = iExpectedValueNumGroups(this.NumGroups);
            expectedSize = [this.FilterSize expectedNumChannelsPerGroup ...
                    this.NumFiltersPerGroup expectedNumGroups];

            value = iGatherAndValidateParameter(value, expectedSize);
            
            if ~isempty(value)
                % Set NumGroups if channel-wise
                if isequal(this.NumGroups, 'channel-wise')
                    this.PrivateLayer.NumGroups = size(value,5);
                end
                
                % Calling inferSize sets NumChannelsPerGroup if auto
                totInputChannels = size(value,3) * this.NumGroups;
                this.PrivateLayer = this.PrivateLayer.inferSize( [NaN NaN totInputChannels] );
                
                % Reshape to internal format
                value = reshape(value, [this.FilterSize ...
                    this.NumChannelsPerGroup this.NumFiltersPerGroup*this.NumGroups]);
            end
            this.PrivateLayer.Weights.Value = value;
        end
        
        function val = get.Bias(this)
            val = this.PrivateLayer.Bias.HostValue;
            if ~isempty(val)
                % Reshape to external format
                val = reshape(val, [1 1 this.NumFiltersPerGroup this.NumGroups]);
            end
        end
        
        function this = set.Bias(this, value)
            expectedNumGroups = iExpectedValueNumGroups(this.NumGroups);
            expectedSize = [1 1 this.NumFiltersPerGroup expectedNumGroups];
            
            value = iGatherAndValidateParameter(value, expectedSize);            

            if ~isempty(value)
                % Set NumGroups if channel-wise
                if isequal(this.NumGroups, 'channel-wise')
                    this.PrivateLayer.NumGroups = size(value,4);
                end                
                % Reshape to internal format
                value = reshape(value, [1 1 this.NumFiltersPerGroup*this.NumGroups]);
            end
            this.PrivateLayer.Bias.Value = value;
        end
        
        function val = get.ChannelMean(this)
            val = this.PrivateLayer.ChannelMean;
        end
        
        function this = set.ChannelMean(this, value)
            expectedSize = [1 1 sum(this.NumChannelsPerGroup*this.NumGroups)];
            value = iGatherAndValidateParameter(value, expectedSize);            
            this.PrivateLayer.ChannelMean = value;
        end
        
        function val = get.FilterSize(this)
            val = this.PrivateLayer.FilterSize;
        end
        
        function val = get.NumChannelsPerGroup(this)
            val = this.PrivateLayer.NumChannelsPerGroup;
            if isempty(val)
                val = 'auto';
            end
        end
        
        function val = get.NumFiltersPerGroup(this)
            val = this.PrivateLayer.NumFiltersPerGroup;
        end
        
        function val = get.NumGroups(this)
            val = this.PrivateLayer.NumGroups;
            if isempty(val)
                val = 'channel-wise';
            end
        end
        
        function val = get.Stride(this)
            val = this.PrivateLayer.Stride;
        end

        function val = get.DilationFactor(this)
            val = this.PrivateLayer.DilationFactor;
        end
        
        function val = get.PaddingMode(this)
            val = this.PrivateLayer.PaddingMode;
        end
        
        function val = get.PaddingSize(this)
            val = this.PrivateLayer.PaddingSize;
        end
        
        function val = get.WeightsInitializer(this)
            if iIsCustomInitializer(this.PrivateLayer.Weights.Initializer)
                val = this.PrivateLayer.Weights.Initializer.Fcn;
            else
                val = this.PrivateLayer.Weights.Initializer.Name;
            end
        end
        
        function this = set.WeightsInitializer(this, value)
            value = iAssertValidWeightsInitializer(value);        
            % Create the initializer with in and out indices of the weights
            % Here they involve only the channels and filters per group.
            % size:
            % FilterSize(1)-by-FilterSize(2)-by-NumChannelsPerGroup-by-
            % NumFiltersPerGroup-by-NumGroups
            this.PrivateLayer.Weights.Initializer = ...
                iInitializerFactory(value, [1 2 3], [1 2 4]);
        end
        
        function val = get.BiasInitializer(this)
            if iIsCustomInitializer(this.PrivateLayer.Bias.Initializer)
                val = this.PrivateLayer.Bias.Initializer.Fcn;
            else
                val = this.PrivateLayer.Bias.Initializer.Name;
            end
        end
        
        function this = set.BiasInitializer(this, value)
            value = iAssertValidBiasInitializer(value);
            % Bias initializers do not require to pass in and out indices
            this.PrivateLayer.Bias.Initializer = iInitializerFactory(value);
        end
        
        function val = get.WeightLearnRateFactor(this)
            val = this.PrivateLayer.Weights.LearnRateFactor;
        end
        
        function this = set.WeightLearnRateFactor(this, value)
            value = gather(value);
            iAssertValidFactor(value,'WeightLearnRateFactor');
            this.PrivateLayer.Weights.LearnRateFactor = value;
        end
        
        function val = get.BiasLearnRateFactor(this)
            val = this.PrivateLayer.Bias.LearnRateFactor;
        end
        
        function this = set.BiasLearnRateFactor(this, value)
            value = gather(value);
            iAssertValidFactor(value,'BiasLearnRateFactor');
            this.PrivateLayer.Bias.LearnRateFactor = value;
        end
        
        function val = get.WeightL2Factor(this)
            val = this.PrivateLayer.Weights.L2Factor;
        end
        
        function this = set.WeightL2Factor(this, value)
            value = gather(value);
            iAssertValidFactor(value,'WeightL2Factor');
            this.PrivateLayer.Weights.L2Factor = value;
        end
        
        function val = get.BiasL2Factor(this)
            val = this.PrivateLayer.Bias.L2Factor;
        end
        
        function this = set.BiasL2Factor(this, value)
            value = gather(value);
            iAssertValidFactor(value,'BiasL2Factor');
            this.PrivateLayer.Bias.L2Factor = value;
        end
    end
    
    methods(Static)
        function this = loadobj(in)
            this = iLoadGroupedConvolution2DLayerCustom(in);
        end
    end
    
    methods(Hidden, Access = protected)
        function [description, type] = getOneLineDisplay(this)
            numFiltersString = int2str( sum(this.NumFiltersPerGroup) );
            filterSizeString = i2DSizeToString( this.FilterSize );
            if ~isequal(this.NumChannelsPerGroup, 'auto')
                numChannelsString = ['x' int2str( this.NumChannelsPerGroup )];
            else
                numChannelsString = '';
            end
            if isequal(this.NumGroups, 1)
                numGroupsString = "";
            elseif isequal(this.NumGroups, 'channel-wise')
                numGroupsString = string(this.NumGroups)+" ";
                % Do not show 'x1' channels since implicit
                numChannelsString = '';                
            else
                numGroupsString = string(this.NumGroups)+" groups of ";
            end
            strideString = "["+int2str( this.Stride )+"]";
            if ~isequal(this.DilationFactor, [1 1])
                dilationFactorString = ", dilation factor ["+int2str( this.DilationFactor )+"],";
            else
                dilationFactorString = "";
            end
            
            if this.PaddingMode ~= "manual"
                paddingSizeString = "'"+this.PaddingMode+"'";
            else
                paddingSizeString = "["+int2str( this.PaddingSize )+"]";
            end
            
            description = iGetMessageString( ...
                'nnet_cnn:layer:GroupedConvolution2DLayer:oneLineDisplay', ...
                numGroupsString, ...
                numFiltersString, ...
                filterSizeString, ...
                numChannelsString, ...
                strideString, ...
                dilationFactorString, ...
                paddingSizeString );

            type = iGetMessageString( 'nnet_cnn:layer:GroupedConvolution2DLayer:Type' );
        end
        
        function groups = getPropertyGroups( this )
            hyperparameters = {
                'FilterSize'
                'NumGroups'                
                'NumChannelsPerGroup'
                'NumFiltersPerGroup'
                'Stride'
                'DilationFactor'
                'PaddingMode'
                'PaddingSize'
                };
            learnableParameters = {'Weights', 'Bias'};
            
            groups = [
                this.propertyGroupGeneral( {'Name'} )
                this.propertyGroupHyperparameters( hyperparameters )
                this.propertyGroupLearnableParameters( learnableParameters )
                ];
        end
        
        function footer = getFooter( this )
            variableName = inputname(1);
            footer = this.createShowAllPropertiesFooter( variableName );
        end
    end
end

function messageString = iGetMessageString( varargin )
messageString = getString( message( varargin{:} ) );
end

function sizeString = i2DSizeToString( sizeVector )
% i2DSizeToString   Convert a 2-D size stored in a vector of 2 elements
% into a string separated by 'x'.
sizeString = [ ...
    int2str( sizeVector(1) ) ...
    'x' ...
    int2str( sizeVector(2) ) ];
end

function iAssertValidFactor(value,factorName)
iEvalAndThrow(@()...
    nnet.internal.cnn.layer.paramvalidation.validateLearnFactor(value,factorName));
end

function obj = iLoadGroupedConvolution2DLayerCustom(in)
internalLayer = nnet.internal.cnn.layer.GroupedConvolution2DCustom( ...
    in.Name, in.FilterSize, in.NumChannelsPerGroup, in.NumFiltersPerGroup, ...
    in.NumGroups, in.Stride, in.DilationFactor, in.PaddingMode, in.PaddingSize);
internalLayer.Weights = nnet.internal.cnn.layer.learnable...
    .PredictionLearnableParameter.fromStruct(in.Weights);
internalLayer.Bias = nnet.internal.cnn.layer.learnable...
    .PredictionLearnableParameter.fromStruct(in.Bias);
internalLayer.ChannelMean = in.ChannelMean;

obj = nnet.cnn.layer.GroupedConvolution2DLayerCustom(internalLayer);
end

function expectedValue = iExpectedValue(value)
expectedValue = value;
if isequal(expectedValue, 'auto')
    expectedValue = NaN;
end
end

function expectedValue = iExpectedValueNumGroups(value)
expectedValue = value;
if isequal(expectedValue, 'channel-wise')
    expectedValue = NaN;
end
end

function iAssertValidLayerName(name)
iEvalAndThrow(@()...
    nnet.internal.cnn.layer.paramvalidation.validateLayerName(name));
end

function iAssertValidStride(value)
iEvalAndThrow(@()...
    nnet.internal.cnn.layer.paramvalidation.validateSizeParameter(value, 'Stride'));
end

function iAssertValidDilationFactor(value)
iEvalAndThrow(@()...
    nnet.internal.cnn.layer.paramvalidation.validateSizeParameter(value, 'DilationFactor'));
end

function iAssertValidPaddingSize(value)
iEvalAndThrow(@()...
    nnet.internal.cnn.layer.paramvalidation.validatePaddingSize(value));
end

function rowVectorOfTwo = iMakeIntoRowVectorOfTwo(scalarOrRowVectorOfTwo)
rowVectorOfTwo = ...
    nnet.internal.cnn.layer.paramvalidation.makeIntoRowVectorOfTwo(scalarOrRowVectorOfTwo);
end

function paddingSize = iCalculatePaddingSize(padding)
paddingSize = nnet.internal.cnn.layer.padding.calculatePaddingSize(padding);
end

function value = iAssertValidWeightsInitializer(value)
validateattributes(value, {'function_handle','char','string'}, {});
if(ischar(value) || isstring(value))
    value = validatestring(value, {'narrow-normal', ...
                           'glorot', ...
                           'he', ...
                           'zeros', ...
                           'ones'});
end
end

function value = iAssertValidBiasInitializer(value)
validateattributes(value, {'function_handle','char','string'}, {});
if(ischar(value) || isstring(value))
    value = validatestring(value, {'narrow-normal', ...
                           'zeros', ...
                           'ones'});
end
end

function initializer = iInitializerFactory(varargin)
initializer = nnet.internal.cnn.layer.learnable.initializer...
    .initializerFactory(varargin{:});
end

function tf = iIsCustomInitializer(init)
tf = isa(init, 'nnet.internal.cnn.layer.learnable.initializer.Custom');
end

function iEvalAndThrow(func)
% Omit the stack containing internal functions by throwing as caller
try
    func();
catch exception
    throwAsCaller(exception)
end
end

function value = iGatherAndValidateParameter(value, expectedSize)
try
    value = nnet.internal.cnn.layer.paramvalidation...
        .gatherAndValidateNumericParameter(value,'default',expectedSize);
catch exception
    throwAsCaller(exception)
end
end