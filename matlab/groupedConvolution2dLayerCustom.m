function layer = groupedConvolution2dLayerCustom( varargin )
% groupedConvolution2dLayer   2D grouped convolution layer
%
%   layer = groupedConvolution2dLayer(filterSize, numFiltersPerGroup,
%   numGroups) creates a layer for 2D grouped convolution. filterSize
%   specifies the height and width of the filters. It can be a scalar, in
%   which case the filters will have the same height and width, or a vector
%   [h w] where h specifies the height for the filters, and w specifies the
%   width. numFiltersPerGroup specifies the number of filters per group.
%   numGroups specifies the number of groups. It can be a numeric scalar or
%   the string 'channel-wise'. If 'channel-wise', the layer is setup for
%   channel-wise (or depth-wise) convolution where the number of groups
%   equals the number of incoming channel and is automatically determined
%   during training. 
%   
%   The number of channels per group is automatically set during training
%   and equals the number of input channels divided by the number of
%   groups.
% 
%   layer = groupedConvolution2dLayer(filterSize, numFiltersPerGroup,
%   numGroups, 'PARAM1', VAL1, 'PARAM2', VAL2, ...) specifies optional
%   parameter name/value pairs for creating the layer:
%
%       'Stride'                  - The step size for traversing the input
%                                   vertically and horizontally. This can
%                                   be a scalar, in which case the same
%                                   value is used for both dimensions, or
%                                   it can be a vector [u v] where u is the
%                                   vertical stride, and v is the
%                                   horizontal stride. The default is 
%                                   [1 1].
%       'DilationFactor'            - The step size for sampling the input or
%                                   equivalently the up-sampling factor of
%                                   the filter. It corresponds to an
%                                   effective filter size of filterSize +
%                                   (filterSize-1) * (dilationFactor-1).
%                                   This can be a scalar, in which case the
%                                   same value is used for both dimensions,
%                                   or it can be a vector [dHeight dWidth]
%                                   where dHeight is the vertical dilation,
%                                   and dWidth is the horizontal dilation.
%                                   The default is [1 1].
%       'Padding'                 - The padding applied to the input
%                                   along the edges. This can be:
%                                     - the character array 'same'. Padding
%                                       is set so that the output size 
%                                       is the same as the input size 
%                                       when the stride is 1. More 
%                                       generally, the output size is 
%                                       ceil(inputSize/stride), where 
%                                       inputSize is the height and width 
%                                       of the input.
%                                     - a scalar, in which case the same
%                                       padding is applied vertically and
%                                       horizontally.
%                                     - a vector [a b] where a is the 
%                                       padding applied to the top and 
%                                       bottom of the input, and b is the
%                                       padding applied to the left and 
%                                       right.
%                                     - a vector [t b l r] where t is the
%                                       padding applied to the top, b is
%                                       the padding applied to the bottom,
%                                       l is the padding applied to the 
%                                       left, and r is the padding applied 
%                                       to the right.
%                                   Note that the padding dimensions must
%                                   be less than the pooling region
%                                   dimensions. The default is 0.
%       'Weights'                 - Layer weights, specified as a
%                                   filterSize-by-numChannelsPerGroup-by-
%                                   numFiltersPerGroup-by-numGroups array
%                                   or [], where numChannelsPerGroup is the
%                                   number of channels per group. The
%                                   default is [].
%       'Bias'                    - Layer biases, specified as a
%                                   1-by-1-by-numFiltersPerGroup-by-
%                                   numGroups array or []. The default is
%                                   [].
%       'WeightLearnRateFactor'   - A number that specifies multiplier for
%                                   the learning rate of the weights. The
%                                   default is 1.
%       'BiasLearnRateFactor'     - A number that specifies a multiplier
%                                   for the learning rate for the biases. 
%                                   The default is 1.
%       'WeightL2Factor'          - A number that specifies a multiplier
%                                   for the L2 weight regulariser for the
%                                   weights. The default is 1.
%       'BiasL2Factor'            - A number that specifies a multiplier
%                                   for the L2 weight regulariser for the
%                                   biases. The default is 0.
%       'WeightsInitializer'      - The function to initialize the weights,
%                                   specified as 'glorot', 'he',
%                                   'narrow-normal', 'zeros', 'ones' or a
%                                   function handle. The default is
%                                   'glorot'.
%       'BiasInitializer'         - The function to initialize the bias,
%                                   specified as 'narrow-normal', 'zeros',
%                                   'ones' or a function handle. The
%                                   default is 'zeros'.
%       'Name'                    - A name for the layer. The default is
%                                   ''.
%   Example 1:
%       Create a grouped convolutional layer with 2 groups of 10 filters
%       each that have a height and width of 11, and that down-samples the
%       input by a factor of 4 in both the horizontal and vertical
%       directions.
%
%       layer = groupedConvolution2dLayer(11, 10, 2, 'Stride', 4, ...
%           'Padding', 'same');
%
%   Example 2:
%       Create a channel-wise convolutional layer, where each incoming
%       channel belongs to a separate group. Specify that the layer's
%       output has the same number of channels as the input by choosing a
%       single filter per group with height and width 3:
%
%       layer = groupedConvolution2dLayer(3, 1, 'channel-wise');
%
%   Example 3:
%       Create a grouped convolutional layer with 3 groups of 5 filters
%       each that have a height of 2 and width of 4. Manually initialize
%       the weights from a Gaussian with standard deviation 0.01,
%       corresponding to 6 incoming channels per group:
%
%       layer = groupedConvolution2dLayer([2 4], 5, 3, ...
%           'Weights', randn([2 4 6 5 3])*0.01);
%
%   Example 4:
%       Create a separable convolutional layer by composing a channel-wise
%       convolution and a point-wise convolution. Specify the filter
%       multiplicity of the channel-wise step to be 3, the filter size to
%       be 5-by-5 and the stride to be 2 in both horizontal and vertical
%       directions. Specify the number of filters of the point-wise layer
%       to be 32, and set the learn rate factor of the bias of the
%       channel-wise step to zero to avoid redundancy in the
%       parametrization:
%
%       separableConvolution = [
%           groupedConvolution2dLayer(5, 3, 'channel-wise', ...
%               'Stride', 2, 'BiasLearnRateFactor', 0)
%           convolution2dLayer(1, 32)];

%   See also nnet.cnn.layer.GroupedConvolution2DLayer, convolution2dLayer.

%   Copyright 2018 The MathWorks, Inc.

% Parse the input arguments.
args = iParseInputArguments(varargin{:});

% Create an internal representation of a convolutional layer.
internalLayer = nnet.internal.cnn.layer.GroupedConvolution2DCustom(args.Name, ...
    args.FilterSize, ...
    args.NumChannelsPerGroup, ...
    args.NumFiltersPerGroup, ...
    args.NumGroups, ...
    args.Stride, ...
    args.DilationFactor, ...
    args.PaddingMode, ...
    args.PaddingSize);

internalLayer.Weights.L2Factor = args.WeightL2Factor;
internalLayer.Weights.LearnRateFactor = args.WeightLearnRateFactor;

internalLayer.Bias.L2Factor = args.BiasL2Factor;
internalLayer.Bias.LearnRateFactor = args.BiasLearnRateFactor;

layer = nnet.cnn.layer.GroupedConvolution2DLayerCustom(internalLayer);
layer.WeightsInitializer = args.WeightsInitializer;
layer.BiasInitializer = args.BiasInitializer;
layer.Weights = args.Weights;
layer.Bias = args.Bias;

if ~isempty(args.ChannelMean)
    layer.ChannelMean = args.ChannelMean;
end
end

function inputArguments = iParseInputArguments(varargin)
varargin = nnet.internal.cnn.layer.util.gatherParametersToCPU(varargin);
parser = iCreateParser();
parser.parse(varargin{:});
inputArguments = iConvertToCanonicalForm(parser.Results);
end

function p = iCreateParser()
p = inputParser;
defaultStride = 1;
defaultDilationFactor = 1;
defaultPadding = 0;
defaultWeightLearnRateFactor = 1;
defaultBiasLearnRateFactor = 1;
defaultWeightL2Factor = 1;
defaultBiasL2Factor = 0;
defaultName = '';
defaultWeightsInitializer = 'glorot';
defaultBiasInitializer = 'zeros';
defaultLearnable = [];
defaultChannelMean = [];

p.addParameter('Name', defaultName, @iAssertValidLayerName);
p.addRequired('FilterSize',@iAssertValidFilterSize);
p.addRequired('NumFiltersPerGroup',@iAssertScalarPositiveInteger);
p.addRequired('NumGroups', @iAssertValidNumGroups);
p.addParameter('Stride', defaultStride, @iAssertValidStride);
p.addParameter('DilationFactor', defaultDilationFactor, @iAssertValidDilationFactor);
p.addParameter('Padding', defaultPadding, @iAssertValidPadding);
p.addParameter('WeightLearnRateFactor', defaultWeightLearnRateFactor, @iAssertValidFactor);
p.addParameter('BiasLearnRateFactor', defaultBiasLearnRateFactor, @iAssertValidFactor);
p.addParameter('WeightL2Factor', defaultWeightL2Factor, @iAssertValidFactor);
p.addParameter('BiasL2Factor', defaultBiasL2Factor, @iAssertValidFactor);
p.addParameter('WeightsInitializer', defaultWeightsInitializer);
p.addParameter('BiasInitializer', defaultBiasInitializer);
p.addParameter('Weights', defaultLearnable);
p.addParameter('Bias', defaultLearnable);
p.addParameter('ChannelMean', defaultChannelMean);
end

function iAssertValidFilterSize(value)
nnet.internal.cnn.layer.paramvalidation.validateSizeParameter(value, 'FilterSize');
end

function iAssertScalarPositiveInteger(value)
validateattributes(value, {'numeric'}, ...
    {'scalar','integer','positive'});
end

function iAssertValidStride(value)
nnet.internal.cnn.layer.paramvalidation.validateSizeParameter(value, 'Stride');
end

function iAssertValidDilationFactor(value)
nnet.internal.cnn.layer.paramvalidation.validateSizeParameter(value, 'DilationFactor');
end

function iAssertValidPadding(value)
nnet.internal.cnn.layer.paramvalidation.validatePadding(value);
end

function iAssertValidNumGroups(value)
if(ischar(value) || isstring(value))
    validatestring(string(value),"channel-wise");
else
    iAssertScalarPositiveInteger(value)
end
end

function iAssertValidFactor(value)
nnet.internal.cnn.layer.paramvalidation.validateLearnFactor(value);
end

function iAssertValidLayerName(name)
nnet.internal.cnn.layer.paramvalidation.validateLayerName(name)
end

function inputArguments = iConvertToCanonicalForm(params)
% Make sure integral values are converted to double and strings to char vectors
inputArguments = struct;
inputArguments.FilterSize = double( iMakeIntoRowVectorOfTwo(params.FilterSize) );
inputArguments.NumFiltersPerGroup = double( params.NumFiltersPerGroup );
inputArguments.Stride = double( iMakeIntoRowVectorOfTwo(params.Stride) );
inputArguments.DilationFactor = double( iMakeIntoRowVectorOfTwo(params.DilationFactor) );
inputArguments.PaddingMode = iCalculatePaddingMode(params.Padding);
inputArguments.PaddingSize = double( iCalculatePaddingSize(params.Padding) );
inputArguments.NumChannelsPerGroup = [];
inputArguments.NumGroups = double( iConvertToEmptyIfChannelWise(params.NumGroups) );
inputArguments.WeightLearnRateFactor = params.WeightLearnRateFactor;
inputArguments.BiasLearnRateFactor = params.BiasLearnRateFactor;
inputArguments.WeightL2Factor = params.WeightL2Factor;
inputArguments.BiasL2Factor = params.BiasL2Factor;
inputArguments.WeightsInitializer = params.WeightsInitializer;
inputArguments.BiasInitializer = params.BiasInitializer;
inputArguments.Weights = params.Weights;
inputArguments.Bias = params.Bias;
inputArguments.Name = char(params.Name);
inputArguments.ChannelMean = params.ChannelMean;
end

function tf = iIsRowVectorOfTwo(x)
tf = isrow(x) && numel(x)==2;
end

function rowVectorOfTwo = iMakeIntoRowVectorOfTwo(scalarOrRowVectorOfTwo)
if(iIsRowVectorOfTwo(scalarOrRowVectorOfTwo))
    rowVectorOfTwo = scalarOrRowVectorOfTwo;
else
    rowVectorOfTwo = [scalarOrRowVectorOfTwo scalarOrRowVectorOfTwo];
end
end

function paddingMode = iCalculatePaddingMode(padding)
paddingMode = nnet.internal.cnn.layer.padding.calculatePaddingMode(padding);
end

function paddingSize = iCalculatePaddingSize(padding)
paddingSize = nnet.internal.cnn.layer.padding.calculatePaddingSize(padding);
end

function y = iConvertToEmptyIfChannelWise(x)
if(iIsChannelWiseString(x))
    y = [];
else
    y = x;
end
end

function tf = iIsChannelWiseString(x)
tf = strcmp(x, 'channel-wise');
end