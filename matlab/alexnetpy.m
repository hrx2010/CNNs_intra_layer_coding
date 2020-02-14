function net = alexnetpy()
   load('alexnetv2.mat','weight','biases','strides');
   net = alexnetv2;                                                                                                  
   switch class(net)                                                                              
     case 'SeriesNetwork'                                                                         
       lgraph = layerGraph(net.Layers);                                                           
     case 'DAGNetwork'                                                                            
       lgraph = layerGraph(net);                                                                  
   end                                                                                            
   lgraph = modifyConvParams(lgraph,weight,biases,strides);
   net = assembleNetwork(lgraph);
end