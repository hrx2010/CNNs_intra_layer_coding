function net = alexnetpy()
   load('alexnetv2.mat','weight','biases');                                                       
   net = alexnetv2;                                                                               
                                                                                                  
   switch class(net)                                                                              
     case 'SeriesNetwork'                                                                         
       lgraph = layerGraph(net.Layers);                                                           
     case 'DAGNetwork'                                                                            
       lgraph = layerGraph(net);                                                                  
   end                                                                                            
   lgraph = modifyConvParams(lgraph,weight,biases);                                               
   net = assembleNetwork(lgraph);
end