net = importONNXNetwork('unet.onnx','OutputLayerType','pixelclassification');
%%
dummy = ones(256,256,10);
%%
c = semanticseg(dummy,net);