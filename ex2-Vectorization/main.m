% Change the filenames if you've saved the files under different names
% On some platforms, the files might be saved as
% train-images.idx3-ubyte / train-labels.idx1-ubyte
images = loadMNISTImages('../mnist/train-images-idx3-ubyte');
labels = loadMNISTLabels('../mnist/train-labels-idx1-ubyte');

% We are using display_network from the autoencoder code
display_network(images(:,1:100)); % Show the first 100 images
disp(labels(1:10));
print -djpeg example.jpg   % save the visualization to a file

visibleSize = 28*28;
hiddenSize = 196;
sparsityParam = 0.1;
lambda = 3e-3;
beta = 3;
patches = images(:, 1:10000);

theta = initializeParameters(hiddenSize, visibleSize);

addpath minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 400;    % Maximum number of iterations of L-BFGS to run
options.display = 'on';

[opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   visibleSize, hiddenSize, ...
                                   lambda, sparsityParam, ...
                                   beta, patches), ...
                              theta, options);
W1 = reshape(opttheta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
display_network(W1', 12);

print -djpeg weights.jpg   % save the visualization to a file

