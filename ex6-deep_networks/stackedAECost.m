function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful
M = size(data, 2);
groundTruth = full(sparse(labels, 1:M, 1));


%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%
N = numel(stack);
z = cell(N+1, 1);
a = cell(N+1, 1);
a{1} = data;


for l = (1:N)
    z_temp = stack{l}.w * a{l};
    z{l+1} = bsxfun(@plus, z_temp, stack{l}.b);
    a{l+1} = sigmoid(z{l+1});
end

td = softmaxTheta * a{N+1};
td = bsxfun(@minus, td, max(td));
tmp = exp(td);
dsum = sum(tmp);
p = bsxfun(@rdivide, tmp, dsum);

y = groundTruth;
cost = (-1/M) * sum(sum(y .* log(p))) ;
%%% !!! Only add weight decay on classifiler layer
cost = cost + (lambda / 2) * sum(sum(softmaxTheta.^2));
    
softmaxThetaGrad = (-1/M) *(y - p) * a{N+1}' + lambda * softmaxTheta;

% delta
d = cell(N+1);
d{N+1} = -(softmaxTheta' * (y - p)) .* a{N+1} .* (1 -a{N+1});

for l = (N:-1:2)
    d{l} = stack{l}.w' * d{l+1} .* a{l} .* (1-a{l});
end

for l = (N:-1:1)
    stackgrad{l}.w = d{l+1} * a{l}' / M;
    stackgrad{l}.b = sum(d{l+1}, 2) / M;
end



% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
