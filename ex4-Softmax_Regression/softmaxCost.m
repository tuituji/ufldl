function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

e_theta_x = exp(theta * data);
e_theta_x_sum = sum(e_theta_x, 1);
e_theta_x_sum = repmat(e_theta_x_sum, [numClasses, 1]);
p = e_theta_x ./ e_theta_x_sum;
sum_k = sum(groundTruth .* log(p), 1);
cost = -sum(sum_k) / numCases + lambda * sum(sum(theta .* theta)) / 2;

thetagrad = - 1.0 / numCases * (groundTruth - p) * data' + lambda * theta;

% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

