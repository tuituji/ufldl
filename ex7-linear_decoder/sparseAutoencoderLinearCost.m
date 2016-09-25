function [cost,grad,features] = sparseAutoencoderLinearCost(theta, visibleSize, hiddenSize, ...
                                                            lambda, sparsityParam, beta, data)
% -------------------- YOUR CODE HERE --------------------
% Instructions:
%   Copy sparseAutoencoderCost in sparseAutoencoderCost.m from your
%   earlier exercise onto this file, renaming the function to
%   sparseAutoencoderLinearCost, and changing the autoencoder to use a
%   linear decoder.
% -------------------- YOUR CODE HERE --------------------                                    
W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

cost = 0;
W1grad = zeros(size(W1));
W2grad = zeros(size(W2));
b1grad = zeros(size(b1));
b2grad = zeros(size(b2));

num = size(data, 2);

z2 = W1 * data + repmat(b1, [1, num]);
a2 = 1 ./ (1 + exp(-z2));
z3 = W2 * a2 + repmat(b2, [1, num]);
a3 = z3;


h = a3;

rho_ = sum(a2, 2) / num;
rho = repmat(sparsityParam, [hiddenSize, 1]);

cost = sum(sum((h - data) .^ 2)) / (2 * num) + (lambda / 2) * (sum(sum(W1 .^ 2)) +  sum(sum(W2 .^ 2))) + beta * sum(KL_divergence(rho, rho_));


sparsity_delta = repmat(- rho ./ rho_ + (1 - rho) ./ (1 - rho_), [1, num]);

delta3 =  -(data - h);
delta2 =  (W2' * delta3 + beta * sparsity_delta) .* sigmoid1(z2);

W1grad = delta2 * data'/num + lambda * W1;
W2grad = delta3 * a2'/num + lambda * W2;
b1grad = sum(delta2,2)/num;
b2grad = sum(delta3,2)/num;

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

features = a3;

end


function kl = KL_divergence(x, y)
    kl = x .* log(x ./ y) + (1 - x) .* log((1 - x) ./ (1 - y));
end


function sigm = sigmoid(x)

    sigm = 1 ./ (1 + exp(-x));
end

function sigm1 = sigmoid1(x)

    sigm1 = sigmoid(x) .* (1 - sigmoid(x));
end

