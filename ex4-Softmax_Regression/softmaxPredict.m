function [pred] = softmaxPredict(softmaxModel, data)

% softmaxModel - model trained using softmaxTrain
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
%
% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
% Unroll the parameters from theta
theta = softmaxModel.optTheta;  % this provides a numClasses x inputSize matrix
pred = zeros(1, size(data, 2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.

numClasses = softmaxModel.numClasses;

e_theta_x = exp(theta * data);
e_theta_x_sum = sum(e_theta_x, 1);
e_theta_x_sum = repmat(e_theta_x_sum, [numClasses, 1]);
p = e_theta_x ./ e_theta_x_sum;

[c, i] = max(p, [], 1);
pred = i;

% ---------------------------------------------------------------------

end

