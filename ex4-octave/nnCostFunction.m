function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% this is a hacky data-science style way to convert a vector of label numbers into a matrix where each row has an element of value 1 at the corresponding index.
% it works by creating one of each possible row value ([0 0 0 0 0 0 0 0 0 1], [0 0 0 0 0 0 0 0 1 0] etc.) for a final 10 x 10 matrix, then using a vectorised array 
% index to select those rows as needed to form the final m x 10 matrix.
y_matrix = eye(num_labels)(y,:);
y = y_matrix;

% the ones are multiplied with theta for a constant offset term
a1 = [ones(m, 1) X];

% calculate hypothesis: a1 = X; a2 = sigmoid(Theta1 * a1); a3 = sigmoid(Theta2 * a2);
% Theta1 size: 25 x 401. X size: m x 401
% a2 size: m x 25
z2 = a1 * transpose(Theta1);
a2 = sigmoid(z2);
a2 = [ones(m, 1) a2];

% same thing but add the constant node to Theta2
% Theta1 size: 10 x 26. a2 size: m x 25
% a3 size: m * 10
z3 = a2 * transpose(Theta2);
a3 = sigmoid(z3);

% a3 is the output of the final model layer, with m rows and 10 columns
hypothesis = a3;

% You need to return the following variables correctly 
J = 0;
Delta1 = 0;
Delta2 = 0;

for i = 1:m
    example_cost = 1 / m * sum(-y(i, :) .* log(hypothesis(i, :)) - ((1 - y(i, :)) .* log(1 - hypothesis(i, :))));
    J += example_cost;
end

regularization = lambda / (2 * m) * (sum(sum( Theta1(:, 2:end).^2 )) + sum(sum(Theta2(:, 2:end).^2)));
J += regularization;

d3 = y - hypothesis;
d2 = d3 * Theta2(:,2:end) .* sigmoidGradient(z2);
Delta1 = Delta1 + transpose(d2) * a1;
Delta2 = Delta2 + transpose(d3) * a2;

% partial derivatives of J with respect to Theta1 and Theta2 respectively
Theta1_grad = 1 / m .* Delta1;
Theta2_grad = 1 / m .* Delta2;

% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
