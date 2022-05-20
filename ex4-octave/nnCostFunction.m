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
         
% the ones are multiplied with theta for a constant offset term
X = [ones(m, 1) X];

% each label is converted to an array of length num_labels with each array element corresponding to the row value in y set to 1 and all others to 0
new_y = zeros(m, num_labels);
% for each row in new_y, the value of the corresponding element in y is used to index the element in the new_y row that is set to 1
for row = 1:m
    new_y(row, y(row)) = 1;
end
y = new_y;

% calculate hypothesis: a1 = X; a2 = sigmoid(Theta1 * a1); a3 = sigmoid(Theta2 * a2);
% Theta1 size: 25 x 401. X size: m x 401
% a2 size: m x 25
a2 = sigmoid(X * transpose(Theta1));
% same thing but add the constant node to Theta2
% Theta1 size: 10 x 26. a2 size: m x 25
% a3 size: m * 10
a3 = sigmoid([ones(m, 1) a2] * transpose(Theta2));

% a3 is the output of the final model layer, with m rows and 10 columns
hypothesis = a3;

size(hypothesis)
% You need to return the following variables correctly 
J = 0;

for i = 1:m
    example_cost = 1 / m * sum(-y(i, :) .* log(hypothesis(i, :)) - ((1 - y(i, :)) .* log(1 - hypothesis(i, :))));
    J += example_cost;
end

Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
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



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
