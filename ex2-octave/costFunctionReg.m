function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 1 / m * sum(-y .* log(sigmoid(X * theta)) - (1 - y) .* log(1 - sigmoid(X * theta))) + lambda / (2 * m) * sum(theta(2:end) .^ 2);
cost = sigmoid(X * theta) - y;
regularization = (lambda / m .* theta(2:end));
grad = (1 / m .* transpose(X) * cost) + [0; regularization];

end
