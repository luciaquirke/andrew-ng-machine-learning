function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
regularization = lambda * sum(theta(2:end).^2);
regression_error = sum((X * theta - y).^2);
J = (regression_error + regularization) / (2 * m);


% % 1 column
% size(theta)

% for each row of (X * theta - y) multiply it by that same row of x
% 1 row result
unregularized_grad = transpose(X * theta - y) * X;

grad_regularization = [zeros(1, 1), transpose(lambda * theta(2:end))];
grad =  1 / m * (unregularized_grad + grad_regularization);

grad = grad(:);

end
