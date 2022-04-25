function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
% theta has 2 rows and 1 col
% X has m rows and 2 cols
% X * theta has m rows and 1 col. y has m rows and 1 col. so we subtract
% for m rows and 1 col - the error for each datapoint. then we do pointwise
% squaring and average it out

J = sum((X * theta - y).^2)/(2 * m);

end
