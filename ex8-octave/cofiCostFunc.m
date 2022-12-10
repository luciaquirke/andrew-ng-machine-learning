function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

J = 1 / 2 * sum(
    ((X * transpose(Theta) - Y)(R == 1)).^2
    ) + lambda / 2 * sum(sum(
        Theta.^2
    )) + lambda / 2 * sum(sum(
        X.^2
    ));

Theta_grad = zeros(size(Theta));
X_grad = zeros(size(X));

% # user's coefficients
% size(Theta)
% % # movie's coefficients
% size(X)
% # user's actual ratings
% size(Y)
% # whether the user has rated a movie 
% size(R)

for i = 1 : num_movies
    X_grad(i, :) = R(i, :) .* (X(i, :) * transpose(Theta) - Y(i, :)) * Theta;
    regularization = lambda * X(i, :);
    X_grad(i, :) = X_grad(i, :) + regularization;
end

for j = 1 : num_users
    error = R(:, j) .* (X * transpose(Theta(j, :)) - Y(:, j));
    Theta_grad(j, :) = (transpose(error) * X);
    regularization = lambda * Theta(j, :);
    Theta_grad(j, :) = Theta_grad(j, :) + regularization;
end

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

grad = [X_grad(:); Theta_grad(:)];

end
