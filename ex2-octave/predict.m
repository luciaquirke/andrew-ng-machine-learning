function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

% p is a vector of 0's and 1's
% >= is applied elementwise, so this line is equivalent to 
% p = arrayfun(@(x) x >= 0.5, X * theta);
p = sigmoid(X * theta) >= 0.5;

end
