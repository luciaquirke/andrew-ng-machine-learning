function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
best_error = 1;
candidate_vals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

for i = 1 : length(candidate_vals)
    for j = 1: length(candidate_vals)
        function implicit_sigma_gaussian_kernel = implicitSigmaGaussianKernel(x1, x2)
            implicit_sigma_gaussian_kernel = gaussianKernel(x1, x2, candidate_vals(j));
        end

        model = svmTrain(X, y, candidate_vals(i), @implicitSigmaGaussianKernel);
        predictions = svmPredict(model, Xval);
        model_error = mean(double(predictions ~= yval));
        
        if (model_error < best_error)
            best_error = model_error;
            C = candidate_vals(i);
            sigma = candidate_vals(j);
        end
    end
end




% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
end
