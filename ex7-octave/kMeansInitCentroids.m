function centroids = kMeansInitCentroids(X, K)
%KMEANSINITCENTROIDS This function initializes K centroids that are to be 
%used in K-Means on the dataset X
%   centroids = KMEANSINITCENTROIDS(X, K) returns K initial centroids to be
%   used with the K-Means on the dataset X
%

centroids = zeros(K, size(X, 2));

rand_indices = randperm(size(X, 1));
centroids = X(rand_indices(1:K), :);

end

