function [f,g] = logistic_regression_vec(theta, X,y)
  %
  % Arguments:
  %   theta - A column vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));
  
  h = theta'*X;
  f = sum(y.*log(sigmf(h,[1,0])) + (1-y).*log(1-sigmf(h,[1,0])));
  %f = sum(bsxfun(@times,y,log(sigmf(f,[1,0]))) + bsxfun(@times,1-y,log(1-sigmf(f,[1,0]))));
  %
  % TODO:  Compute the logistic regression objective function and gradient 
  %        using vectorized code.  (It will be just a few lines of code!)
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %
%%% YOUR CODE HERE %%%
  g = X*(h - y)';