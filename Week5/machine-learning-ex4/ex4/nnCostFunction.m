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
         
% You need to return the following variables correctly 
J = 0;
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


%Cost Function w/o regularization

costSum=0;
Theta1_wo_bias = Theta1(:,2:end);
Theta2_wo_bias = Theta2(:,2:end);



for i = 1:m,
yy=zeros(num_labels,1);


a1 = [1;X(i,:)'];

z2 = Theta1 * a1;
a2 = sigmoid(z2);

a2=[1;a2]; %bias term


z3 = Theta2 * a2;
a3 = sigmoid(z3);

yy(y(i),:)=1;


t1 = yy .* log(a3);
t2 = (1-yy) .* log(1-a3);

costSum = costSum + (-(1/m)) * sum(t1+t2);

end;

%regularization
layer1 = Theta1_wo_bias .^ 2;
layer2 = Theta2_wo_bias .^ 2;
%fprintf("%f \n size wo bias 1",layer1);
%fprintf("%f \n size wo bias 2",layer2);

regTerm = (lambda/(2*m)) * (sum(layer1(:)) + sum(layer2(:)));



J = costSum + regTerm;


%backpropogation
delta1=0;
delta2=0;

for t = 1:m,

yy = zeros(num_labels,1);

a1 = [1;X(t,:)'];

z2 = Theta1 * a1;
a2 = sigmoid(z2);

a2=[1;a2];

z3 = Theta2 * a2;
a3 = sigmoid(z3);


yy(y(t),1)=1;

error3 = a3 - yy;

error2 = (Theta2_wo_bias' * error3) .* sigmoidGradient(z2);

delta2 = delta2 + error3 * a2' ;
delta1 = delta1 + error2 * a1' ;

end;

%regularization for thetas


Theta1_wo_bias = [zeros(size(Theta1_wo_bias,1),1),Theta1_wo_bias];
Theta2_wo_bias = [zeros(size(Theta2_wo_bias,1),1),Theta2_wo_bias];


Theta1_grad = (1/m) * (delta1) .+  ((lambda/m) * (Theta1_wo_bias));
Theta2_grad = (1/m) * (delta2) .+  ((lambda/m) * (Theta2_wo_bias));



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
 