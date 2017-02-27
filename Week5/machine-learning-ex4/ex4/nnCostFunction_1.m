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


X = [ones(m,1) X];
tot=0;
tot=zeros(m,1);

for i = 1:m,


% x is row vector of 1*401 , Th theta is a matrix of 25*401
% to get activation of layer of 1*25
z_value_2 = X(i,:) * Theta1';
%z_value_2 = Theta1 * X(i,:)';
activation_layer_2 =sigmoid(z_value_2);

%num_cols_a2 = size(activation_layer_2,2);
%activation_layer_2(:,num_cols_a2+1) = ones(1,1);
% this will give activation_layer_2 of 1*26 where 1st element is 1
activation_layer_2=[1 activation_layer_2];



% Theta2 is 10*26  , activation_layer_2 is 1*26 =>>  activation_layer_3 is 1*10
%z_value_3 = Theta2 * activation_layer_2;
z_value_3 = activation_layer_2 * Theta2'  ;
activation_layer_3= sigmoid(z_value_3);

%fprintf('%f activation_3\n', activation_layer_3);


op_rows = size(activation_layer_3,2);
yy=zeros(op_rows,1);

yy(y(i),:)=1;

%fprintf('%f   both\n' ,yy);


t1 = yy .* log(activation_layer_3');

%fprintf('%f   Temr-1\n' ,t1);

t2 = (1-yy) .* log((1-activation_layer_3'));

%fprintf('%f   Temr-2\n' ,t2);


sum_tot= sum(t1+t2);

%fprintf('%f   sum_total\n' ,sum_tot);

tot(i,:) = ((-1/m)*(sum_tot));

%val = ((-1/m)*(sum_tot));
%fprintf('%f   total\n' ,val);





end ;


% Backpropogation

delta_1=0;
delta_2=0;
delta_3=0;

for i =1:3,

% X { 5000*401} , Theta1 { 25*401} , x {1*401}
z_2 = X(i,:) * Theta1';
activation_layer_2 = sigmoid(z_2); % {1*25}
% Theta2 {10*26} activation_layer_1 {1*26}
activation_layer_2=[1 activation_layer_2];
z_3 = activation_layer_2 * Theta2'; % {1*10}
activation_layer_3 = sigmoid(z_3);
%fprintf('%f   Counter\n' ,i);


%fprintf('%f   activation_layer_3\n' ,activation_layer_3);


yy = zeros(size(activation_layer_3,2),1);


yy(y(i),:)=1;

%fprintf('%f   actual_OP\n' ,yy);


error_3 = activation_layer_3'-yy  ;

z_2=[1 z_2];

%fprintf('%f   Theta2_size\n' ,size(Theta2'));
%fprintf('%f   err_3\n' ,size(error_3));
%fprintf('%f   sigmoidGradient\n' ,size((sigmoidGradient(z_2))'));



error_2 = (Theta2' * error_3) .* (sigmoidGradient(z_2))';
%removing bias term
error_2 = error_2(2:end);


%if (i==1)
%delta_1=zeros(size(error_3,1),1);
%delta_2 = zeros(size(error_2,1),1);
%endif;
%activation_layer_2=activation_layer_2(2:end);

%fprintf('%f   error_2\n' ,size(error_2));
%fprintf('%f   error_3\n' ,size(error_3));

%fprintf('%f   ac_1\n' ,size(X(i,2:end)'));

delta_1 = delta_1 + (error_2 * X(i,:)) ;
delta_2 = delta_2 + (error_3 * activation_layer_2) ;


%fprintf('%f   delta_1\n' ,size(delta_1));

%fprintf('%f   delta_2\n' ,size(delta_2));

%delta_1=delta_1+ error_2 * sigmoid(X(i,:));
%delta_2=delta_2+ error_3 * sigmoid(X(i,:));


%fprintf('%f   delta_1\n' ,size(delta_1));


end;

%delta_2 = delta_2(2:end);

%fprintf('%f   drror_1\n' ,size(delta_1));
%fprintf('%f   drror_2\n' ,size(delta_2));
Theta1_grad = (1/m) * (delta_1);
Theta2_grad = (1/m) * (delta_2);


%fprintf('%f   Theta1_grad\n' ,size(Theta1_grad));

%fprintf('%f   Theta2_grad\n' ,size(Theta2_grad));





%%%Regularized Version



Theta1(:,1)=0;
Theta2(:,1)=0;

sqrTheta1= Theta1 .^ 2;
sqrTheta2= Theta2 .^ 2;

%fprintf('%f   sqrTheta1\n' ,size(sqrTheta1));
%fprintf('%f   sqrTheta2\n' ,size(sqrTheta2));



sumSqrTheta1 = sum(sqrTheta1(:));
sumSqrTheta2 = sum(sqrTheta2(:));

%combineTheta= [sumSqrTheta1(:);sumSqrTheta2(:)]
%fprintf('%f   sumSqrTheta1\n' ,size(combineTheta));

%fprintf('%f   sumSqrTheta2\n' ,size(sumSqrTheta2));


reg_term= (lambda/(2*m)) * (sumSqrTheta1+sumSqrTheta2);




J=sum(tot)+reg_term;














% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
 