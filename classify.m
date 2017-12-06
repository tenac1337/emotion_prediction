%# load your data
% input 41x12290

load Features.mat;
load Label_A.mat;
load Label_E.mat;
load Label_I.mat;
load Label_P.mat;
load Label_V.mat;

input = transpose(FV);
[numOfInstances numOfDims] = size(input);
numOfTrainInstances = .8 * numOfInstances;
numOfTestInstances = numOfInstances - numOfTrainInstances;

total_ind = 1:numOfInstances;
train_ind = randperm(numOfInstances, numOfTrainInstances);
test_ind = setdiff(total_ind, train_ind);
input_train = input(train_ind, 1:numOfDims);
input_test = input(test_ind, 1:numOfDims);

LL_A = LL_A - 1;
LL_E = LL_E - 1;
LL_I = LL_I - 1;
LL_P = LL_P - 1;
LL_V = LL_V - 1;

LL_A_train = LL_A(train_ind);
LL_E_train = LL_E(train_ind);
LL_I_train = LL_I(train_ind);
LL_P_train = LL_P(train_ind);
LL_V_train = LL_V(train_ind);

LL_A_test = LL_A(test_ind);
LL_E_test = LL_E(test_ind);
LL_I_test = LL_I(test_ind);
LL_P_test = LL_P(test_ind);
LL_V_test = LL_V(test_ind);

target_train = [LL_A_train, LL_E_train, LL_I_train, LL_P_train, LL_V_train];
target_test = [LL_A_test, LL_E_test, LL_I_test, LL_P_test, LL_V_test];

numOfClasses = size(target_train,2);
%# parameters of the learning algorithm
LEARNING_RATE = 0.1;
MAX_ITERATIONS = 10000;
MIN_ERROR = 1e-4;

% BEGIN TRAINING

%# five output nodes connected to 41-dimensional input nodes + biases
weights = randn(numOfClasses, numOfDims+1);

isDone = false;               %# termination flag
iter = 0;                     %# iterations counter
while ~isDone
    iter = iter + 1;

    %# for each instance
    err = zeros(numOfTrainInstances,numOfClasses);
    for i=1:numOfTrainInstances
        %# compute output: Y = W*X + b, then apply threshold activation
        output = (weights * [input_train(i,:)';1] >= 0 );                       %#'

        %# error: err = T - Y
        err(i,:) = target_train(i,:)' - output;                                  %#'

        %# update weights (delta rule): delta(W) = alpha*(T-Y)*X
        weights = weights + LEARNING_RATE * err(i,:)' * [input_train(i,:) 1];    %#'
    end

    %# Root mean squared error
    rmse = sqrt(sum(err.^2,1)/numOfTrainInstances);
    fprintf(['Iteration %d: ' repmat('%f ',1,numOfClasses) '\n'], iter, rmse);

    %# termination criteria
    if ( iter >= MAX_ITERATIONS || all(rmse < MIN_ERROR) )
        isDone = true;
    end
end

% EVALUATE

correct = 0;
for i=1:numOfTestInstances
       %# compute output: Y = W*X + b, then apply threshold activation
       output_test = (weights * [input_test(i, :)';1] >= 0 );
       output_test_val = output_test';
       target_test_val = target_test(i,:);
       disp(output_test_val);
       disp(target_test_val);
       if output_test_val == target_test_val
           correct = correct + 1;
       end
end 