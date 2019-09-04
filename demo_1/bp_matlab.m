%% task: use no matlab module to construct a 'fake' fc-neural network model 
% the trained model is fitting for y=f(x)
%% model of the full-connected networks
%   in    hid    out  
%      /o -- o\
%     //o -- o \
%    // o -- o  \
%   //. o -- o . \
%  o  .        .  o
%   \\. o -- o . /
%    \\ o -- o  /
%     \\o -- o /
%      \o -- o/
% mention that the number of weight between hiden_layer & output_layer is 8
% not 8*8=64 that perhaps demonstrates the "bad" regression result  
%% Back-Propagation y = 1 + x + x*x
% 117082910078-donghao 2018-6-4
clc;
clear;
%% step0:
TrainCount = 5000; % train 10000 epochs
DataNum = 1001; % data set 1001
Hide_Out = zeros(1,8); % initial hide layer output(train) = 1 x 8 matrix
Hide_OutTest = zeros(1,8); % initial hide layer output(test) = 1 x 8 matrix
% learning rate is the speed for parameter updating
Rate_H = 0.02; % hiden layer learning rate
Rate_O = 0.01; % output layer learning rate
%% step1:
x = -5:0.01:6;% derive sample data date set is 1101 double numbers
y = 1 + x + x.*x;% y(label) is derived from x by this function

%% step2: initialize parameters & normalize data
x_Nor = (x -(-5) + 1)/(5 - (-5) + 1);% normalizing x data
% normalized_x=(x-x_min)/(x_max-x_min);
y_Nor = (y - 0.75 + 1)/(31 - 0.75 + 1);% normalizing label data
% normalized_y=(y-y_min)/(y_max-y_min);
% initialize 8 hiden layer weights as rand number in [-1,1]
w=2*(rand(1,8)-0.5);
% initialize 8 output layer weights as rand number in [-1,1]
v=2*(rand(1,8)-0.5);
dw = zeros(1,8);% initialize hiden layer biases as 0 
dv = zeros(1,8);% initialize output layer biases as 0
%% seperate the date set
xNum=length(x);% total 1101 data
Itest=1:11:xNum;% itest=1,12,23... test data set has 101 samples
Xtest_Nor=x_Nor(Itest);% derive corresponding data by the indexes
Ytest_Nor=y_Nor(Itest);% derive corresponding label by the indexes

Xtrain_Nor=x_Nor;% normalized_x_for_train
Ytrain_Nor=y_Nor;% normalized_y_for_train
Xtrain_Nor(Itest)=[];% cut down the test data from train data
Ytrain_Nor(Itest)=[];

%% step3:??????
Count = 0;%???????
e2 = 0;%
while(1)
    %% training data capcibility = 1000
    % initial error
    e1 = 0;
    for i = 1:1000
       %% forward-propagation
            for j = 1:8
               % hiden_layer=weight1*train_data
               sum1 = w(j)*Xtrain_Nor(i);
               % recitfier_function=logistic
               Hide_Out(j) = 1/(1 + exp(-1*sum1));
            end
            sum2 = 0;
            for k = 1:8
                % output_layer=weight2*hiden_layer
                % sum2=v1*hide_output1+v2*output2+...v8*output8
                sum2 = sum2 + v(k)*Hide_Out(k);
            end
            % output of full-connected network
            outputdata = sum2;
            % get the error=|nn_output-label|+error
            e1 = e1 + abs(outputdata - Ytrain_Nor(i));
       %% back-propagation
            for s = 1:8
                %% the kernel of back-propagation
                
                % refreshing output layer weights
                % the function of d_weights is based on the b-p markdown
                % function(7) and function(12)
                dv(s) = Rate_O * (Ytrain_Nor(i) - outputdata) * Hide_Out(s);
                % new_weight = old_weight + d_weights
                v(s) = v(s) + dv(s);
                
                % refreshing hiden layer weights
                alfa = 0;
                % get the delta: (label - output)
                % the form of alfa is due to the structure of hiden layer
                % based on the b-p markdown function(11) function(12)
                alfa = alfa + (Ytrain_Nor(i) - outputdata) * v(s);
                % (Hide_Out(s)*(1 - Hide_Out(s)))=the derivative of f'(z_i^l) in function (11)
                % xtrain_nor=a_j^l in function (12)
                dw(s) = Rate_H * alfa * (Hide_Out(s)*(1 - Hide_Out(s))) * Xtrain_Nor(i);
                % new_weight = old_weight + d_weights
                w(s) = w(s) + dw(s);
            end
    end
    %% compute the error on training dateset
    e11 = e1 / 1000; % compute the error on every data
    myErr1(Count+1)=e11; 
    %% test the neural network model
    e2=0; % initialize error on test data set
    trained_model_output_container=zeros(1,101);
    for p = 1:100 % test data set capcibility = 100
        %% forward propagation on test data set
        for l = 1:8
            sum3 = w(l)* Xtest_Nor(p);
            Hide_OutTest(l) =  1/(1 + exp(-1*sum3));
        end
        sum4 = 0;
        for m = 1:8
            sum4 = sum4 + v(m)*Hide_OutTest(m);
        end 
        % test data set f-p output
        OutputData_Test = sum4;
        % test date set total error
        e2 = e2 + abs(OutputData_Test - Ytest_Nor(p));
        trained_model_output_container(p) = OutputData_Test;
    end
    %% compute error
    e12 = e2 / 100; % e12 is error on each sample
    myErr2(Count+1)=e12;
    %% epochs+1
    Count = Count + 1;
    %% plot the error map
    % every 100 counts paint the error map once
    if mod(Count,10000)==0 || Count==TrainCount
    figure(1);
    hold on
    % plot(Count,e11,'r',Count,e12,'b');
    plot(1:Count,myErr1,'r',1:Count,myErr2,'g')
    title('train & test error on data set is')
    hold off
    end
    %% training terminte-criteria
    if(Count >TrainCount || e12 < 0.0001)
        break; 
    end
end
%% plot the y=f(x) & fitting curve
figure(2);
hold on
x_new=Xtest_Nor(1:100);
y_new=trained_model_output_container(1:100);
plot(x_Nor,y_Nor,'r');
plot(x_new,y_new,'b');
title('fc-net(1xnx1) to fit the curve')
xlabel('x');
ylabel('y');
hold off