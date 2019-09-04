%% task: use no matlab module to construct a 'real' fc-neural network model 
% the kernel function is wx without b
% the trained model is fitting for y=f(x)
%% structure of the full-connected networks
%   in   hid   out  
%      /o X o\
%     //o X o \
%    // o X o  \
%   //. o X o . \
%  o  .   X   .  o
%   \\. o X o . /
%    \\ o X o  /
%     \\o X o /
%      \o X o/
% mention that the number of weight between hiden_layer & output_layer is 8
% not 8*8=64 that perhaps demonstrates the "bad" regression result  
%% Back-Propagation y = 1 + x + x*x
% 117082910078-donghao 2018-6-26
clc;
clear all;
%% step0:
traincount = 20000; % train 10000 epochs
batch_size=20; % set batch size for weights refreshing
datanum = 1001; % data set 1001

% learning rate is the speed for parameter updating
rate_hiden_layer = 0.1; % hiden layer learning rate
rate_output_layer = 0.05; % output layer learning rate
%% step1:
x = -5:0.01:6;% derive sample data set is 1101 double numbers
y = sin(x);
y_handle_1=@(x) sin(x);
y_handle_2=@(x) -sin(x);
%% step2: initialize parameters & normalize data
min=fminbnd(y_handle_1,-5,6);
minvalue=sin(min);

max=fminbnd(y_handle_2,-5,6);
maxvalue=sin(max);

normalized_x=(x-(-5))/(6-(-5));
normalized_y=(y-minvalue)/(maxvalue - minvalue);

% initialize 8 hiden layer weights as rand number in [-1,1]
w=rand(8,1);

% initialize 64 output layer weights as rand number in [-1,1]
v=(rand(8,8));

d_w = zeros(8,1);% initialize hiden layer biases as 0 
d_v = zeros(8,8);% initialize output layer biases as 0
%% seperate the data set
xNum=length(x);% total 1101 data
index_test=1:11:xNum;% itest=1,12,23... test data set has 101 samples
normalized_test_x=normalized_x(index_test);% derive corresponding data by the indexes
normalized_test_y=normalized_y(index_test);% derive corresponding label by the indexes

normalized_train_x=normalized_x;% normalized_x_for_train
normalized_train_y=normalized_y;% normalized_y_for_train
normalized_train_x(index_test)=[];% cut down the test data from train data
normalized_train_y(index_test)=[];

train_capcibility=length(normalized_train_x);
test_capcibility=length(normalized_test_x);
%% step3:??????
count = 0; % initial epochs
e2 = 0;
error2=0;
while(1)
    %% training data capcibility = 1000
    % initial error
    e1 = 0;
    for i = 1:train_capcibility
        %% forward-propagation
        % use matrix multiplation to replace for loops 
        % z1= w*x=8x1  normalized_train_x(i)=1x1  z2=8x1 vector
        z2=w*normalized_train_x(i);
        % rectifier func=logistic   a2=f(z2) a2 = 8x1 vector
        a2=1./(1 + exp(-z2));
        
        % v_ji weights from i to j v=8x8 a3=8x1 z3=8x1
        % v_ji row:j  column:i
        z3=v*a2;
        a3=1./(1 + exp(-z3));
        logit=sum(a3);
        % get the error=|nn_output-label|+error
        e1 = e1 + abs(logit - normalized_train_y(i));
        error2 = error2 + (logit - normalized_train_y(i));

% -------------------------------------------------------------
        
        %% back-propagation-1
        % output_layer&hiden_layer weights refreshing
        k=ones(1,8);
        % delta_4=1x1  b-p matlab function(7)
        delta_4=-(normalized_train_y(i) - logit)*1;
        % delta_3=8x1  b-p matlab function(9)
        delta_3=((k*delta_4)').*(a3.*(1-a3));
        % delta_2=8x1
        delta_2=((v')*delta_3).*(a2.*(1-a2));
        % d_w = 8x1 b-p matlab function(12)
        d_w=normalized_train_x(i)*delta_2;
        
        % d_v = 8x8
        d_v=a2*(delta_3'); 
        % transpose d_v  we get the final right d_v
        % for b-p matlab function (3) (12)
        d_v=d_v';

        % new_weight = old_weight - alpha * d_weights
        v = v - (rate_hiden_layer * d_v);
        % new_weight = old_weight - alpha * d_weights
        w = w - (rate_output_layer * d_w);

% -------------------------------------------------------------

%         %% back-propagation-2
%         if mod(i,batch_size)==0
%             error_on_batch=error2;
%             error2=0;
%             error_on_single_sample = error_on_batch/batch_size;
%         
%             % output_layer&hiden_layer weights refreshing
%             k=ones(1,8);
%             % delta_4=1x8 on batch
%             delta_4=-(error_on_single_sample)*1;
%             % delta3=1x8
%             delta_3=((k*delta_4)').*(a3.*(1-a3));
%             % delta2=1x8
%             delta_2=((v')*delta_3).*(a2.*(1-a2));
%             % d_w = 8x1
%             d_w=normalized_train_x(i)*delta_2;
%             % d_v = 8x8
%             d_v=a2*(delta_3');
%             % new_weight = old_weight - alpha * d_weights
%             v = v - (rate_hiden_layer * d_v);
%             % new_weight = old_weight - alpha * d_weights
%             w = w - (rate_output_layer * d_w);
%         end
    end
    %% compute the error on training dataset
    e11 = e1 / train_capcibility; % compute the average error on single data
    myerror1(count+1)=e11; 
    %% epochs+1
    count = count + 1;
    %% plot the error map
    % every 1000 counts paint the error map once
    if count==traincount
    figure(1);
    hold on
    % plot(Count,e11,'r',Count,e12,'b');
    plot(1:count,myerror1,'r')
    end
    %% training terminte-criteria
    if(count > traincount || e11 < 0.0001)
        break; 
    end
end
%% test the neural network model
e2=0; % initialize error on test data set
trained_model_output_container=zeros(1,100);
for p = 1:test_capcibility % test data set capcibility = 100
    %% forward propagation on test data set
    z2_test=w*normalized_test_x(p);  
    % rectifier func=logistic   a1=f(z1) a1 = 8x1 vector
    a2_test=1./(1 + exp(-z2_test));
        
    % v_ji weights from i to j v=8x8 a2=8x1 z3=8x1
    % v_ji row:j  column:i
    z3_test=v*a2_test;
    a3_test=1./(1 + exp(-z3_test));
    logit_test=sum(a3_test);
        
    % test data set total error
    e2 = e2 + abs(logit_test - normalized_test_y(p));
    trained_model_output_container(p) = logit_test;
end
%% compute error
e12 = e2 / test_capcibility; % e12 is error on single sample
plot(e12,'o')
title(['final average error on test set is ',num2str(e12)])
hold off
%% plot the y=f(x) & fitting curve
figure(2);
hold on
x_new=normalized_test_x(1:test_capcibility);
y_new=trained_model_output_container(1:test_capcibility);
plot(normalized_test_x,normalized_test_y,'r');
% plot(x_new,y_new,'b');
size=10;
scatter(x_new,y_new,size);
title('fc-net(1xnx1) to fit the curve')
xlabel('x');
ylabel('y');
hold off