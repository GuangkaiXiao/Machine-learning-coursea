%%week 1 linear regression with multiple vairables
%%initialization
clear;
close all; clc
fprintf('Loading data...\n');

data=load('ex1data2.txt');
X=data(:,1:2); y=data(:,3);
m=length(y);
pause;

%%featureNormalize function
X_norm=X;
mu=zeros(1,size(X,2));
sigma=zeros(1,size(X,2));

mean_X1=mean(X(:,1));
std_X1=std(X(:,1));
mean_X2=mean(X(:,2));
std_X2=std(X(:,2));

mu(:,1)=mean_X1;
mu(:,2)=mean_X2;
sigma(:,1)=std_X1;
sigma(:,2)=std_X2;

m=length(y);
for i=1:m
X_norm(i,1)=(X(i,1)-mu(:,1))/sigma(:,1);
X_norm(i,2)=(X(i,2)-mu(:,2))/sigma(:,2);
end
X=[ones(m,1) X_norm];

alpha=0.01;
num_iters=400;
theta=zeros(3,1);

%%computeCostMulti function
function J = computeCostMulti(X, y, theta)
m = length(y); 
J = 0;
h=X*theta;
J=1/(2*m)*sum((h-y).^2);
end

%% gradientDescentMulti function
function [theta,J_history]=gradientDescentMulti(X,y,theta,alpha,num_iters)
m=length(y);
J_history=zeros(num_iters,1);
for iter=1:num_iters

delta=1/m*(X'*X*theta-X'*y);

theta=theta-alpha.*delta;
J_history(iter)=computeCostMulti(X,y,theta);
end
end


[theta,J_history]=gradientDescentMulti(X,y,theta,alpha,num_iters)

%%Plot the convegence graph
figure;
plot(1:numel(J_history),J_history,'-b','LineWidth',2);
xlabel('number of iteration');
ylabel('Cost J');
fprintf('Theta computed from gradient descent:\n');
fprintf('%f \n',theta);
pause;

%%Estimate the price of a 1650sq-ft, 3br house
XX=load('ex1data2.txt');
XX=[ones(m,1) XX];
X=XX;
price=[0,1650,3]*theta;
fprintf('for (0,1650,3)..using gradient descent $ %f\n',price);

%% solve with normal equation
data=csvread('ex1data2.txt');
X=data(:,1:2);
Y=data(:,3);
m=length(y);
X=[ones(m,1) X];
pause;

%% normalEqn function
function [theta]=normalEqn(X,y)

theta=zeros(size(X,2),1);
theta=pinv(X'*X)*X'*y;
end

theta=normalEqn(X,y);

printf('Theta computed from the mormal equation:\n');
printf('%f\n',theta);

price=[0,1650,3]*theta;
fprintf('for (0,1650,3)..using gradient descent $ %f\n',price);









