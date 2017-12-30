%%week 1 linear regression with single vairable
%%initialization
clear;
close all; clc
fprintf('Running the warmupexercise...\n');

%%warmupexercise function
function A=warmupexercise()
  
  A=eye(5);
end
warmupexercise()
pause;

%%plot data
data=load('ex1data1.txt');
X=data(:,1); y=data(:,2);
m=length(y);

%%plotdata function
function plotdata(x,y)
  figure;
  plot(x,y,'rx','MarkerSize',10);
  ylabel('Profit in $10,000S');
  xlabel('Population of city in 10,000S');
end  
plotdata(X,y);
pause;

%%Cost function
X=[ones(m,1),data(:,1)];
theta=zeros(2,1);

iterations=1500;
alpha=0.01;

%% computecost function
function J=computecost(X,y,theta)
  m=length(y);
  J=0;
  J=(1/(2*m))*sum((X*theta-y).^2);
end 
J=computecost(X,y,theta);
fprintf('with theta=[0,0]\nCost function= %f\n',J);
fprintf('Expected cost value(approx)32.07\n');
pause;

J=computecost(X,y,[-1;2]);
fprintf('with theta=[-1,2]\nCost function= %f\n',J);
fprintf('Expected cost value(approx)54.24\n');
pause;

%%gradientDescent function
function [theta, J_history]=gradientDescent(X,y,theta,alpha,num_iters)
m=length(y);
J_history=zeros(num_iters,1);
  for iter=1:num_iters
    delta=1/m*(X'*X*theta-X'*y);
    theta=theta-alpha.*delta;
    J_history(iter)=computecost(X,y,theta);

end
%J_history
end
theta=gradientDescent(X,y,theta,alpha,iterations);
pause;


%% plot the linear fit
hold on;
plot(X(:,2),X*theta,'-')
legend('training data','Linear regression')
hold off;
pause;

predict1=[1,3.5]*theta;
predict2=[1,7]*theta;
fprintf('[1,3.5]*theta= %f\n',predict1);
fprintf('[1,7]*theta= %f\n',predict2);
pause;

%%Visualizing J
theta0_val=linspace(-10,10,100);
theta1_val=linspace(-1,4,100);

J_vals=zeros(length(theta0_val),length(theta1_val));

for i=1:length(theta0_val)
  for j=1:length(theta1_val)
    t=[theta0_val(i);theta1_val(j)];
    J_vals(i,j)=computecost(X,y,t);
    end
    
end

J_vals=J_vals';
figure;
surf(theta0_val,theta1_val,J_vals)
xlabel('\theta_0');
ylabel('\theta_1');

figure;
contour(theta0_val,theta1_val,J_vals,logspace(-2,3,20))
xlabel('\theta_0');
ylabel('\theta_1');
hold on;
plot(theta(1),theta(2),'rx','MarkerSize',10,'LineWidth',2);














