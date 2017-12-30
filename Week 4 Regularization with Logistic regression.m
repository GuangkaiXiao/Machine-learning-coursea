%%week 4 regularization with Logistic regression
%%initialization
clear;
close all; clc
fprintf('Loading data...\n');

data=load('ex2data2.txt');
X=data(:,[1:2]); y=data(:,3);

%%plotData function
function plotData(X,y)
figure;
hold on;
pos=find(y==1);
neg=find(y==0);
plot(X(pos,1),X(pos,2),'r+','LineWidth',2,'MarkerSize',7);
plot(X(neg,1),X(neg,2),'ko','LineWidth',2,'MarkerSize',7);
end

plotData(X,y);
hold on;
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
legend('y = 1', 'y = 0')
hold off;
pause;

%% sigmodid function
function g =sigmoid(z)
g=zeros(size(z));
g=1./(1+exp(-z));
end


%%mapFeature function
function out = mapFeature(X1, X2)
degree = 6;
out = ones(size(X1(:,1)));
for i = 1:degree
    for j = 0:i
        out(:, end+1) = (X1.^(i-j)).*(X2.^j);
    end
end

end

X = mapFeature(X(:,1), X(:,2));
initial_theta = zeros(size(X, 2), 1);
lambda = 1;

%%costFunctionReg function
function [J, grad] = costFunctionReg(theta, X, y, lambda)
m = length(y);  
J = 0;
grad = zeros(size(theta));
z=X*theta;
h=sigmoid(z);
logisf=(-y)'*log(h)-(1-y)'*log(1-h);

J=((1/m).*sum(logisf))+(lambda/(2*m)).*sum(theta.^2);

k=length(theta)-1;
n=length(theta);
grad(1)=1/m.*(sum(X'(1,:)*h-X'(1,:)*y));

for j=2:n
	grad(j)=(1/m).*(sum(X'(j,:)*h-X'(j,:)*y)+lambda.*theta(j,1));
end

end

[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);
fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Expected cost (approx): 0.693\n');
fprintf('Gradient at initial theta (zeros) - first five values only:\n');
fprintf(' %f \n', grad(1:5));
fprintf('Expected gradients (approx) - first five values only:\n');
fprintf(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n');

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

test_theta = ones(size(X,2),1);
[cost, grad] = costFunctionReg(test_theta, X, y, 10);

fprintf('\nCost at test theta (with lambda = 10): %f\n', cost);
fprintf('Expected cost (approx): 3.16\n');
fprintf('Gradient at test theta - first five values only:\n');
fprintf(' %f \n', grad(1:5));
fprintf('Expected gradients (approx) - first five values only:\n');
fprintf(' 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n');

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%%fminc
%% plotDecisionBoundary functuion
function plotDecisionBoundary(theta, X, y)
plotData(X(:,2:3), y);
hold on

if size(X, 2) <= 3
    % Only need 2 points to define a line, so choose two endpoints
    plot_x = [min(X(:,2))-2,  max(X(:,2))+2];

    % Calculate the decision boundary line
    plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));

    % Plot, and adjust axes for better viewing
    plot(plot_x, plot_y)
    
    % Legend, specific for the exercise
    legend('Admitted', 'Not admitted', 'Decision Boundary')
    axis([30, 100, 30, 100])
else
    % Here is the grid range
    u = linspace(-1, 1.5, 50);
    v = linspace(-1, 1.5, 50);

    z = zeros(length(u), length(v));
    % Evaluate z = theta*x over the grid
    for i = 1:length(u)
        for j = 1:length(v)
            z(i,j) = mapFeature(u(i), v(j))*theta;
        end
    end
    z = z'; % important to transpose z before calling contour

    % Plot z = 0
    % Notice you need to specify the range [0, 0]
    contour(u, v, z, [0, 0], 'LineWidth', 2)
end
hold off
end

initial_theta = zeros(size(X, 2), 1);
lambda = 1;
options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, J, exit_flag]=fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);

plotDecisionBoundary(theta, X, y);
hold on;
title(sprintf('lambda = %g', lambda))

xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
legend('y = 1', 'y = 0', 'Decision boundary')
hold off;
pause;

%% predict the accuracies
function p=predict(theta,X)
m=size(X,1);
p=zeros(m,1);
z=X*theta;
h=sigmoid(z);
for i=1:m
  if h(i)>=0.5
    p(i)=1
  else
    p(i)=0
end
end
end
p=predict(theta,X);
accuracies=mean(p==y)*100;
fprintf('the accuries for lambda=1 is %f\n',accuracies);




