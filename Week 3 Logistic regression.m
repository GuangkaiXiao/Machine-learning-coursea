%%week 3 Logistic regression
%%initialization
clear;
close all; clc
fprintf('Loading data...\n');

data=load('ex2data1.txt');
X=data(:,1:2); y=data(:,3);

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
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend('Admitted','not Admitted');
hold off;
pause;

%%compute cost and gradient
[m,n]=size(X);
X=[ones(m,1),X];
initial_theta=zeros(n+1,1);

%% sigmodid function
function g =sigmoid(z)
g=zeros(size(z));
g=1./(1+exp(-z));
end

%%costFunction function
function [J,grad]=costFunction(theta,X,y)
 
J=0;
m=length(y);
grad=zeros(size(theta));
z=X*theta;
h=sigmoid(z);
logisf=(-y)'*log(h)-(1-y)'*log(1-h);
J=(1/m)*sum(logisf);
grad=1/m*((X'*h-X'*y)');
end
[cost,grad]=costFunction(initial_theta,X,y);
fprintf('cost function at theta(zeros):%f\n',cost);
fprintf('expected cost(aaprox):0.693\n');
pause;
test_theta=[-24;0.2;0.2];
[cost,grad]=costFunction(test_theta,X,y);
fprintf('cost function at test_theta:%f\n',cost);
fprintf('expected cost(aaprox):2,647\n');

%%fminc
options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta,cost]=fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);
fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);
pause;


%% plot bpoundary

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

plotDecisionBoundary(theta, X, y);
hold on;
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend('Admitted', 'Not admitted')
hold off;
pause;

%% predict and accuracies

prob=sigmoid([1 45 85]*theta);
fprintf(['For a student with scores 45 and 85, we predict an admission probability of %f\n'], prob);

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

p = predict(theta, X);
accuracies=mean(p==y)*100
fprintf('Train Accuracy: %f\n', accuracies);
fprintf('\nProgram paused. Press enter to continue.\n');
pause;