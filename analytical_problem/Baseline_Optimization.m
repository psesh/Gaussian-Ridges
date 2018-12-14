clear; close all; clc;
%==========================================================================
%
% A simple application of Manifold optimization to estimate a Gaussian
% ridge.
%
% Pranay Seshadri,
% The Alan Turing Institute
% December 6th, 2018
%==========================================================================
% Preliminaries
N = 50;
N_test = 50;
H = N + N_test + 1;
d = 10; m = 2;
test_train = randperm(H);
train = test_train(1:N);
test = test_train(N + 1:N + N_test);
X = rand(H, d)*2 - 1;
X_train = X(train, :);
X_test = X(test, :);
Ureal = rand(d, m);
[Q,~] = qr(Ureal); Ureal = Q(:,1:2);
U_train = X_train * Ureal;
U_test = X_test * Ureal;
f_train = zeros(N, 1) ; %+ 0.001 * randn(N, 1);
for i = 1 : N
    u = U_train(i,:);
    f_train(i) = ((u(1) + u(2)));
end
for i = 1 : N_test
    u = U_test(i,:);
    f_test(i) = ((u(1) + u(2)));
end

%% Optimization test!
Uo = randn(d,d); [Q, ~] = qr(Uo); Uo = Q(:,1:2);
[N, d] = size(X_train);
[N_test, ~] = size(X_test);
U_train = X_train * Uo;

% Manifold optimization problem
manifold = stiefelfactory(d, m);
problem.M = manifold;
problem.costgrad = @(A) cost(A, X_train, X_test, f_train, f_test);
[Uopt, xcost, info, options] = conjugategradient(problem, Uo);

%% Plotting the optimized solution! Woot!
close all;

vopt_test = X_test * Uopt;
vopt_train = X_train * Uopt;
vreal = X_test * Ureal;
vtrain = X_train * Ureal;

figure1 = figure;
set(gca, 'FontSize', 18, 'LineWidth', 2); hold on; box on; grid on;
plot3(vreal(:,1), vreal(:,2), f_test, 'x', 'DisplayName', 'Training', 'MarkerSize', 14, 'LineWidth', 2);
plot3(vtrain(:,1), vtrain(:,2), f_train, 's', 'DisplayName', 'Testing', 'MarkerSize', 14, 'LineWidth', 2);
xlabel('$\mathbf{u}_{1, true}$', 'Interpreter', 'Latex');
ylabel('$\mathbf{u}_{2, true}$', 'Interpreter', 'Latex');
zlabel('Function, $f(\mathbf{U}_{true}^{T} \mathbf{x} )$','Interpreter', 'Latex');
legend show;
hold off;

figure2 = figure;
set(gca, 'FontSize', 18, 'LineWidth', 2); hold on; box on; grid on;
plot3(vopt_train(:,1), vopt_train(:,2), f_train, 'x', 'DisplayName', 'Optimized training', 'MarkerSize', 14, 'LineWidth', 2);
plot3(vopt_test(:,1), vopt_test(:,2), f_test, 's', 'DisplayName', 'Optimized testing', 'MarkerSize', 14, 'LineWidth', 2);
xlabel('$\mathbf{u}_{1, opt}$', 'Interpreter', 'Latex');
ylabel('$\mathbf{u}_{2, opt}$', 'Interpreter', 'Latex');
zlabel('Function, $f(\mathbf{U}_{opt}^{T} \mathbf{x} )$','Interpreter', 'Latex');
legend show;
hold off;

% Plotting the convergence trajectory! Woot!
figure3 = figure;
set(gca, 'FontSize', 18, 'LineWidth', 2, 'YScale','log'); hold on; box on; grid on;
semilogy([info.iter], [info.gradnorm], '.-', 'LineWidth', 3); 
xlabel('Iteration number'); ylabel('Norm of the gradient of $r$', 'Interpreter', 'Latex');
hold off;

figure4 = figure;
set(gca, 'FontSize', 18, 'LineWidth', 2, 'YScale','log'); hold on; box on; grid on;
semilogy([info.iter], [info.cost], '.-', 'LineWidth', 3); 
xlabel('Iteration number'); ylabel('Objective function $r$', 'Interpreter', 'Latex');
hold off;

%% Cost function and its gradient!
function [r, dr] = cost(A, X_train, X_test, f_train, f_test)
[N, d] = size(X_train);
[N_test, ~] = size(X_test);
U_train = X_train * A;
U_test = X_test * A;
hyp=struct('mean',[],'cov',[0.0 0.0 0.0],'lik',0.);
meanfunc=[];
covfunc=@covSEard;
likfunc=@likGauss;
hyp2 = minimize(hyp, @gp, -300, @infGaussLik, meanfunc, covfunc, likfunc, U_train, f_train);
ell_1 = exp(hyp2.cov(1));
ell_2 = exp(hyp2.cov(2));
sf = exp(hyp2.cov(3));
sn = exp(hyp2.lik);
P = diag([ell_1^2 , ell_2^2]);
[~, m] = size(A);
inv_P = inv(P);
K = zeros(N,N);
for i = 1 : N
    for j = 1 : N
        x = U_train(i,:);
        z = U_train(j,:);
        K(i,j) = sf^2 * exp(-(x-z)* inv_P *(x-z)'/2);
    end
end
G = K + sn * eye(N);
Ue = chol(G);
inv_G = inv(Ue) * inv(Ue)';
b = inv_G * f_train;
K_test = zeros(N_test,N);
for i = 1 : N_test
    for j = 1 : N
        x = U_test(i,:);
        z = U_train(j,:);
        K_test(i,j) = sf^2 * exp(-(x-z)*inv_P*(x-z)'/2);
    end
end
g_test = K_test * b ;
r = 0.5 * norm(f_test - g_test', 2)^2;
dr = zeros(d,m);
for i = 1 : N_test
    for j = 1 : m
        U_tilde(:,j) = U_test(i, j) - U_train(:,j);
    end
    dy = -(-inv_P * U_tilde' * (K_test(i,:)' .* b) * X_test(i,:) )';
    dr = dr +   ( (f_test(i) - g_test(i) ) * (dy - A * dy' * A) );
end
end


