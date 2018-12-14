clear; close all; clc;
warning('off','all')
%==========================================================================
%
% A simple application of Manifold optimization to estimate a Gaussian
% ridge function
%
% Pranay Seshadri,
% The Alan Turing Institute
% December 14th, 2018
%==========================================================================
% Preliminaries
N = 100;
N_test = 100;
H = N + N_test + 1;
d = 5; m = 2;
test_train = randperm(H);
train = test_train(1:N);
test = test_train(N + 1:N + N_test);
X = randn(H, d);
X_train = X(train, :);
X_test = X(test, :);
Ureal = rand(d, m);
[Q,~] = qr(Ureal); Ureal = Q(:,1:2);
U_train = X_train * Ureal;
U_test = X_test * Ureal;
f_train = zeros(N, 1) ; %+ 0.001 * randn(N, 1);
MU = [0.0 0.0];
SIGMA = [0.25 0.3; 0.3 1];
for i = 1 : N
    u = U_train(i,:);
    f_train(i) = mvnpdf([u(1), u(2)], MU, SIGMA) ;
end
for i = 1 : N_test
    u = U_test(i,:);
    f_test(i) =  mvnpdf([u(1), u(2)], MU, SIGMA);
end

%% Optimization test!
repeats = 1;
for i = 1 : repeats
    Uo = randn(d,d); [Q, ~] = qr(Uo); Uo = Q(:,1:2);
    [N, d] = size(X_train);
    [N_test, ~] = size(X_test);
    U_train = X_train * Uo;

    % Manifold optimization problem
    manifold = stiefelfactory(d, m);
    problem.M = manifold;
    problem.costgrad = @(A) cost(A, X_train, X_test, f_train, f_test);
    options.maxiter = 20;
    options.verbosity = 2;
    [Uopt, xcost, info, options] = conjugategradient(problem, Uo, options);
    xcost_store(i) = xcost;
    iters_store{i} = [info.iter];
    cost_store{i} = [info.cost];
    grad_store{i} = [info.gradnorm];
end
%% Plotting the optimized solution! Woot!
close all;

vopt_test = X_test * Uopt;
vopt_train = X_train * Uopt;

vreal = X_test * Ureal;
vtrain = X_train * Ureal;

%%
newaxis1=X_train*Uopt(:,1);
newaxis2=X_train*Uopt(:,2);
N=100;
[meshX,meshY]=meshgrid([min(newaxis1)-1:(max(newaxis1)-min(newaxis1)+2)/N:max(newaxis1)+1],[min(newaxis2)-2:(max(newaxis2)-min(newaxis2)+4)/N:max(newaxis2)+2]);
xs=zeros((N+1)^2,2);
for i=1:N+1
    for j=1:N+1
        xs(j+i*N+i-N-1,:)=[meshX(i,j),meshY(i,j)];
    end
end

%set the basis
meanfunc=[];
covfunc=@covSEard;
likfunc=@likGauss;

%train the GPR model
hyp=struct('mean',[],'cov',[0 0 0],'lik',0);
hyp2 = minimize(hyp, @gp, -300, @infGaussLik, meanfunc, covfunc, likfunc, [newaxis1,newaxis2], f_train);
[ymu,ys2] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, [newaxis1,newaxis2], f_train, xs);
ym = reshape(ymu, [N+1, N+1]);
%%
ymu_real = mvnpdf(xs, MU, SIGMA);
ymu_real = reshape(ymu_real, [N+1, N+1]);
%%
close all;

figure1 = figure;
axes1 = axes('Parent',figure1);
set(axes1, 'FontSize', 18, 'LineWidth', 2); hold on; box on; grid on;
surf(meshX,meshY,ym', 'DisplayName', 'Gaussian ridge') ; shading interp;
plot3(vopt_train(:,1), vopt_train(:,2), f_train, 'x', 'DisplayName', 'Training', 'MarkerSize', 8, 'LineWidth', 2);
plot3(vopt_test(:,1), vopt_test(:,2), f_test, 's', 'DisplayName', 'Testing', 'MarkerSize', 8, 'LineWidth', 2);
xlim([-3 3]); ylim([-3 3]); zlim([-0.1 0.4]);
legend1 = legend(axes1,'show');
set(legend1,'EdgeColor',[1 1 1], 'Interpreter', 'Latex', ...
    'Position',[0.170490777151925 0.690904751278105 0.184509222848075 0.150761915388561]);
% Create colorbar
colorbar('peer',axes1);
view([-22 10]);
caxis([0 0.4]);
print('D1.png', '-dpng', '-r400');

figure2 = figure;
axes2 = axes('Parent',figure2);
set(axes2, 'FontSize', 18, 'LineWidth', 2); hold on; box on; grid on;
surf(meshX,meshY,ymu_real', 'DisplayName', 'Function') ; shading interp;
plot3(vtrain(:,1), vtrain(:,2), f_train, 'x', 'DisplayName', 'Training', 'MarkerSize', 8, 'LineWidth', 2);
plot3(vreal(:,1), vreal(:,2), f_test, 's', 'DisplayName', 'Testing', 'MarkerSize', 8, 'LineWidth', 2);
xlim([-3 3]); ylim([-3 3]);zlim([-0.1 0.4]);
legend2 = legend(axes2,'show');
set(legend2,'EdgeColor',[1 1 1], 'Interpreter', 'Latex', ...
    'Position',[0.170490777151925 0.690904751278105 0.184509222848075 0.150761915388561]);
view([-22 10]);
caxis([0 0.4]);
colorbar('peer',axes2);
print('D2.png', '-dpng', '-r400');
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


