clear; close all; clc;
warning('off','all')
%==========================================================================
%
% Code for the Turbomachinery problem. You will need the turbomachinery data 
% set to run this code. Please get in touch with the author regarding that.
%
% Pranay Seshadri,
% The Alan Turing Institute
% December 14th, 2018
%==========================================================================
% Preliminaries
preliminaries;
d = 25; m = 2;
repeats = 20;
for i = 1 : repeats
    
    % Different random seeds!
    rng(randi(100));
    
    % Random start!
    M = randn(25,2);
    [Q, ~] = qr(M);
    Uo = Q(:,1:m);
    
    % Manifold optimization problem
    manifold = stiefelfactory(d, m);
    problem.M = manifold;
    problem.costgrad = @(A) cost(A, X_train, X_test, f_train, f_test);
    options.maxiter = 15;
    options.verbosity = 2;
    [Uopt, xcost, info, options] = conjugategradient(problem, Uo, options);
    xcost_store(i) = xcost;
    iters_store{i} = [info.iter];
    cost_store{i} = [info.cost];
    grad_store{i} = [info.gradnorm];
    uopt_store{i} = Uopt;
end

%% Plotting the optimized solution! Woot!
close all;
[~, cycles] = sort(xcost_store);
cycle = cycles(2);
Uopt = uopt_store{cycle};
vopt_test = X_test * Uopt;
vopt_train = X_train * Uopt;
v_both = X * Uopt;
f_both = f;
N=80;
newaxis1 = linspace(-1.5, 1.5, N)';
newaxis2=linspace(-1.5, 1.5, N)';
[meshX,meshY]=meshgrid([newaxis1, newaxis2]);
xs=zeros((N)^2,2);
counter = 1;
for i=1:N
    for j=1:N
        xs(counter, :)=[meshX(i,j),meshY(i,j)];
        counter = counter + 1;
    end
end

%set the basis
meanfunc=[];
covfunc=@covSEard;
likfunc=@likGauss;

%train the GPR model
hyp=struct('mean',[],'cov',[0 0 0],'lik',0);
hyp2 = minimize(hyp, @gp, -300, @infGaussLik, meanfunc, covfunc, likfunc, v_both, f);
[ymu,ys2] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, v_both, f, xs);
ym = reshape(ymu, [N, N]);
ys2 = reshape(ys2, [N, N]);

%%
close all;

figure1 = figure;
axes1 = axes('Parent',figure1);
set(axes1, 'FontSize', 18, 'LineWidth', 2); hold on; box on; grid on; colormap jet;
scatter3(X_test * Uopt(:,1), X_test * Uopt(:,2), f_test, 100, f_test, 'filled'); hold on;
scatter3(X_train * Uopt(:,1), X_train * Uopt(:,2), f_train, 100, f_train, 'filled');
plot3(X_test * Uopt(:,1), X_test * Uopt(:,2), f_test, 'ko', 'MarkerSize', 10, 'LineWidth', 0.5);
plot3(X_train * Uopt(:,1), X_train * Uopt(:,2), f_train, 'ko', 'MarkerSize', 10, 'LineWidth', 0.5);
s1=surf(newaxis1,newaxis2,ym'); s1.EdgeColor='none';s1.FaceAlpha=0.6;
s2=surf(newaxis1,newaxis2,sqrt(ys2')*1.96+ym','FaceColor',[0.4 0.4 0.4]);s2.EdgeColor='none';s2.FaceAlpha=0.3;
s3=surf(newaxis1,newaxis2,-sqrt(ys2')*1.96+ym','FaceColor',[0.4 0.4 0.4]);s3.EdgeColor='none';s3.FaceAlpha=0.3;
view([-161 24]); zlim([-1 0.5]);
xlim([-2 2]); ylim([-2 2]);
xlabel('$\mathbf{u}_{1}$', 'Interpreter', 'Latex');
ylabel('$\mathbf{u}_{2}$', 'Interpreter', 'Latex');
zlabel('Normalized efficiency', 'Interpreter', 'Latex');
print('T2.png', '-dpng', '-r400');
hold off;

figure2 = figure;
axes2 = axes('Parent',figure2);
set(axes2, 'FontSize', 18, 'LineWidth', 2); hold on; box on; grid on;
contourf(newaxis1,newaxis2, 1.96 * sqrt(ys2'),   'LineColor', 'w') ; shading interp;
caxis([0 0.3]);
colorbar('peer',axes2);
xlabel('$\mathbf{u}_{1}$', 'Interpreter', 'Latex');
ylabel('$\mathbf{u}_{2}$', 'Interpreter', 'Latex');
print('T3.png', '-dpng', '-r400');
hold off;

figure4 = figure;
set(gca, 'FontSize', 18, 'LineWidth', 2, 'YScale','log'); hold on; box on; grid on;
for i = 1 : repeats
    semilogy(iters_store{i}, cost_store{i} , '.-', 'LineWidth', 3); 
end
xlabel('Iteration number', 'Interpreter', 'Latex'); ylabel('Objective function', 'Interpreter', 'Latex');
print('T1.png', '-dpng', '-r400');
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
r = 0.5 * norm(f_test - g_test, 2)^2;
dr = zeros(d,m);
for i = 1 : N_test
    for j = 1 : m
        U_tilde(:,j) = U_test(i, j) - U_train(:,j);
    end
    dy = -(-inv_P * U_tilde' * (K_test(i,:)' .* b) * X_test(i,:) )';
    dr = dr +   ( (f_test(i) - g_test(i) ) * (dy - A * dy' * A) );
end
end


