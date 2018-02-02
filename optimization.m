function [Uopt, xcost, info, options] = optimization(X_test, X_train, f_test, f_train, Uo)
[m, n] = size(Uo);
manifold = grassmannfactory(m, n);
problem.M = manifold;
problem.costgrad = @(U) cost(U, X_train, f_train, X_test, f_test);
%checkgradient(problem);
%pause;
options.minstepsize = 1e-12; 
options.tolgradnorm = 1e-15;
options.maxiter=150;
%[Uopt, xcost, info, options] = steepestdescent(problem, Uo, options);
[Uopt, xcost, info, options] = conjugategradient(problem, Uo, options);
end


function [r, dR] = cost(U, X_train, f_train, X_test, f_test)
[m, n] = size(U);
Y_train = X_train * U; 
Y_test = X_test * U;
hyp=struct('mean',[],'cov',[0 0],'lik',0);
meanfunc=[];
covfunc=@covSEard;
likfunc=@likGauss;
gpLink = @infGaussLik;
hyp2 = minimize(hyp, @gp, -300, @infGaussLik, meanfunc, covfunc, likfunc, Y_train, f_train);
%hyp2.mean = [];
%hyp2.cov = [-1.9643 0.5586 -0.0625];
%hyp2.lik = -16.0231;
[gp_test, ~] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, Y_train, f_train, Y_test);
df = finite_difference_grads(hyp2, Y_train, f_train, Y_test);
dR = zeros(m,n);
[M, m] = size(X_test);
for i = 1 : M
  dR = dR +   ( (f_test(i) - gp_test(i) ) * ( X_test(i,:)'  * -df(i,:) ) ); %(-1 because of the sign change!)
end
dR = (eye(m) - U * U') * dR;
r = 0.5 * norm(f_test - gp_test, 2)^2;
end


function df = finite_difference_grads(hyp2, Y_train, f_train, Y_test)
meanfunc=[];
covfunc=@covSEard;
likfunc=@likGauss;
gpLink = @infGaussLik;
h = 1e-6;
[M, m] = size(Y_test);
XX = kron(ones(m+1, 1), Y_test) + h*kron([zeros(1, m); eye(m)], ones(M, 1)); % (M*(m+1))-by-m array of all inputs points 
[f, ~] = gp(hyp2, gpLink, meanfunc, covfunc, likfunc, Y_train, f_train, XX);
df = (reshape(f(M+1:end), M, m) - repmat(f(1:M), 1, m))/h;
end