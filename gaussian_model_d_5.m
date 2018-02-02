%% 0. My first attempt at dimension reduction using manifold optimization!
clear; clc; close all;
m = 5; n = 1;
designA;
counter = 1;
success = 0;
fail = 0;
%% 1. Split the original data set into a training and testing module!
for i = 1 : 25
    X_training = rand(200, m).*2 - 1; % set of random points betwee [-1,1]
    X_testing = rand(200, m).*2 - 1;
    mu = 0.0;
    SIGMA = 0.07;
    g_Ax_training = mvnpdf(X_training * A,mu,SIGMA); 
    g_Ax_testing = mvnpdf(X_testing * A,mu,SIGMA); 
    f_train = g_Ax_training;
    f_test = g_Ax_testing;
    E = rand(m,n);
    [Q, R] = qr(E);
    Uo = Q(:,1:n);
    [Uopt, xcost, info, options] = optimization(X_testing, X_training, f_test, f_train, Uo);
    counting_index = [info.iter];
    if max(counting_index) > 1
        iters{counter} = [info.iter];
        gradnorms{counter} = [info.gradnorm];
        cost{counter} = [info.cost];
        Ubest{counter} = Uopt;
        f_training{counter} = f_train;
        f_testing{counter} = f_test;
        X_training_saved{counter} = X_training;
        X_testing_saved{counter} = X_testing;
        counter = counter + 1;
        success = success + 1;
    else
        fail = fail + 1;
    end
end
Uopt = fliplr(Uopt);
%% 3. Now that we have the optimal design, plot!
close all;
C = jet(counter-1);
figure4 = figure;
set(gca, 'FontSize', 16, 'LineWidth', 2); hold on; box on; grid on;
for j = 1 : counter - 1
    plot(iters{j}, log10(cost{j}), '.-', 'LineWidth', 2, 'Color', C(j,:) );
end
xlabel('Iteration', 'Interpreter', 'Latex'); ylabel('Residual (log), $r$', 'Interpreter', 'Latex');
hold off;
%%
figure5 = figure;
set(gca, 'FontSize', 16, 'LineWidth', 2); hold on; box on; grid on;
for j = 1 : counter - 1
   x =  [X_training_saved{j} * Ubest{j} ; X_testing_saved{j} * Ubest{j} ];
   y = [f_training{j}; f_testing{j}];
   plot(x, y, 'o', 'LineWidth', 1, 'Color', C(j,:) );
end
xlabel('$\mathbf{u}$', 'Interpreter', 'Latex'); ylabel('$f$', 'Interpreter', 'Latex');