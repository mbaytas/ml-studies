clear
clc
close all

%% Part D : SVMs

% Add path to compiled libsvm toolbox
addpath('libsvm');

% Read data from file
rawData  = csvread('input.txt');
% Convert data to libsvm format
rawLabels   = rawData(:, 3);
rawFeatures = rawData(:, 1:2);
features_sparse = sparse(rawFeatures);
libsvmwrite('input.train', rawLabels, features_sparse);
% Read data with libsvm
[labels, features] = libsvmread('input.train');

% Initialize settings for grid search
stepSize = 1;
log2cList = -1:stepSize:10;
log2gList = -10:stepSize:1;
Nlog2c = length(log2cList);
Nlog2g = length(log2gList);
heat = zeros(Nlog2c,Nlog2g); % Init heatmap matrix
bestAccuracy = 0; % Var to store best accuracy
% To see how things go as grid search runs
totalRuns = Nlog2c*Nlog2g;

%% Linear kernel
% Grid search to optimize cost & gamma
runCounter = 1;
for i = 1:Nlog2c
    for j = 1:Nlog2g
        log2c = log2cList(i);
        log2g = log2gList(j);
        disp([num2str(runCounter), '/', num2str(totalRuns)]);
        disp(['Trying c=', num2str(2^log2c), ' and g=', num2str(2^log2g)]);
        % Train with current cost & gamma
        params = ['-q -t 1 -v 5 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
        accuracy = svmtrain(labels, features, params);
        % Update heatmap matrix
        heat(i,j) = accuracy;
        % Update accuracy, cost & gamma if better
        if (accuracy >= bestAccuracy)
            bestAccuracy = accuracy;
            bestC = 2^log2c;
            bestG = 2^log2g;
        end
        runCounter = runCounter+1;
    end
end
% Return heatmap (more accurately, colormap)
hm = figure;
imagesc(heat);
colormap('jet'); 
colorbar;
set(gca,'XTick',1:Nlog2g);
set(gca,'XTickLabel',sprintf('%3.1f|',log2gList));
xlabel('Log_2\gamma');
set(gca,'YTick',1:Nlog2c);
set(gca,'YTickLabel',sprintf('%3.1f|',log2cList));
ylabel('Log_2c');
title('Grid Search over c and \gamma for polynomial kernel');
saveas(hm, 'img/heatPolynomial.jpg');

% Train and test SVM
% Train model using specified kernel and best cost & gamma values
params = ['-t 1 -v 5 -c ', num2str(bestC), ' -g ', num2str(bestG)];
accuracy = svmtrain(labels, features, params);
disp('Polynomial: done!');
