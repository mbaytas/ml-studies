clear
clc
close all

%% Part C : If/Else Classifier

% Read data from file
rawData  = csvread('input.txt');
x = rawData(:, 1);
y = rawData(:, 2);
c = rawData(:, 3);

% Init guesses
[N, D] = size(rawData);
guesses = ones(N, 1);

% Guess
for i = 1:N
    if y(i) > 0.5
        if x(i) < 4
            if y(i) < 7
                guesses(i) = 0;
            end
        end
    end
end

% Analyze
corrects = 0;
for i = 1:N
    if (c(i) == guesses(i))
        corrects = corrects + 1;
    end
end

accuracy = corrects / N;

% Plot guesses
s = 20;
fig3 = scatter(x, y, s, guesses, 'fill');
saveas(fig3, 'img/Fig3.jpg');

% Confusion matrix
[C,order] = confusionmat(c, guesses)