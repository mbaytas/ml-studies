clear
clc
close all

%% Part A : Load & plot

% Read data from file
rawData  = csvread('input.txt');

% Plot data
x = rawData(:, 1);
y = rawData(:, 2);
s = 20;
c = rawData(:, 3);
fig1 = scatter(x, y, s, c, 'fill');
saveas(fig1, 'img/Fig1.jpg');