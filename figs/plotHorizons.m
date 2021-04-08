clear; clc;

x10 = load('../vars/xN10.mat', 'x');
x12 = load('../vars/xN12.mat', 'x');
x13 = load('../vars/xN13.mat', 'x');
x15 = load('../vars/xN15.mat', 'x');
x20 = load('../vars/xN20.mat', 'x');
t   = load('../vars/T_1.mat', 'T_1');

%% plot

figure(3)
clf;
hold on; grid on;
stairs(t.T_1, x10.x(1,:), 'LineWidth', 1.3);
stairs(t.T_1, x12.x(1,:), 'LineWidth', 1.3);
stairs(t.T_1, x13.x(1,:), 'LineWidth', 1.3);
stairs(t.T_1, x15.x(1,:), 'LineWidth', 1.3);
stairs(t.T_1, x20.x(1,:), 'LineWidth', 1.3);
legend('N = 10','N=12','N=13', 'N = 15','N = 20');