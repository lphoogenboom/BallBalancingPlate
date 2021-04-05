clear;clc;

% World constants in BBP
%	g = 9.81;
%	mb = 0.05;
%	Ip = 0.12;
%	l = 0.3;
% 
% Functions used for ss:
%	a = (3/5)*g;
%	b = -(mb*g)/Ip;
%	c = l/(2*Ip);

load('BBP.mat', 'ss')

% Stability check;
%eig(A) % for stable MPC control all eig(A) < 1
   
% Controlability test:
% rank = rank(ss.A);
% Cont = ctrb(ss.A,ss.B);
% [V,D] = eig(ss.A);
% D = diag(D);

K = [];
for i=0:1
	K = [ K ss.A^i*ss.B];
end

[Ve,De] = eig(K); % IS NOT FULL RANK

% define dimensions
nx = size(ss.A,1);
ny = size(ss.C,1);
nu = size(ss.B,2);
N = 5;

%% LQR

Q = 1*eye(nx);
R = 2*eye(nu);

[P,L,G] = dare(ss.A,ss.B,Q,R); 

x0 = [0.2 -0.1 0.3 -0.2 0 0 0 0]';

t = 0:30; 
[x,u] = lqr(t,x0,nx,nu,ss.A,ss.B,G);

figure(1)

subplot(2,2,1)
plot(t,x(1,:));
xlabel('k');
ylabel('x_b');
grid minor;

subplot(2,2,2)
plot(t,x(2,:));
xlabel('k');
ylabel('vx_b');
grid minor;

subplot(2,2,3)
plot(t,x(3,:));
xlabel('k');
ylabel('y_b');
grid minor;

subplot(2,2,4)
plot(t,x(4,:));
xlabel('k');
ylabel('vy_b');
grid minor;

%% Functions

function [x,u] = lqr(t,x0,nx,nu,A,B,G)
    x = zeros(nx,length(t));   
    u = zeros(nu,length(t));
    x(:,1) = x0;

    for k = t(2:end)
        u(:,k) = -G*x(:,k);
        x(:,k+1) = A*x(:,k)+B*u(:,k);
    end
end