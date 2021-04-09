clear; clc; close all;
addpath('./funcs/');
load('vars/BBP.mat', 'ss');
ss = c2d(ss,.1); % discretization
clear functions;

%% State space
A = ss.A;
B = ss.B;
C = ss.C;
Cd = 1*eye(2);
Bd = 0.1*[0.3, 0.5; 0.3, 0.5; 0.5, 1; 1, 0.5; 0.4, 0.36; 0.54, 0.59; 0.56, 0.48; 0.38, 0.62];

% define dimensions
nx = size(A,1);
ny = size(C,1);
nu = size(B,2);
nd = 2;
N = 20;

% tuning variables
Q_gain = 0.3; 
R_gain = 0.11;
x0 = [0,0,0,0,0,0,0,0]';

Q = Q_gain*eye(nx);   
R = R_gain*eye(nu);

% observer gain
L1 = [0.1, 0.05; 0.05, 0.1];          
[P,~,~] = dare(A,B,Q,R);


% augmented state space
Ae = [A, Bd; zeros(nd, nx), eye(nd)];
Be = [B; zeros(nd,nu)];
Ce = [C, Cd];

Qe = blkdiag(Q, zeros(nd));           
Re = R;                                  
Pe = blkdiag(P, zeros(nd)); 
nxe = nx + nd;

[Te,Se] = predict_model(Ae,Be,nxe,nu,N);  
[He,he] = cost_model(Se,Te,nxe,nu,N,Qe,Pe,Re); 


%% simulation time
time = 30;
dt = 0.2; % timestep
T = linspace(0, time, time/dt);
T_1 = [T(1,:),time+dt];
t = length(T);

%% disturbance settings
d = zeros(2, t);

z = 0.2;
f = 2.1;

for i = 1:t
    if i >= 60 && i <= 70
       d(1,i) = 0.005;
       d(2,i) = -0.005;
    elseif i >= 120 && i <= 130
       d(1,i) = 0.005;
       d(2,i) = -0.005;
    end
end

% set trajectory
r = 0.1;
w = 0.2;
c = 0.002;

[xbref,ybref] = ref(r,w,c,t);

y_ref = [xbref;ybref];

% derive optimal inputs
x = zeros(nx,t+1);
x(:,1) = x0;
u_rec = zeros(nu,t);
y = zeros(ny,t);

dhat = zeros(nd,t+1);
xhat = zeros(nx,t+1);
dhat(:,1) = [0;0];
xhat(:,1) = [0;0;0;0;0;0;0;0];

[~,~, Lgain] = idare(Ae', Ce', Qe, R_gain*eye(2));
L = place(Ae', Ce', Lgain)'; %observer gain
Lx = L(1:nx,:);
Ld = L(nx+1:end,:);

for k=1:t
    y(:,k) = C*x(:,k) + Cd*d(:,k);
    
    Ac = [eye(nx)-A,-B; 
          C,zeros(ny,nu)];
    bc = [Bd*dhat(:,k); 
          f*y_ref(:,k)-Cd*dhat(:,k)];

    H0 = blkdiag(zeros(nx),eye(nu));
    h0 = zeros(nx+nu,1);
    xur = quadprog(H0,h0,[],[],Ac,bc,[],[],[]);
    xr = xur(1:nx);
    ur = xur(nx+1:end);

    u_con = sdpvar(nu*N,1);   % define optimization variable
    x_con = sdpvar(length(x(:,1)),1);

    %Constraint=[abs(u_con)<=1.0,... 
                %abs(x_con(1))<=.15, abs(x_con(2))<=2, abs(x_con(3))<=.15,...
                %abs(x_con(4))<=2, abs(x_con(5))<=pi/4,abs(x_con(6))<=3,...
                %abs(x_con(7))<=pi/4, abs(x_con(8))<=3];  % add contraints later

    Constraint=[];
    Objective = 0.5*u_con'*He*u_con+(he*[x(:,k); d(:,k); xr; dhat(:,k); ur])'*u_con;
    optimize(Constraint,Objective);
    u_con = value(u_con);


    % Select the first input only
    u_rec(:,k) = u_con(1:nu);

    % Compute the state/output evolution
    x(:,k+1) = A*x(:,k) + B*u_rec(:,k) + Bd*d(:,k);

    clear u_con

    % Update disturbance estimation
    xhat(:,k+1) = A*xhat(:,k)+Bd*dhat(:,k)+B*u_rec(:,k)+Lx*(y(:,k)-C*xhat(:,k)-Cd*dhat(:,k));
    dhat(:,k+1) = dhat(:,k)+L1*(y(:,k)-C*xhat(:,k)-Cd*dhat(:,k));
end   

% plot values
figure(1)
plot(T,y(1,:),'b','LineWidth', 1.3);
hold on
plot(T,y_ref(1,:),'r','LineWidth', 1.3);
plot(T,d(1,:),'g','LineWidth', 1.3);
hold off

legend('x_b', 'x_bref', 'disturbance');

figure(2)
plot(T,y(2,:),'b','LineWidth', 1.3);
hold on
plot(T,y_ref(2,:),'r','LineWidth', 1.3);
plot(T,d(2,:),'g','LineWidth', 1.3);
hold off

legend('y_b', 'y_bref', 'disturbance');

figure(3)
plot(y_ref(1,1:136),y_ref(2,1:136),'r','LineWidth', 1.3);
hold on
plot(y(1,:),y(2,:),'b','LineWidth', 1.3);
hold off

legend('reference trajectory', 'actual trajectory');

%% functions

function [xbref,ybref] = ref(r,w,c,t)
    theta = zeros(1,t);
    xbref = zeros(1,t);
    ybref = zeros(1,t);
    
    for k = 1:t
        theta(1,k) = w*k+c;
        xbref(1,k) = cos(theta(1,k))*r*(k/t);
        ybref(1,k) = sin(theta(1,k))*r*(k/t);
    end
end

function [T,S] = predict_model(A,B,nx,nu,N)
    % T matrix from initial state
    T = zeros(nx*(N+1),nx);
    for k = 0:N
        T(k*nx+1:(k+1)*nx,:) = A^k;
    end

    % S matrix from input
    S = zeros(nx*(N+1),nu*N);
    for k = 1:N
        for i = 0:k-1
            S(k*nx+1:(k+1)*nx,i*nu+1:(i+1)*nu) = A^(k-1-i)*B;
        end
    end
end

function [H,h] = cost_model(S,T,nx,nu,N,Q,P,R)
    Qbar = blkdiag(kron(eye(N),Q),P);
    Rbar = kron(eye(N),R);
    H = S'*Qbar*S+Rbar;   
    hx0 = S'*Qbar*T;
    hxref = -S'*Qbar*kron(ones(N+1,1),eye(nx));
    huref = -Rbar*kron(ones(N,1),eye(nu));
    h = [hx0 hxref huref];
end