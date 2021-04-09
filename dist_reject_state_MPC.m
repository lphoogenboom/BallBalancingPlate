clear; clc; close all; clear functions;
addpath('./funcs/');
load('vars/BBP.mat', 'ss');
ss = c2d(ss,.1); % discretization
clear functions;

%% State space
A = ss.A;
B = ss.B;
C = ss.C;

% define dimensions
nx = size(A,1);
ny = size(C,1);
nu = size(B,2);
nd = 2;
N = 15;

% tuning variables
Q_gain = 3; 
R_gain = 1;
x0 = [0,0,0,0,0,0,0,0]';

Q = Q_gain*eye(nx);   
R = R_gain*eye(nu);

% observer gain
L = [0.8, 0.2; 0.2, 0.8];          
[P,~,~] = dare(A,B,Q,R);

[T,S] = predict_model(A,B,nx,nu,N);
[H,h] = cost_model(S,T,nx,nu,N,Q,P,R);

%% simulation time
time = 30;
dt = 0.2; % timestep
T = linspace(0, time, time/dt);
T_1 = [T(1,:),time+dt];
t = length(T);

%% disturbance settings
d = zeros(2, t);

for i = 1:t
    if i >= 20 && i <= 30
       d(1,i) = 0.007;
       d(2,i) = -0.009;
    elseif i >= 100
       d(1,i) = -0.009;
       d(2,i) = 0.007;
    end
end

% some reference trajactory
y_ref = zeros(2, t);
% f = 1.5;

for i = 1:t
   if i >= 70
       y_ref(1,i) = 0.05;
       y_ref(2,i) = -0.05;
   end
end
        
%% state feedback disturbance rejection MPC

% derive optimal inputs
x = zeros(nx,t+1);
x(:,1) = x0;
u_rec = zeros(nu,t);
y = zeros(ny,t);

dhat = zeros(nd,t+1);
dhat(:,1) = [0;0];

for k=1:t
    Ac = [eye(nx)-A,-B; 
          C,zeros(ny,nu)];
    bc = [zeros(nx,1); 
          y_ref(:,k)-dhat(:,k)]; %Cd*dhat(:,k)];

    options1 = optimset('quadprog'); 
    options1.OptimalityTolerance = 1e-20;
    options1.ConstraintTolerance = 1.0000e-15;
    options1.Display='none';

    H0 = blkdiag(zeros(nx),eye(nu));
    h0 = zeros(nx+nu,1);
    xur = quadprog(H0,h0,[],[],Ac,bc,[],[],[],options1);
    xr = xur(1:nx);
    ur = xur(nx+1:end);

    u_con = sdpvar(nu*N,1);   % define optimization variable
    x_con = sdpvar(length(x(:,1)),1);

    %Constraint=[abs(u_con)<=1.0,... 
                %abs(x_con(1))<=.15, abs(x_con(2))<=2, abs(x_con(3))<=.15,...
                %abs(x_con(4))<=2, abs(x_con(5))<=pi/4,abs(x_con(6))<=3,...
                %abs(x_con(7))<=pi/4, abs(x_con(8))<=3];  % add contraints later

    Constraint=[];
    Objective = 0.5*u_con'*H*u_con+(h*[x(:,k); xr; ur])'*u_con;
    optimize(Constraint,Objective);
    u_con = value(u_con);


    % Select the first input only
    u_rec(:,k) = u_con(1:nu);

    % Compute the state/output evolution
    x(:,k+1) = A*x(:,k) + B*u_rec(:,k);
    y(:,k) = C*x(:,k+1) + d(:,k); %Cd*d(:,k); % + 0.001*randn(ny,1); noise

    clear u_con

    % Update disturbance estimation
    dhat(:,k+1)=dhat(:,k)+L*(y(:,k)-C*x(:,k+1)-dhat(:,k));
end   

        
%% plot values
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

%% Functions
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

function [x,y] = ref_tracker(x0,t,A,B,C,Cd,d,yref,nx,nu,nd,ny,N,H,h,L,P)
    % derive optimal inputs
    x = zeros(nx,t+1);
    x(:,1) = x0;
    
    y = zeros(ny,t);
    
    dhat = zeros(nd,t+1);
    dhat(:,1) = [0;0];
    
    for k=1:t
        Ac = [eye(nx)-A,-B; 
              C,zeros(ny,nu)];
        bc = [zeros(nx,1); 
              yref(:,k)-dhat(:,k)]; %Cd*dhat(:,k)];

        options1 = optimset('quadprog'); 
        options1.OptimalityTolerance = 1e-20;
        options1.ConstraintTolerance = 1.0000e-15;
        options1.Display='none';

        H0 = blkdiag(zeros(nx),eye(nu));
        h0 = zeros(nx+nu,1);
        xur = quadprog(H0,h0,[],[],Ac,bc,[],[],[],options1);
        xr = xur(1:nx);
        ur = xur(nx+1:end);

        u_con = sdpvar(nu*N,1);   % define optimization variable
        x_con = sdpvar(length(x(:,1)),1);
        
        Constraint=[abs(u_con)<=1.0,... 
                    abs(x_con(1))<=.15, abs(x_con(2))<=2, abs(x_con(3))<=.15,...
                    abs(x_con(4))<=2, abs(x_con(5))<=pi/4,abs(x_con(6))<=3,...
                    abs(x_con(7))<=pi/4, abs(x_con(8))<=3];  % add contraints later
        
        %Constraint=[];
        Objective = 0.5*u_con'*H*u_con+(h*[x(:,k); xr; ur])'*u_con;
        optimize(Constraint,Objective);
        u_con = value(u_con);
        

        % Select the first input only
        u_rec = zeros(nu,t);
        u_rec(:,k) = u_con(1:nu);

        % Compute the state/output evolution
        x(:,k+1) = A*x(:,k) + B*u_rec(:,k);
        y(:,k) = C*x(:,k+1) + d(:,k); %Cd*d(:,k); % + 0.001*randn(ny,1); noise
        
        clear u_uncon
         
        % Update disturbance estimation
        dhat(:,k+1)=dhat(:,k)+L*(y(:,k)-C*x(:,k+1)-dhat(:,k));
    end
end