clear; clc;
%% State space
% Constants:
g = 9.81;
mb = 0.05;
Ip = 0.12;
l = 0.3;

% Functions:
a = (3/5)*g;
b = -(mb*g)/Ip;
c = l/(2*Ip);

% State space:
A = [0,1,0,0,0,0,0,0;
     0,0,0,0,a,0,0,0;
     0,0,0,1,0,0,0,0;
     0,0,0,0,0,0,a,0;
     0,0,0,0,0,1,0,0;
     b,0,0,0,1,0,0,0;
     0,0,0,0,0,0,0,1;
     0,0,b,0,0,0,1,0];
 
B = [0,0,0,0;
     0,0,0,0;
     0,0,0,0;
     0,0,0,0;
     0,0,0,0;
     -c,0,c,0;
     0,0,0,0;
     0,-c,0,c];
  
C = [1,0,0,0,0,0,0,0;
     0,0,1,0,0,0,0,0];
   
D = 0;
   
%BBP = ss(A,B,C,D);
   
% Stability check;
%eig(A) % for stable MPC control all eig(A) < 1
   
% Controlability test:
%rank(A)
%Cont = ctrb(A,B);
%rank(Cont)

% define dimensions
nx = size(A,1);
ny = size(C,1);
nu = size(B,2);
N = 5;

% tuning variables
Q_gain = 3; 
R_gain = 1;
x0 = [0,0,0,0,0,0,0,0]';

Q = Q_gain*eye(nx);   
R = R_gain*eye(nu);

% observer gain
L=[1, 0.6; 0.3, 1];          
P = dare(A,B,Q,R);  

[T,S] = predict_model(A,B,nx,nu,N);
[H,h] = cost_model(S,T,nx,nu,N,Q,P,R);

% simulation
time = 10;
dt = 0.1; % timestep
T = linspace(0, time/dt-1, time/dt);
t = length(T);

% disturbance settings
Cd = eye(2);
% d = [0.003; 0.003]; % +/- 3mm distortion
d = zeros(2, t);
nd = 2;

for i = 1:t
    if i >= 20 && i <= 30
       d(1,i) = 0.02;
       d(2,i) = -0.03;
    elseif i >= 50 && i <= 65
       d(1,i) = -0.03;
       d(2,i) = 0.02;
    end
end

% some reference trajactory
y_ref = zeros(2, t);

for i = 1:t
   if i >= 40
       y_ref(1,i) = 0.1;
       y_ref(2,i) = -0.1;
   end
end
        
[x_t,y_t] = ref_tracker(x0,t,A,B,C,Cd,d,y_ref,nx,nu,nd,ny,N,H,h,L,P);

% plot values
figure(1)
plot(T,y_t(1,:),'b','LineWidth', 1.3);
hold on
plot(T,y_ref(1,:),'r','LineWidth', 1.3);
plot(T,d(1,:),'g','LineWidth', 1.3);
hold off

legend('x_b', 'x_bref', 'disturbance');

figure(2)
plot(T,y_t(2,:),'b','LineWidth', 1.3);
hold on
plot(T,y_ref(2,:),'r','LineWidth', 1.3);
plot(T,d(2,:),'g','LineWidth', 1.3);
hold off

legend('x_b', 'x_bref', 'disturbance');

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

function [x,y] = ref_tracker(x0,t,A,B,C,Cd,d,yref,nx,nu,nd,ny,N,H,h,L,Pdare)
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
              yref(:,k)-Cd*dhat(:,k)];

        % find ur and xr:
        options1 = optimset('quadprog'); 
        options1.OptimalityTolerance = 1e-20;
        options1.ConstraintTolerance = 1.0000e-15;
        options1.Display='none';

        H0 = blkdiag(zeros(nx),eye(nu));
        h0 = zeros(nx+nu,1);
        xur = quadprog(H0,h0,[],[],Ac,bc,[],[],[],options1);
        xr = xur(1:nx);
        ur = xur(nx+1:end);

        u_opt = sdpvar(nu*N,1);
        x_opt = sdpvar(8,1);
        
        Constraint = [x(:,k)'*Pdare*x(:,k)<=5.56 ,abs(u_opt)<=2.5,... 
                      abs(x_opt(1))<=.15, abs(x_opt(2))<=2, abs(x_opt(3))<=.15,...
                      abs(x_opt(4))<=2, abs(x_opt(5))<=pi/4,abs(x_opt(6))<=3,...
                      abs(x_opt(7))<=pi/4, abs(x_opt(8))<=3]; %define constraints
        
        Objective = 0.5*u_opt'*H*u_opt+(h*[x(:,k); xr; ur])'*u_opt;
        optimize(Constraint,Objective);
        
        u_opt = value(u_opt);
        x_opt = value(x_opt);

        % Select the first input only
        u_rec=zeros(nu,t);
        u_rec(:,k) = u_opt(1:nu);
        
        % save optimzed x
        x_rec = x_opt;

        % Compute the state/output evolution
        x(:,k+1) = A*x(:,k) + B*u_rec(:,k);
        y(:,k) = C*x(:,k+1) + Cd*d(:,k); % + 0.001*randn(ny,1); noise
        clear u_uncon

        % Update disturbance estimation
        dhat(:,k+1)=dhat(:,k)+L*(y(:,k)-C*x(:,k+1)-dhat(:,k));
    end
end