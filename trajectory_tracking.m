%% State space
% Constants:
g = 9.81;
mb = 0.05;
Ip = 5.12;
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
L=[0.5 0; 0 0.5];          
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
d = zeros(2, t);
nd = 2;

z = 0.2;

for i = 1:t
    if i >= 10 && i <= 15
       d(1,i) = 0.01*z;
       d(2,i) = -0.01*z;
    elseif i >= 20 && i <= 25
       d(1,i) = -0.01*z;
       d(2,i) = 0.01*z;
    elseif i >= 30 && i <= 35
       d(1,i) = 0.01*z;
       d(2,i) = 0.01*z;
    elseif i >= 40 && i <= 45
       d(1,i) = -0.01*z;
       d(2,i) = -0.01*z;
    elseif i >= 50 && i <= 55
       d(1,i) = -0.015*z;
       d(2,i) = 0.015*z;
    elseif i >= 60 && i <= 65
       d(1,i) = 0.01*z;
       d(2,i) = -0.01*z;
    elseif i >= 70 && i <= 75
       d(1,i) = -0.01*z;
       d(2,i) = 0.01*z;
    elseif i >= 80 && i <= 85
       d(1,i) = -0.01*z;
       d(2,i) = -0.01*z;
    elseif i >= 85 && i <= 90
       d(1,i) = 0.01*z;
       d(2,i) = 0.01*z;
    end
end

% set trajectory
r = 0.1;
w = 0.2;
c = 0.002;

[xbref,ybref] = ref(r,w,c,t);

y_ref = [xbref;ybref];
        
[x_t,y_t] = ref_tracker(x0,t,A,B,C,Cd,d,y_ref,nx,nu,nd,ny,N,H,h,L);

%% plot values
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

figure(3)
plot(y_ref(1,:),y_ref(2,:),'b','LineWidth', 1.3);
hold on
plot(y_t(1,:),y_t(2,:),'r','LineWidth', 1.3);
hold off

legend('actual trajectory', 'reference trajectory');
%% Functions
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

function [x,y] = ref_tracker(x0,t,A,B,C,Cd,d,yref,nx,nu,nd,ny,N,H,h,L)
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
        Constraint=[];  % add contraints later
        Objective = 0.5*u_opt'*H*u_opt+(h*[x(:,k); xr; ur])'*u_opt;
        optimize(Constraint,Objective);
        u_opt = value(u_opt);

        % Select the first input only
        u_rec=zeros(nu,t);
        u_rec(:,k) = u_opt(1:nu);

        % Compute the state/output evolution
        x(:,k+1) = A*x(:,k) + B*u_rec(:,k);
        y(:,k) = C*x(:,k+1) + Cd*d(:,k); % + 0.001*randn(ny,1); noise
        clear u_uncon

        % Update disturbance estimation
        dhat(:,k+1)=dhat(:,k)+L*(y(:,k)-C*x(:,k+1)-dhat(:,k));
    end
end