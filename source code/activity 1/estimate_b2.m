% --- Load both datasets ---
S1 = load('Group4_data1.mat');   % data used to identify b2
S2 = load('Group4_data2.mat');   % data used to validate b2

% --- Known physical parameters (group 4) ---
m1 = 1.17;
m2 = 0.85;
b1 = 0.63;
L1 = 1.25;
L2 = 0.77;

% ---------- IDENTIFICATION USING DATASET 1 ----------

% pull signals from first dataset
t    = S1.t(:);
q1   = S1.q1(:);
q2   = S1.q2(:);
dq1  = S1.dq1(:);
dq2  = S1.dq2(:);
ddq1 = S1.ddq1(:);
ddq2 = S1.ddq2(:);
u2   = S1.u2(:);

delta = q1 - q2;
c = cos(delta);
s = sin(delta);

% part of the second equation that does NOT contain b2
known_term = ...
    (m2*L2^2/3) .* ddq2 + ...
    (L1*L2*m2/2) .* c .* ddq1 - ...
    (L1*L2*m2/2) .* s .* (dq1 - dq2) .* dq1 + ...
    (981*L2*m2/200) .* sin(q2);

% linear least–squares form: y = phi * b2
phi = dq2;
y   = u2 - known_term;

% estimate b2 (one–parameter LS)
b2_hat = (phi' * y) / (phi' * phi);
fprintf('Estimated b2 from dataset 1: %.6f\n', b2_hat);

%% ---- local function for the ODE model ----

function dx = double_pendulum_ode(t, x, params, u_fun)
% simple state-space model of the double pendulum

q1  = x(1);
q2  = x(2);
dq1 = x(3);
dq2 = x(4);

u1 = u_fun.u1(t);
u2 = u_fun.u2(t);

m1 = params.m1;
m2 = params.m2;
b1 = params.b1;
b2 = params.b2;
L1 = params.L1;
L2 = params.L2;

delta = q1 - q2;
c = cos(delta);
s = sin(delta);

% mass matrix
a11 = (m1/3 + m2) * L1^2;
a12 = (L1*L2*m2/2) * c;
a21 = a12;
a22 = (m2*L2^2)/3;

% right-hand side terms
rhs1 = u1 ...
       + (L1*L2*m2/2) * s * (dq1 - dq2) * dq2 ...
       - b1 * dq1 ...
       - (981*L1*(m1 + 2*m2)/200) * sin(q1);

rhs2 = u2 ...
       + (L1*L2*m2/2) * s * (dq1 - dq2) * dq1 ...
       - b2 * dq2 ...
       - (981*L2*m2/200) * sin(q2);

A   = [a11 a12;
       a21 a22];

ddq = A \ [rhs1; rhs2];

dx      = zeros(4,1);
dx(1)   = dq1;
dx(2)   = dq2;
dx(3:4) = ddq;
end

%% ---------- VALIDATION USING DATASET 2 ----------

% pull signals from second dataset
t2    = S2.t(:);
q1_2  = S2.q1(:);
q2_2  = S2.q2(:);
dq1_2 = S2.dq1(:);
dq2_2 = S2.dq2(:);
u1_2  = S2.u1(:);
u2_2  = S2.u2(:);

% pack parameters with identified b2
params.m1 = m1;
params.m2 = m2;
params.b1 = b1;
params.b2 = b2_hat;
params.L1 = L1;
params.L2 = L2;

% input torques as functions of time
u_fun2.u1 = @(t) interp1(t2, u1_2, t, 'linear', 'extrap');
u_fun2.u2 = @(t) interp1(t2, u2_2, t, 'linear', 'extrap');

% initial state taken from first sample of dataset 2
x0_2 = [q1_2(1); q2_2(1); dq1_2(1); dq2_2(1)];

% simulate and compare with dataset 2
ode2 = @(t,x) double_pendulum_ode(t, x, params, u_fun2);
[tsim2, xsim2] = ode45(ode2, t2, x0_2);

figure;
subplot(2,1,1);
plot(t2, q1_2, 'k', 'LineWidth', 1.2); hold on;
plot(tsim2, xsim2(:,1), '--', 'LineWidth', 1.2);
xlabel('Time [s]'); ylabel('q1 [rad]');
legend('data2 q1','model q1');

subplot(2,1,2);
plot(t2, q2_2, 'k', 'LineWidth', 1.2); hold on;
plot(tsim2, xsim2(:,2), '--', 'LineWidth', 1.2);
xlabel('Time [s]'); ylabel('q2 [rad]');
legend('data2 q2','model q2');
sgtitle(sprintf('Validation with b2 = %.6f', b2_hat));
