clear all
clc
tic

% ==================================================================
%               Two-dimensional Taylor-Green vortex                 
% ==================================================================

set(groot, 'DefaultAxesFontSize', 18, ...                 % Axes font size
           'DefaultTextFontSize', 20, ...                 % Text font size
           'DefaultColorbarFontSize', 18, ...             % Colorbar label font size
           'DefaultAxesTitleFontSizeMultiplier', 1.2, ... % Title font size multiplier
           'DefaultAxesLabelFontSizeMultiplier', 1.1);    % Axis label font size multiplier

%% Physical and numerical parameters

N1   = 200;       % total grid points in x
N2   = 200;       % total grid points in y
Re   = 10;        % Reynolds number
beta = 1e3;       % artificial compressibility parameter
dt   = 1/200;     % time step size
T    = 1.0;       % final simulation time
Nt   = round(T/dt);

% Full computational grid on [0, 2*pi] x [0, 2*pi]
x  = linspace(0, 2*pi, N1);
y  = linspace(0, 2*pi, N2);
dx = (2*pi)/(N1-1);
dy = (2*pi)/(N2-1);
[X, Y] = meshgrid(x, y);

% Number of interior nodes
mx = N1-1;
my = N2-1;
m  = mx * my;

% 1D finite difference operators with periodic boundary conditions
ex = ones(mx,1);
ey = ones(my,1);

% 1D periodic Laplacian
Lx = spdiags([ex -2*ex ex],[-1 0 1],mx,mx);
Lx(1,mx)  = 1;    % periodic connection from 1 to mx
Lx(mx,1)  = 1;    % periodic connection from mx to 1
Lx = Lx / dx^2;

Ly = spdiags([ey -2*ey ey],[-1 0 1],my,my);
Ly(1,my)  = 1;    % periodic connection from 1 to my
Ly(my,1)  = 1;    % periodic connection from my to 1
Ly = Ly / dy^2;

% 1D periodic Gradient
Dx = spdiags([-ex 0*ex ex],[-1 0 1],mx,mx);
Dx(1,mx) = -1;    % periodic boundary coupling: i=1 <- i=mx
Dx(mx,1) =  1;    % periodic boundary coupling: i=mx -> i=1
Dx = Dx / (2*dx);

Dy = spdiags([-ey 0*ey ey],[-1 0 1],my,my);
Dy(1,my) = -1;    % periodic boundary coupling: j=1 <- j=my
Dy(my,1) =  1;    % periodic boundary coupling: j=my -> j=1
Dy = Dy / (2*dy);

% 2D operators (Kronecker tensor products)
Ix = speye(mx);
Iy = speye(my);
L  = kron(Iy, Lx) + kron(Ly, Ix);   % 2D periodic Laplacian
A2 = kron(Iy, Dx);                  % periodic gradient d/dx
A3 = kron(Dy, Ix);                  % periodic gradient d/dy

% Matrix M assembly
A1 = (1/dt)*speye(m) - (1/Re)*L;
Z  = sparse(m,m);
I3 = speye(m);
M  = [ A1,         Z,      A2;
       Z,          A1,     A3;
      beta*A2,  beta*A3,   I3 ];

% Initial conditions on the full grid
u_mat        =  cos(X).*sin(Y);
v_mat        = -sin(X).*cos(Y);
p_an_spatial = (cos(2*X)+cos(2*Y))/4;

% Extract interior nodes excluding the redundant periodic boundary index
u = reshape( u_mat(1:end-1, 1:end-1)' , m, 1 );        % using 1:end-1 instead of 2:end-1
v = reshape( v_mat(1:end-1, 1:end-1)' , m, 1 );        % velocity component v
p = reshape( p_an_spatial(1:end-1, 1:end-1)' , m, 1 ); % pressure field p
xk = [u; v; p];

% Time-stepping loop up to t = 1.0
for k = 1:Nt
    rhs = [ (1/dt)*u;
            (1/dt)*v;
             zeros(m,1) ];
    xk = M \ rhs;
    u  = xk(         1:m );
    v  = xk((m+1):2*m );
    p  = xk((2*m+1):3*m);
end

% ====================================
% Heatmaps for u, v and vorticity
% ====================================

% Reconstruct 2D matrices
U = reshape(u, mx, my)';
V = reshape(v, mx, my)';

% Subgrid of periodic interior nodes
x_int = x(1:end-1);
y_int = y(1:end-1);
[X_int, Y_int] = meshgrid(x_int, y_int);

% Velocity fields u and v: heatmaps
figure;
fields = {U, V};
names  = {'u', 'v'};
for k = 1:2
    subplot(1,2,k);
    contourf( X_int, Y_int, fields{k}, 20, 'LineColor', 'none' );
    cb = colorbar;
    cb.Label.String = sprintf('%s (t = %.2f)', names{k}, T);
    title( sprintf('%s at t = %.2f', names{k}, T) );
    axis equal tight;
    xlabel('x'); ylabel('y');
end

% Numerical vorticity at t = 1
omega_vec = (kron(Iy, Dx)*v) - (kron(Dy, Ix)*u);
OMEGA     = reshape(omega_vec, mx, my)';

figure;
contourf( X_int, Y_int, OMEGA, 20, 'LineColor', 'none' );
cb = colorbar;
cb.Label.String = sprintf('\\omega (t = %.2f)', 1);
title('Numerical vorticity at t = 1.00');
axis equal tight;
xlabel('x'); ylabel('y');

% Exact analytical vorticity at t = 1
t_ex        = T;
omega_exact = -2 * cos(X_int) .* cos(Y_int) * exp(-2*(1/Re)*t_ex);

figure;
contourf( X_int, Y_int, omega_exact, 20, 'LineColor', 'none' );
cb = colorbar;
cb.Label.String = sprintf('\\omega (t = %.2f)', t_ex);
title(sprintf('Exact analytical vorticity at t = %.2f', t_ex));
axis equal tight;
xlabel('x'); ylabel('y');

toc

%% Mean Squared Error (MSE) calculation for vorticity
% Difference between numerical and exact analytical solutions
error_mat = OMEGA - omega_exact;

% MSE and RMS
MSE = mean(error_mat(:).^2);
RMS = sqrt(MSE);

% Display results
fprintf('Mean Squared Error (MSE) = %.2e\n', MSE);

