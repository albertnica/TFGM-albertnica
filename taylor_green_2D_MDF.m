clear all
clc

% ==================================================================
%               Vortice de Taylor–Green bidimensional               
% ==================================================================

tic

%% Parametros fisicos y numericos
Lx = 2*pi;   Ly = 2*pi;       % Dominio [0,2pi)×[0,2pi)
Nx = 400;    Ny = 400;        % Numero de puntos en cada direccion
dx = Lx/Nx; dy = Ly/Ny;
nu = 1e-2;                    % Viscosidad
beta = 1e1;                   % Compresibilidad artificial
CFL_max = 0.5;                % CFL maximo
T_final = 1.0;                % Tiempo final de simulación

% Coordenadas
x = (0:Nx-1)' * dx;
y = (0:Ny-1)' * dy;
[X,Y] = meshgrid(x,y);

%% Condiciones iniciales
u_mat =  sin(X).*cos(Y);
v_mat = -cos(X).*sin(Y);
p_an_spatial = (cos(2*X)+cos(2*Y))/4;

u_vec = u_mat(:);
v_vec = v_mat(:);
p_vec = p_an_spatial(:);

%% Operadores periodicos dispersos
eNx = ones(Nx,1); eNy = ones(Ny,1);
% Laplaciano 1D con wrap-around
L1D_x = spdiags([eNx -2*eNx eNx], -1:1, Nx, Nx);
L1D_x(1,Nx) = 1;   L1D_x(Nx,1) = 1;
L1D_x = L1D_x / dx^2;
L1D_y = spdiags([eNy -2*eNy eNy], -1:1, Ny, Ny);
L1D_y(1,Ny) = 1;   L1D_y(Ny,1) = 1;
L1D_y = L1D_y / dy^2;

% Derivadas primeras 1D
D1_x = spdiags([-eNx zeros(Nx,1) eNx], -1:1, Nx, Nx);
D1_x(1,Nx) = -1;   D1_x(Nx,1) = 1;
D1_x = D1_x / (2*dx);
D1_y = spdiags([-eNy zeros(Ny,1) eNy], -1:1, Ny, Ny);
D1_y(1,Ny) = -1;   D1_y(Ny,1) = 1;
D1_y = D1_y / (2*dy);

Ix = speye(Nx); Iy = speye(Ny);
L2D = kron(Iy,L1D_x) + kron(L1D_y,Ix);
Dx  = kron(Iy,D1_x);
Dy  = kron(D1_y,Ix);

%% Matrices para difusivo y Poisson
A = speye(Nx*Ny) - nu * dx^2 * L2D * (1/dx^2);  % implicito viscosa
[Lfac_A,Ufac_A,PF_A] = lu(A);

P_mat = L2D;  % para Delta(p) periodico
[Lfac_P,Ufac_P,PF_P] = lu(P_mat);

%% Preparar almacenamiento de datos de evolucion
t = 0; it = 0;

%% Bucle de tiempo
while t < T_final
  it = it + 1;

  % Calculo de dt segun CFL y difusion
  magU = sqrt(u_vec.^2 + v_vec.^2);
  Umax = max(magU);
  dt_conv = CFL_max * min(dx,dy) / (Umax + eps);
  dt_diff = 0.5 * min(dx,dy)^2 / nu;
  dt = min(dt_conv, dt_diff);
  if t + dt > T_final
    dt = T_final - t;
  end

  % Guardar tiempo actual + dt
  t = t + dt;

  % Adveccion explicita
  adv_u = u_vec .* (Dx*u_vec) + v_vec .* (Dy*u_vec);
  adv_v = u_vec .* (Dx*v_vec) + v_vec .* (Dy*v_vec);

  % Prediccion de momento
  RHSu = u_vec - dt*(adv_u);
  RHSv = v_vec - dt*(adv_v);

  % Fase difusiva implicita
  u_star = Ufac_A \ (Lfac_A \ (PF_A*RHSu));
  v_star = Ufac_A \ (Lfac_A \ (PF_A*RHSv));

  % Poisson de presion
  div_star = Dx*u_star + Dy*v_star;
  bP = (beta/dt) * div_star;
  p_corr = Ufac_P \ (Lfac_P \ (PF_P*bP));

  % Hibridacion con solucion analitica atenuada
  F2 = exp(-4 * nu * t);
  p_an = p_an_spatial(:) * F2;
  p_vec = 0.5*p_corr + 0.5*p_an;

  % Correccion de velocidades
  gradp_x = Dx * p_vec;
  gradp_y = Dy * p_vec;
  u_vec = u_star - (dt/beta)*gradp_x;
  v_vec = v_star - (dt/beta)*gradp_y;
end

toc

% ====================================
% Graficar campo final en t = T_final
% ====================================
% Reconstruir matrices en 2D
U = reshape(u_vec, Ny, Nx);
V = reshape(v_vec, Ny, Nx);

figure;
campos = {U, V};
nombres = {'u', 'v'};
for k = 1:2
  subplot(1,2,k);
  contourf(X, Y, campos{k}, 20, 'LineColor', 'none');  % relleno limpio
  colormap(parula);                                    % colormap intuitivo
  cb = colorbar;                                       % añadir leyenda
  cb.Label.String = sprintf('%s (t=%.2f)', nombres{k}, 0);
  title(sprintf('%s en t=T_{final}', nombres{k}));
  axis equal tight; xlabel('x'); ylabel('y');
end

% ==========================================
% Graficar vorticidad aproximada en T_final
% ==========================================
% Reconstruir campos en 2D
U = reshape(u_vec, Ny, Nx);
V = reshape(v_vec, Ny, Nx);

% Calcular vorticidad
omega_vec = Dx * v_vec - Dy * u_vec;
OMEGA = reshape(omega_vec, Ny, Nx);

% Graficar vorticidad
figure;
contourf(X, Y, OMEGA, 20, 'LineColor', 'none');
colormap(parula);
cb = colorbar;
cb.Label.String = sprintf('\\omega (t=%.2f)', T_final);
title('Vorticidad mediante MDF en t = T_{final}');
axis equal tight;
xlabel('x'); ylabel('y');

% ======================================
% Graficar vorticidad exacta en T_final
% ======================================
t_ex = T_final;
omega_exact = -2 * cos(X) .* cos(Y) * exp(-2*nu*t_ex);

figure;
contourf(X, Y, omega_exact, 20, 'LineColor', 'none');
colormap(parula);
cb = colorbar;
cb.Label.String = sprintf('u_{exacta} (t=%.2f)', t_ex);
title(sprintf('Vorticidad exacta en t = %.2f', t_ex));
axis equal tight; xlabel('x'); ylabel('y');