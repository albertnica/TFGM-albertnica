% ------------------------------------------------------------------------------
% Script completo para resolver Navier–Stokes incompresible con compresibilidad
% artificial (β finito) mediante diferencias finitas en un dominio [0,1]×[0,1].
% Se emplea un término implícito para la difusión y explícito para la convección.
% Finalmente se grafican las velocidades y la vorticidad numérica y exacta.
% ------------------------------------------------------------------------------

clear variables
close all
clc

%% ----------------------  Parámetros y discretización  ------------------------
N    = 50;         % Número de nodos interiores en cada dirección (x e y)
Lx   = 1.0;        % Longitud del dominio en x
Ly   = 1.0;        % Longitud del dominio en y
hx   = Lx / (N + 1);  % Tamaño de paso en x
hy   = Ly / (N + 1);  % Tamaño de paso en y

nu    = 1e-3;      % Viscosidad cinemática
tau   = 1e-2;      % Paso temporal
beta  = 1e5;       % Parámetro de compresibilidad artificial

% Número de pasos temporales y tiempo final:
M       = 100;                     % Número de iteraciones en el tiempo
T_final = M * tau;                % Tiempo final

% Número total de incógnitas en interior:
N_int   = N * N;                   % Número de nodos interiores (grid 2D)
N_vel   = 2 * N_int;               % Nº incógnitas de velocidad (u y v)
N_press = N_int;                   % Nº incógnitas de presión

% Construimos la rejilla de nodos interiores para graficar:
x_interior = (1:N) * hx;           % Coordenadas interiores en x
y_interior = (1:N) * hy;           % Coordenadas interiores en y
[X, Y]     = meshgrid(x_interior, y_interior);


%% -------------- 1) Construcción del Laplaciano 2D con Dirichlet  --------------
e = ones(N,1);

% Segunda derivada en x (interiores, Dirichlet en fronteras x=0 y x=1):
Tx = spdiags([-e, 2*e, -e], [-1, 0, 1], N, N) / hx^2;

% Segunda derivada en y (interiores, Dirichlet en fronteras y=0 y y=1):
Ty = spdiags([-e, 2*e, -e], [-1, 0, 1], N, N) / hy^2;

% Laplaciano 2D en interior = I⊗Tx + Ty⊗I
L2D = kron(speye(N), Tx) + kron(Ty, speye(N));   % Tamaño N_int×N_int

% Operador L_h para velocidad (u,v) es bloque diagonal [L2D 0; 0 L2D]
L_h = [ L2D,                    sparse(N_int, N_int);
        sparse(N_int, N_int),   L2D ];          % Tamaño N_vel×N_vel


%% --------- 2) Construcción del operador gradiente discreto G_h  --------------
% Derivada centrada en x para escalares en interior:
Dx1D = spdiags([-e, zeros(N,1), e], [-1, 0, 1], N, N) / (2 * hx);  % N×N
% Derivada centrada en y para escalares en interior:
Dy1D = spdiags([-e, zeros(N,1), e], [-1, 0, 1], N, N) / (2 * hy);  % N×N

% Gx = grad p en x sobre 2D interior: I_N ⊗ Dx1D
Gx = kron(speye(N), Dx1D);    % Tamaño N_int×N_int
% Gy = grad p en y sobre 2D interior: Dy1D ⊗ I_N
Gy = kron(Dy1D, speye(N));    % Tamaño N_int×N_int

% Bloque B = [Gx; Gy], acciones de ∇p en u y en v
B = [ Gx;
      Gy ];                   % Tamaño (2*N_int)×N_int

%% ---------- 3) Construcción del operador divergencia discreto D_h  -------------
% Bloque C = [ Gx, Gy ], aplicado a [u; v] → ∇·u
C = [ Gx,  Gy ];               % Tamaño N_int × (2*N_int)

%% ------------------ 4) Construcción del bloque D = (1/(β τ)) I  ---------------
D = (1 / (beta * tau)) * speye(N_int);   % Tamaño N_int×N_int

%% ------------------- 5) Construcción del bloque A ----------------------------
% A = (1/τ) I + ν L_h
A = (1 / tau) * speye(N_vel) + nu * L_h;  % Tamaño N_vel×N_vel

%% ---------------- 6) Ensamblaje del sistema monolítico -----------------------
% Matriz global M de tamaño (2*N_int + N_int) × (2*N_int + N_int)
M = [ A,       B;
      C,       D ];                        % Dim: (3*N_int)×(3*N_int)

%% --------------- 7) Condiciones iniciales y forzamiento ----------------------
% Velocidad inicial nula y fuerza externa nula
U_n   = zeros(N_vel,   1);   % [u^n; v^n], tamaño 2*N_int
F_np1 = zeros(N_vel,   1);   % Fuerza externa en tiempo n+1 (se asume cero)

% Inicialmente, presiones se pueden tomar cero
P_n   = zeros(N_press, 1);   % p^n, tamaño N_int

% Creamos vectores para guardar la solución en cada paso (si lo deseamos)
% Para almacenar sólo el último paso, guardaremos u_final, v_final, p_final.
u_final = zeros(N_int,1);
v_final = zeros(N_int,1);
p_final = zeros(N_int,1);

%% ------------------------- 8) Bucle temporal -----------------------------------
for n = 1 : M
    % -- 8.1) Construcción del término convectivo explícito C(U_n) --
    convU = convectiveTerm(U_n, N, hx, hy);   % Tamaño N_vel×1

    % -- 8.2) Construcción del vector R^n = (1/τ)*U_n + convU + F_np1 --
    R_n = (1 / tau) * U_n + convU + F_np1;     % N_vel×1

    % -- 8.3) Construcción del RHS monolítico [R_n; 0] --
    RHS = [ R_n; zeros(N_press, 1) ];         % Dim: (3*N_int)×1

    % -- 8.4) Resolución del sistema monolítico para U_{n+1}, P_{n+1} --
    X_np1 = M \ RHS;                          % Dim: (3*N_int)×1

    % -- 8.5) Extraer u_{n+1}, v_{n+1}, p_{n+1} de X_np1 --
    % Primeras 2*N_int entradas son [u; v; u; v; ...]
    u_vec = X_np1(1 : 2 : 2*N_int);            % N_int×1 (componentes u)
    v_vec = X_np1(2 : 2 : 2*N_int);            % N_int×1 (componentes v)
    % Últimas N_int entradas son p^{n+1}
    p_vec = X_np1(2*N_int + 1 : end);          % N_int×1

    % Guardar la solución final si n == M
    if n == M
        u_final = u_vec;
        v_final = v_vec;
        p_final = p_vec;
    end

    % -- 8.6) Preparar para el siguiente paso --
    U_n = X_np1(1 : 2*N_int);      % Actualizamos [u; v] para siguiente iteración
    P_n = p_vec;                   % Actualizamos presión (aunque no se usa en convectivo)
    % (En este ejemplo, F_np1 se mantiene cero; si cambia, se actualiza aquí)
end

%% ---------------------- 9) Construcción de Dx y Dy ----------------------------
% Para calcular la vorticidad ω = ∂v/∂x - ∂u/∂y, necesitamos matrices
% que apliquen la derivada centrada en x e y a escalares en 2D interior.
%
% Usamos exactamente Gx y Gy definidos antes, pues aplican ∂/∂x y ∂/∂y
% a un vector escalar de dimensión N_int. Así:
Dx = Gx;    % Aplica ∂/∂x al vector v_vec
Dy = Gy;    % Aplica ∂/∂y al vector u_vec

%% ----------------- 10) Graficar resultados en t = T_final ---------------------
% Reconstruir matrices U y V en 2D (Ny×Nx)
U = reshape(u_final, Ny, Nx);   % cada fila fija en y
V = reshape(v_final, Ny, Nx);

% --- (a) Velocidades u y v en t = T_final ---
figure;
campos   = {U, V};
nombres  = {'u', 'v'};
for k = 1 : 2
    subplot(1, 2, k);
    contourf(X, Y, campos{k}, 20, 'LineColor', 'none');
    colormap(parula);
    cb = colorbar;
    cb.Label.String = sprintf('%s (t = %.2f)', nombres{k}, T_final);
    title(sprintf('%s en t = %.2f', nombres{k}, T_final));
    axis equal tight;
    xlabel('x'); ylabel('y');
end

% --- (b) Vorticidad numérica en t = T_final ---
omega_vec = Dx * v_final - Dy * u_final;    % N_int×1
OMEGA     = reshape(omega_vec, Ny, Nx);

figure;
contourf(X, Y, OMEGA, 20, 'LineColor', 'none');
colormap(parula);
cb = colorbar;
cb.Label.String = sprintf('\\omega_{num} (t = %.2f)', T_final);
title(sprintf('Vorticidad numérica en t = %.2f', T_final));
axis equal tight;
xlabel('x'); ylabel('y');

% --- (c) Vorticidad exacta en t = T_final (Taylor–Green) ---
t_ex       = T_final;
omega_exact = -2 .* cos(X) .* cos(Y) .* exp(-2 * nu * t_ex);

figure;
contourf(X, Y, omega_exact, 20, 'LineColor', 'none');
colormap(parula);
cb = colorbar;
cb.Label.String = sprintf('\\omega_{exacta} (t = %.2f)', t_ex);
title(sprintf('Vorticidad exacta en t = %.2f', t_ex));
axis equal tight;
xlabel('x'); ylabel('y');


%% -------------------- Función auxiliar: término convectivo --------------------
function convU = convectiveTerm(U_n, N, hx, hy)
    % U_n: vector (2*N^2×1) = [u(1,1); v(1,1); u(2,1); v(2,1); ...; u(N,N); v(N,N)]
    % Convierte en convU = [u ∂_x u + v ∂_y u; u ∂_x v + v ∂_y v] en cada nodo interior.
    N_int = N * N;
    convU = zeros(2*N_int, 1);
    
    % Recorremos cada nodo interior (i,j)
    for j = 1 : N
        for i = 1 : N
            % Índices en el vector U_n:
            idx_u = 2 * ((j - 1) * N + (i - 1)) + 1;  % posición de u(i,j)
            idx_v = idx_u + 1;                       % posición de v(i,j)
            u_ij = U_n(idx_u);
            v_ij = U_n(idx_v);
            
            % Vecinos en x para u(i,j):
            if i == 1
                u_ip1 = U_n(idx_u + 2);    % u(2,j)
                u_im1 = 0;                 % Dirichlet u(0,j) = 0
            elseif i == N
                u_ip1 = 0;                              % u(N+1,j) = 0
                u_im1 = U_n(idx_u - 2);                 % u(N-1,j)
            else
                u_ip1 = U_n(idx_u + 2);                 % u(i+1,j)
                u_im1 = U_n(idx_u - 2);                 % u(i-1,j)
            end
            
            % Vecinos en y para u(i,j):
            if j == 1
                u_ijp1 = U_n(idx_u + 2*N);  % u(i,2)
                u_ijm1 = 0;                 % u(i,0) = 0
            elseif j == N
                u_ijp1 = 0;                              % u(i,N+1) = 0
                u_ijm1 = U_n(idx_u - 2*N);               % u(i,N-1)
            else
                u_ijp1 = U_n(idx_u + 2*N);               % u(i,j+1)
                u_ijm1 = U_n(idx_u - 2*N);               % u(i,j-1)
            end
            
            % Vecinos en x para v(i,j):
            if i == 1
                v_ip1 = U_n(idx_v + 2);
                v_im1 = 0;
            elseif i == N
                v_ip1 = 0;
                v_im1 = U_n(idx_v - 2);
            else
                v_ip1 = U_n(idx_v + 2);
                v_im1 = U_n(idx_v - 2);
            end
            
            % Vecinos en y para v(i,j):
            if j == 1
                v_ijp1 = U_n(idx_v + 2*N);
                v_ijm1 = 0;
            elseif j == N
                v_ijp1 = 0;
                v_ijm1 = U_n(idx_v - 2*N);
            else
                v_ijp1 = U_n(idx_v + 2*N);
                v_ijm1 = U_n(idx_v - 2*N);
            end
            
            % Derivadas centradas de u en x e y:
            du_dx = (u_ip1 - u_im1) / (2 * hx);
            du_dy = (u_ijp1 - u_ijm1) / (2 * hy);
            
            % Derivadas centradas de v en x e y:
            dv_dx = (v_ip1 - v_im1) / (2 * hx);
            dv_dy = (v_ijp1 - v_ijm1) / (2 * hy);
            
            % Término convectivo para u(i,j):
            conv_u = u_ij * du_dx + v_ij * du_dy;
            % Término convectivo para v(i,j):
            conv_v = u_ij * dv_dx + v_ij * dv_dy;
            
            % Almacenamos en convU
            convU(idx_u) = conv_u;
            convU(idx_v) = conv_v;
        end
    end
end
