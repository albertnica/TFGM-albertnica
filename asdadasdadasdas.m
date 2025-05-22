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
N        = 50;         % Número de nodos interiores en cada dirección (x e y)
Lx       = 1.0;        % Longitud del dominio en x
Ly       = 1.0;        % Longitud del dominio en y
hx       = Lx / (N + 1);  % Tamaño de paso en x
hy       = Ly / (N + 1);  % Tamaño de paso en y

nu       = 1e-3;       % Viscosidad cinemática
tau      = 1e-2;       % Paso temporal
beta     = 1e5;        % Parámetro de compresibilidad artificial

% Número de pasos temporales y tiempo final:
numSteps = 100;                     % Nº de iteraciones en el tiempo
T_final  = numSteps * tau;          % Tiempo final

% Definimos Nx y Ny para reshape
Nx       = N;
Ny       = N;

% Número total de incógnitas en interior:
N_int    = N * N;                   % Nº de nodos interiores (grid 2D)
N_vel    = 2 * N_int;               % Nº incógnitas de velocidad (u y v)
N_press  = N_int;                   % Nº incógnitas de presión

% Construimos la rejilla de nodos interiores para graficar:
x_interior = (1:N) * hx;           % Coordenadas interiores en x
y_interior = (1:N) * hy;           % Coordenadas interiores en y
[X, Y]      = meshgrid(x_interior, y_interior);


%% -------------- 1) Construcción del Laplaciano 2D con Dirichlet  --------------
e  = ones(N,1);
Tx = spdiags([-e, 2*e, -e], [-1, 0, 1], N, N) / hx^2;
Ty = spdiags([-e, 2*e, -e], [-1, 0, 1], N, N) / hy^2;

L2D = kron(speye(N), Tx) + kron(Ty, speye(N));   % Tamaño N_int×N_int
L_h = [ L2D, sparse(N_int, N_int);
        sparse(N_int, N_int), L2D ];             % Tamaño N_vel×N_vel


%% --------- 2) Construcción del operador gradiente discreto G_h  --------------
Dx1D = spdiags([-e, zeros(N,1), e], [-1, 0, 1], N, N) / (2 * hx);
Dy1D = spdiags([-e, zeros(N,1), e], [-1, 0, 1], N, N) / (2 * hy);

Gx = kron(speye(N), Dx1D);    % Tamaño N_int×N_int
Gy = kron(Dy1D, speye(N));    % Tamaño N_int×N_int

B  = [ Gx; Gy ];              % Tamaño (2*N_int)×N_int


%% ---------- 3) Construcción del operador divergencia discreto D_h  -------------
C  = [ Gx, Gy ];              % Tamaño N_int × (2*N_int)


%% ------------------ 4) Construcción del bloque D = (1/(β τ)) I  ---------------
D_block = (1 / (beta * tau)) * speye(N_int);   % Tamaño N_int×N_int


%% ------------------- 5) Construcción del bloque A ----------------------------
A = (1 / tau) * speye(N_vel) + nu * L_h;        % Tamaño N_vel×N_vel


%% ---------------- 6) Ensamblaje del sistema monolítico -----------------------
SystemMatrix = [ A,       B;
                 C,       D_block ];            % Dim: (3*N_int)×(3*N_int)


%% --------------- 7) Condiciones iniciales y forzamiento ----------------------
U_n     = zeros(N_vel,   1);   % [u^n; v^n]
F_np1   = zeros(N_vel,   1);   % Fuerza externa (cero)
P_n     = zeros(N_press, 1);   % p^n

u_final = zeros(N_int,1);
v_final = zeros(N_int,1);
p_final = zeros(N_int,1);


%% ------------------------- 8) Bucle temporal -----------------------------------
for n = 1 : numSteps
    convU = convectiveTerm(U_n, N, hx, hy);      % Término convectivo explícito
    R_n   = (1 / tau) * U_n + convU + F_np1;      % Vector R^n
    RHS   = [ R_n; zeros(N_press, 1) ];          % RHS monolítico

    X_np1 = SystemMatrix \ RHS;                  % Resolución monolítica

    % Extraemos u, v, p de X_np1
    u_vec = X_np1(1 : 2 : 2*N_int);
    v_vec = X_np1(2 : 2 : 2*N_int);
    p_vec = X_np1(2*N_int + 1 : end);

    if n == numSteps
        u_final = u_vec;
        v_final = v_vec;
        p_final = p_vec;
    end

    U_n = X_np1(1 : 2*N_int);
    P_n = p_vec;
end


%% ---------------------- 9) Construcción de Dx y Dy ----------------------------
Dx = Gx;
Dy = Gy;


%% ----------------- 10) Graficar resultados en t = T_final ---------------------
U = reshape(u_final, Ny, Nx);
V = reshape(v_final, Ny, Nx);

figure;
for k = 1 : 2
    subplot(1, 2, k);
    if k == 1; Z = U; name = 'u'; else; Z = V; name = 'v'; end
    contourf(X, Y, Z, 20, 'LineColor', 'none');
    cb = colorbar;
    cb.Label.String = sprintf('%s (t = %.2f)', name, T_final);
    title(sprintf('%s en t = %.2f', name, T_final));
    axis equal tight; xlabel('x'); ylabel('y');
end

omega_vec   = Dx * v_final - Dy * u_final;
OMEGA       = reshape(omega_vec, Ny, Nx);

figure;
contourf(X, Y, OMEGA, 20, 'LineColor', 'none');
cb = colorbar;
cb.Label.String = sprintf('\\omega_{num} (t = %.2f)', T_final);
title(sprintf('Vorticidad numérica en t = %.2f', T_final));
axis equal tight; xlabel('x'); ylabel('y');

omega_exact = -2 .* cos(X) .* cos(Y) .* exp(-2 * nu * T_final);
figure;
contourf(X, Y, omega_exact, 20, 'LineColor', 'none');
cb = colorbar;
cb.Label.String = sprintf('\\omega_{exacta} (t = %.2f)', T_final);
title(sprintf('Vorticidad exacta en t = %.2f', T_final));
axis equal tight; xlabel('x'); ylabel('y');


%% -------------------- Función auxiliar: término convectivo --------------------
function convU = convectiveTerm(U_n, N, hx, hy)
    N_int = N * N;
    convU = zeros(2*N_int, 1);
    for j = 1 : N
        for i = 1 : N
            idx_u = 2 * ((j - 1) * N + (i - 1)) + 1;
            idx_v = idx_u + 1;
            u_ij  = U_n(idx_u);
            v_ij  = U_n(idx_v);
            % Calculamos vecinos con condiciones de contorno Dirichlet = 0
            if i == 1;   u_im1 = 0; else; u_im1 = U_n(idx_u - 2); end
            if i == N;   u_ip1 = 0; else; u_ip1 = U_n(idx_u + 2); end
            if j == 1;   u_ijm1 = 0; else; u_ijm1 = U_n(idx_u - 2*N); end
            if j == N;   u_ijp1 = 0; else; u_ijp1 = U_n(idx_u + 2*N); end

            if i == 1;   v_im1 = 0; else; v_im1 = U_n(idx_v - 2); end
            if i == N;   v_ip1 = 0; else; v_ip1 = U_n(idx_v + 2); end
            if j == 1;   v_ijm1 = 0; else; v_ijm1 = U_n(idx_v - 2*N); end
            if j == N;   v_ijp1 = 0; else; v_ijp1 = U_n(idx_v + 2*N); end

            du_dx = (u_ip1 - u_im1) / (2 * hx);
            du_dy = (u_ijp1 - u_ijm1) / (2 * hy);
            dv_dx = (v_ip1 - v_im1) / (2 * hx);
            dv_dy = (v_ijp1 - v_ijm1) / (2 * hy);

            convU(idx_u) = u_ij * du_dx + v_ij * du_dy;
            convU(idx_v) = u_ij * dv_dx + v_ij * dv_dy;
        end
    end
end
