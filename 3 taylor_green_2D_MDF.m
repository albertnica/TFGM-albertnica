clear all
clc
tic

% ==================================================================
%               Vortice de Taylor–Green bidimensional               
% ==================================================================

set(groot, 'DefaultAxesFontSize', 18, ...                 % Tamaño de fuente de ejes
           'DefaultTextFontSize', 20, ...                 % Tamaño de fuente de textos
           'DefaultColorbarFontSize', 18, ...             % Tamaño de fuente de las etiquetas de colorbar
           'DefaultAxesTitleFontSizeMultiplier', 1.2, ... % Multiplicador para titulos
           'DefaultAxesLabelFontSizeMultiplier', 1.1);    % Multiplicador para ejes

%% Parametros fisicos y numericos

N1   = 200;       % puntos totales en x
N2   = 200;       % puntos totales en y
Re   = 10;        % número de Reynolds
beta = 1e3;       % compresibilidad artificial
dt   = 1/200;     % paso de tiempo
T    = 1.0;       % tiempo final
Nt   = round(T/dt);

% Mallado completo en [0, 2*pi] × [0, 2*pi]
x  = linspace(0, 2*pi, N1);
y  = linspace(0, 2*pi, N2);
dx = (2*pi)/(N1-1);
dy = (2*pi)/(N2-1);
[X, Y] = meshgrid(x, y);

% Número de nodos interiores
mx = N1-1;
my = N2-1;
m  = mx * my;

% Operadores 1D de diferencias finitas con periodicidad
ex = ones(mx,1);
ey = ones(my,1);

% Laplaciano 1D periodico
Lx = spdiags([ex -2*ex ex],[-1 0 1],mx,mx);
Lx(1,mx)  = 1;    % periodicidad conectando 1 con mx
Lx(mx,1)  = 1;    % periodicidad conectando mx con 1
Lx = Lx / dx^2;

Ly = spdiags([ey -2*ey ey],[-1 0 1],my,my);
Ly(1,my)  = 1;    % periodicidad conectando 1 con my
Ly(my,1)  = 1;    % periodicidad conectando my con 1
Ly = Ly / dy^2;

% Gradiente 1D periodico
Dx = spdiags([-ex 0*ex ex],[-1 0 1],mx,mx);
Dx(1,mx) = -1;    % periodicidad, coeficiente de i=1 ← i=mx
Dx(mx,1) =  1;    % periodicidad, coeficiente de i=mx → i=1
Dx = Dx / (2*dx);

Dy = spdiags([-ey 0*ey ey],[-1 0 1],my,my);
Dy(1,my) = -1;    % periodicidad, coeficiente de j=1 ← j=my
Dy(my,1) =  1;    % periodicidad, coeficiente de j=my → j=1
Dy = Dy / (2*dy);

% Operadores 2D (Kronecker)
Ix = speye(mx);
Iy = speye(my);
L  = kron(Iy, Lx) + kron(Ly, Ix);   % Laplaciano 2D periódico
A2 = kron(Iy, Dx);                  % gradiente ∂/∂x periódico
A3 = kron(Dy, Ix);                  % gradiente ∂/∂y periódico

% Montaje de M
A1 = (1/dt)*speye(m) - (1/Re)*L;
Z  = sparse(m,m);
I3 = speye(m);
M  = [ A1,         Z,      A2;
       Z,          A1,     A3;
      beta*A2,  beta*A3,   I3 ];

% Condiciones iniciales en la malla completa
u_mat        =  cos(X).*sin(Y);
v_mat        = -sin(X).*cos(Y);
p_an_spatial = (cos(2*X)+cos(2*Y))/4;

% Extraccion de interiores excluyendo el ultimo indice periodico
u = reshape( u_mat(1:end-1, 1:end-1)' , m, 1 );        % uso 1:end-1 en lugar de 2:end-1
v = reshape( v_mat(1:end-1, 1:end-1)' , m, 1 );        % similar para v
p = reshape( p_an_spatial(1:end-1, 1:end-1)' , m, 1 ); % similar para p
xk = [u; v; p];

% Bucle temporal hasta t = 1
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
% Mapas de calor para u, v y vorticidad
% ====================================

% Reconstruir matrices 2D 
U = reshape(u, mx, my)';
V = reshape(v, mx, my)';

% Submalla de nodos interiores periodicos
x_int = x(1:end-1);
y_int = y(1:end-1);
[X_int, Y_int] = meshgrid(x_int, y_int);

% Campos u y v: mapas de calor
figure;
campos  = {U, V};
nombres = {'u', 'v'};
for k = 1:2
    subplot(1,2,k);
    contourf( X_int, Y_int, campos{k}, 20, 'LineColor', 'none' );
    cb = colorbar;
    cb.Label.String = sprintf('%s (t = %.2f)', nombres{k}, T);
    title( sprintf('%s en t = %.2f', nombres{k}, T) );
    axis equal tight;
    xlabel('x'); ylabel('y');
end

% Vorticidad numérica en t=1
omega_vec = (kron(Iy, Dx)*v) - (kron(Dy, Ix)*u);
OMEGA     = reshape(omega_vec, mx, my)';

figure;
contourf( X_int, Y_int, OMEGA, 20, 'LineColor', 'none' );
cb = colorbar;
cb.Label.String = sprintf('\\omega (t = %.2f)', 1);
title('Vorticidad numérica en t = 1.00');
axis equal tight;
xlabel('x'); ylabel('y');

% Vorticidad exacta en t=1
t_ex        = T;
omega_exact = -2 * cos(X_int) .* cos(Y_int) * exp(-2*(1/Re)*t_ex);

figure;
contourf( X_int, Y_int, omega_exact, 20, 'LineColor', 'none' );
cb = colorbar;
cb.Label.String = sprintf('\\omega (t = %.2f)', t_ex);
title(sprintf('Vorticidad exacta en t = %.2f', t_ex));
axis equal tight;
xlabel('x'); ylabel('y');

toc

%% Cálculo del Error Cuadrático Medio (ECM) de la vorticidad
% Diferencia entre solución numérica y exacta
error_mat = OMEGA - omega_exact;

% ECM y RMS
ECM = mean(error_mat(:).^2);
RMS = sqrt(ECM);

% Mostrar resultados
fprintf('Error Cuadrático Medio (ECM) = %.2e\n', ECM);
