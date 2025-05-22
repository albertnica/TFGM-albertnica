%% -------------------------------------------------------------
% Resolución de la ecuación del calor en un triángulo equilátero
% de lado 10, con vértices en (0,0), (0,10) y (5√3,5).
%
% - PDE: ∂u/∂t − Δu = 0
% - Dominio: triángulo equilátero, vértices A=(0,0), B=(0,10), C=(5√3,5)
% - Condición de contorno Dirichlet homogénea: u = 0 en los 3 lados
% - Condición inicial: u(x,y,0) = sin(π x) sin(π y)
% - Malla: triangular con Hmax = 0.5
% - Vector de tiempos: t ∈ [0, 1] con Δt = 0.05
% 
% Observación: 
%   Se usan coeficientes m=0, d=1, c=1, a=0, f=0 para ajustar la caja PDE a 
%   la ecuación de calor (primer orden en t). Así MATLAB no exige derivada inicial.
%% -------------------------------------------------------------

clc
clear all

%% 1. Crear modelo PDE para problema transitorio (ecuación del calor)
model = createpde();  % No se pasa argumento “1” para evitar tratar como problema elíptico

%% 2. Definir la geometría: triángulo equilátero de lado 10
%    Vértices: (0,0), (0,10), (5*sqrt(3), 5)
R1 = [2, 3,   0,    0,   5*sqrt(3),    0,   10,   5]';
g  = decsg(R1);         % Decodifica la descripción geométrica
geometryFromEdges(model, g);

%  Opcional: visualizar la geometría y las aristas numeradas
figure
pdegplot(model,"EdgeLabels","on")
title("Geometría: triángulo equilátero (lado = 10)")
axis equal

%% 3. Aplicar condición de contorno Dirichlet homogénea (u = 0) en las 3 aristas
%    Tras pdegplot, las aristas quedan etiquetadas como 1, 2 y 3.
applyBoundaryCondition(model, "dirichlet", "Edge", [1, 2, 3], "u", 0);

%% 4. Generar malla triangular con tamaño de elemento Hmax = 0.5
generateMesh(model, "Hmax", 0.5);

%  Opcional: visualizar la malla para comprobar densidad y cobertura
figure
pdemesh(model)
title("Malla del triángulo (Hmax = 0.5)")
axis equal

%% 5. Especificar coeficientes de la PDE para la ecuación del calor
%    La forma general es: 
%      m ∂^2u/∂t^2 + d ∂u/∂t - ∇·(c ∇u) + a u = f
%    Para ∂u/∂t - Δu = 0 elegimos:
%      m = 0, d = 1, c = 1, a = 0, f = 0.
specifyCoefficients(model, ...
    "m", 0, ...   % Sin término de segundo orden en t
    "d", 1, ...   % Término de primer orden: ∂u/∂t
    "c", 1, ...   % Difusividad unitaria (−Δu)
    "a", 0, ...   % Sin amortiguamiento
    "f", 0);      % Sin fuente de calor

%% 6. Fijar la condición inicial: u(x,y,0) = sin(π x) sin(π y)
setInitialConditions(model, @(location) sin(pi * location.x) .* sin(pi * location.y));

%% 7. Definir vector de tiempos para simular (de t=0 a t=1, con Δt = 0.05)
tlist = 0 : 0.05 : 1;  % [0, 0.05, 0.10, ..., 1.00]

%% 8. Resolver la PDE transitoria
%    solvepde con vector de tiempos devuelve la evolución nodal de u.
result = solvepde(model, tlist);

%    result.NodalSolution es de tamaño [NúmeroDeNodos × length(tlist)].

%% 9. Visualizar la solución en t = 1 (último instante)
u_final = result.NodalSolution(:, end);  % Columna final → t = 1

figure
pdeplot(model, "XYData", u_final, "ZData", u_final);
title("Temperatura en el triángulo para t = 1")
xlabel("x")
ylabel("y")
zlabel("u(x,y,1)")
colorbar
view(3)

%% 10. Animación de la evolución en varios instantes (cada 5 pasos)
figure
for idx = 1 : 5 : length(tlist)
    u_temp = result.NodalSolution(:, idx);
    pdeplot(model, "XYData", u_temp, "ZData", u_temp);
    title(sprintf("Temperatura en t = %.2f", tlist(idx)));
    xlabel("x"), ylabel("y"), zlabel("u");
    colorbar
    view(3)
    pause(0.1)
end

%% -------------------------------------------------------------
% FIN DEL CÓDIGO COMPLETO
%% -------------------------------------------------------------

