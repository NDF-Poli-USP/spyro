Lx = 4.0;
Lz = 4.0;

Nx = 100;
Nz = 100;
h  = Lx / Nx;

// Pontos
Point(1) = {0, 0, 0, h};
Point(2) = {Lx, 0, 0, h};
Point(3) = {Lx, 2, 0, h};
Point(4) = {0, 2, 0, h};
Point(5) = {Lx, 4, 0, h};
Point(6) = {0, 4, 0, h};

// Linhas
Line(1) = {1, 2}; // bottom sólido
Line(2) = {2, 3}; // direita sólida
Line(3) = {3, 4}; // interface
Line(4) = {4, 1}; // esquerda sólida

Line(5) = {4, 6}; // esquerda fluido
Line(6) = {6, 5}; // topo fluido
Line(7) = {5, 3}; // direita fluido

// Superfícies
Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};    // sólido

Curve Loop(2) = {3, 5, 6, 7};
Plane Surface(2) = {2};    // fluido

// PHYSICALS que o Firedrake vai ler:
Physical Surface(2) = {1}; // solid_id   = 2  (superfície 1 = parte de baixo)
Physical Surface(1) = {2}; // fluid_id   = 1  (superfície 2 = parte de cima)
Physical Line(3)    = {3}; // interface_id = 3 (linha y=2)
