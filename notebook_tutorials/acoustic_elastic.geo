//------------------------------------------------------------
// PARAMETERS
//------------------------------------------------------------
Lx = 4.0;
Lz = 4.0;

Nx = 100;   // total divisions in x
Nz = 100;   // total divisions in z

// Half for fluid, half for solid
Nz_s = Nz/2;   // solid (bottom)
Nz_f = Nz/2;   // fluid (top)

//------------------------------------------------------------
// POINTS
//------------------------------------------------------------

// bottom rectangle (solid: y = 0 to y = 2)
Point(1) = {0,   0, 0,  Lx/Nx};
Point(2) = {Lx,  0, 0,  Lx/Nx};
Point(3) = {Lx,  2, 0,  Lx/Nx};
Point(4) = {0,   2, 0,  Lx/Nx};

// top rectangle (fluid: y = 2 to y = 4)
Point(5) = {0,   2, 0,  Lx/Nx};
Point(6) = {Lx,  2, 0,  Lx/Nx};
Point(7) = {Lx,  4, 0,  Lx/Nx};
Point(8) = {0,   4, 0,  Lx/Nx};

//------------------------------------------------------------
// LINES
//------------------------------------------------------------

// Solid boundary
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

// Fluid boundary
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};

// Interface (solid/fluid boundary)
Line(9) = {4, 5};
Line(10) = {6, 3};

//------------------------------------------------------------
// CURVE LOOPS AND SURFACES
//------------------------------------------------------------

// Solid region
Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

// Fluid region
Curve Loop(2) = {5, 6, 7, 8};
Plane Surface(2) = {2};

//------------------------------------------------------------
// PHYSICAL GROUPS
//------------------------------------------------------------

// Subdomains
Physical Surface(1) = {2}; // fluid_id = 1
Physical Surface(2) = {1}; // solid_id = 2

// Interface
Physical Line(3) = {9};    // interface_id = 3

//------------------------------------------------------------
// MESH REFINEMENT
//------------------------------------------------------------

// x-direction
Transfinite Line{1,2,3,4,5,6,7,8} = Nx + 1 Using Progression 1;

// z-direction for solid
Transfinite Line{1,4} = Nz_s + 1 Using Progression 1;
Transfinite Line{2,3} = Nz_s + 1 Using Progression 1;

// z-direction for fluid
Transfinite Line{6,7} = Nz_f + 1 Using Progression 1;
Transfinite Line{5,8} = Nz_f + 1 Using Progression 1;

// Structured mesh
Transfinite Surface{1};
Transfinite Surface{2};
Recombine Surface{1};
Recombine Surface{2};
