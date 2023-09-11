//Point(1) = {0,    -3.5,  0, 0.75/5};
//Point(2) = {17.0, -3.5,  0, 0.75/5};
//Point(3) = {17.0,  0.0,  0, 0.25/5};
//Point(4) = {0,     0.0,  0, 0.25/5};

Point(1) = {6,    -3.5,  0, 0.75/5};
Point(2) = {13.0, -3.5,  0, 0.75/5};
Point(3) = {13.0, -0.5,  0, 0.25/5};
Point(4) = {6,    -0.5,  0, 0.25/5};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Curve Loop(5) = {1, 2, 3, 4};

Plane Surface(1) = {5};

Physical Line(6) = {1};
Physical Line(7) = {2};
Physical Line(8) = {3};
Physical Line(9) = {4};

Physical Surface(1) = {1};
                             
