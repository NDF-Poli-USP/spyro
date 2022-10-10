Point(1) = {0,    -15.0,  0, 0.5};
Point(2) = {35.0, -15.0,  0, 0.5};
Point(3) = {35.0,  0.0,  0, 0.5};
Point(4) = {0,     0.0,  0, 0.5};

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
