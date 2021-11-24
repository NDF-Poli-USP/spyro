// Gmsh project created on Wed Jul 28 06:11:29 2021
//+
Point(1) = {-1, 0, 0, 1.0};
//+
Point(2) = {18, 0, 0, 1.0};
//+
Point(3) = {-1, -4.5, 0, 1.0};
//+
Point(4) = {18, -4.5, 0, 1.0};
//+
Point(5) = {17, -4.5, 0, 1.0};
//+
Point(6) = {17, 0, 0, 1.0};
//+
Point(7) = {0, 0, 0, 1.0};
//+
Point(8) = {0, -4.5, 0, 1.0};
//+
Point(9) = {0, -3.5, 0, 1.0};
//+
Point(10) = {17, -3.5, 0, 1.0};
//+
Point(11) = {-1, -3.5, 0, 1.0};
//+
Point(12) = {18, -3.5, 0, 1.0};
//+
Point(13) = {-1, -1, 0, 1.0};
//+
Point(14) = {0, -1, 0, 1.0};
//+
Point(15) = {17, -1, 0, 1.0};
//+
Point(16) = {17, -1, 0, 1.0};
//+
Point(17) = {18, -1, 0, 1.0};
//+
Line(1) = {1, 7};
//+
Line(2) = {7, 6};
//+
Line(3) = {6, 2};
//+
Line(4) = {13, 14};
//+
Line(5) = {14, 15};
//+
Line(6) = {15, 17};
//+
Line(7) = {11, 9};
//+
Line(8) = {9, 10};
//+
Line(9) = {3, 8};
//+
Line(10) = {8, 5};
//+
Line(11) = {5, 4};
//+
Line(12) = {10, 12};
//+
Line(13) = {1, 13};
//+
Line(14) = {13, 11};
//+
Line(15) = {11, 3};
//+
Line(16) = {7, 14};
//+
Line(17) = {14, 9};
//+
Line(18) = {9, 8};
//+
Line(19) = {6, 15};
//+
Line(20) = {15, 10};
//+
Line(21) = {10, 5};
//+
Line(22) = {2, 17};
//+
Line(23) = {17, 12};
//+
Line(24) = {12, 4};
//+
Curve Loop(1) = {1, 16, -4, -13};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {14, 7, -17, -4};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {15, 9, -18, -7};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {5, 20, -8, -17};
//+
Plane Surface(4) = {4};
//+
Curve Loop(5) = {2, 19, -5, -16};
//+
Plane Surface(5) = {5};
//+
Curve Loop(6) = {19, 6, -22, -3};
//+
Plane Surface(6) = {6};
//+
Curve Loop(7) = {20, 12, -23, -6};
//+
Plane Surface(7) = {7};
//+
Curve Loop(8) = {21, 11, -24, -12};
//+
Plane Surface(8) = {8};
//+
Curve Loop(9) = {8, 21, -10, -18};
//+
Plane Surface(9) = {9};
//+
Transfinite Curve {2, 5, 8, 10} = 300 Using Progression 1;
//+
Transfinite Curve {13, 16, 19, 22} = 30 Using Progression 1;
//+
Transfinite Curve {14, 17, 20, 23} = 50 Using Progression 1;
//+
Transfinite Curve {1, 4, 7, 9, 3, 6, 12, 11, 21, 24, 15, 18} = 20 Using Progression 1;
//+
Physical Curve(1) = {2, 1, 3};
//+
Physical Curve(2) = {13, 14, 15, 9, 10, 11, 24, 23, 22};
//+
Physical Surface(3) = {5, 1, 2, 4, 9, 3, 6, 7, 8};
