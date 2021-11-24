//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {2, 0, 0, 1.0};
//+
Point(3) = {2, -1.5, 0, 1.0};
//+
Point(4) = {0, -1.5, 0, 1.0};
//+
Point(5) = {0, -0.2, 0, 1.0};
//+
Point(6) = {2, -0.2, 0, 1.0};


//+
Point(10) = {0.5, 0., 0, 1.0};
//+
Point(11) = {1.5, 0., 0, 1.0};
//+
Point(12) = {1.5, -0.7, 0, 1.0};
//+
Point(13) = {0.5, -0.7, 0, 1.0};

//+
Point(14) = {0.5, -0.2, 0, 1.0};
//+
Point(15) = {1.5, -0.2, 0, 1.0};
//+
Point(18) = {0.5, -1.5, 0, 1.0};
//+
Point(20) = {0, -0.7, 0, 1.0};
//+
Point(21) = {2, -0.7, 0, 1.0};
//+
Line(5) = {14, 15};
//+
Line(6) = {13, 12};
//+
Line(7) = {14, 13};
//+
Line(8) = {15, 12};
//+
//+
Point(22) = {1.5, -1.5, 0, 1.0};
//+
Line(9) = {1, 10};
//+
Line(10) = {5, 14};
//+
Line(11) = {15, 6};
//+
Line(12) = {10, 11};
//+
Line(13) = {11, 2};
//+
Line(14) = {20, 13};
//+
Line(15) = {12, 21};
//+
Line(16) = {4, 18};
//+
Line(17) = {18, 22};
//+
Line(18) = {22, 3};
//+
Line(19) = {1, 5};
//+
Line(20) = {5, 20};
//+
Line(21) = {20, 4};
//+
Line(22) = {2, 6};
//+
Line(23) = {6, 21};
//+
Line(24) = {21, 3};
//+
Line(25) = {13, 18};
//+
Line(26) = {12, 22};
//+
Line(27) = {10, 14};
//+
Line(28) = {11, 15};
//+
Curve Loop(1) = {10, -27, -9, 19};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {27, 5, -28, -12};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {28, 11, -22, -13};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {20, 14, -7, -10};
//+
Plane Surface(4) = {4};
//+
Curve Loop(5) = {7, 6, -8, -5};
//+
Plane Surface(5) = {5};
//+
Curve Loop(6) = {8, 15, -23, -11};
//+
Plane Surface(6) = {6};
//+
Curve Loop(7) = {21, 16, -25, -14};
//+
Plane Surface(7) = {7};
//+
Curve Loop(8) = {25, 17, -26, -6};
//+
Plane Surface(8) = {8};
//+
Curve Loop(9) = {26, 18, -24, -15};
//+
Plane Surface(9) = {9};
//+
Transfinite Curve {5, 6, 17, 12} = 50 Using Progression 1;
//+
Transfinite Curve {21, 25, 26, 24} = 15 Using Progression 1.15;
//+
Transfinite Curve {9, 10, 14, 16} = 15 Using Progression 0.95;
//+
Transfinite Curve {18, 15, 13, 11} = 15 Using Progression 1.05;
//+
Transfinite Curve {20, 7, 8, 23} = 25 Using Progression 1;
//+
Transfinite Curve {19, 27, 28, 22} = 10 Using Progression 1;
//+
Transfinite Surface {1};
//+
Transfinite Surface {2};
//+
Transfinite Surface {3};
//+
Transfinite Surface {6};
//+
Transfinite Surface {1};
//+
Transfinite Surface {4};
//+
Transfinite Surface {7};
//+
Transfinite Surface {8};
//+
Transfinite Surface {9};
//+
Transfinite Surface {6};
//+
Transfinite Surface {5};
//+
Transfinite Surface {2};
//+
Transfinite Surface {3};
//+
Physical Curve(1) = {13, 12, 9};
//+
Physical Curve(2) = {19, 20, 21, 16, 17, 18, 24, 23, 22};
//+
Physical Surface(3) = {4, 1, 2, 5, 3, 6, 9, 8, 7};
