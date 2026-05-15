/**********************************************************************
 * Gmsh script with additional explanations
 **********************************************************************/

/* -------------------------------------------------------------------- */
/* Define user-adjustable constants (parameters)                        */
/* -------------------------------------------------------------------- */
/* Lf: resolution (base size) around the fault plane                    */
/* Ls: resolution (base size) near the free surface                     */
/* dip: fault dipping angle in degrees                                  */
DefineConstant[ Lf = {0.020, Min 0, Max 10, Name "Fault resolution" } ];
DefineConstant[ Ls = {0.020, Min 0, Max 10, Name "Resolution near free surface" } ];
DefineConstant[ dip = {60,   Min 0, Max 90, Name "Dipping angle" } ];

/* Convert the dipping angle from degrees to radians */
dip_rad = dip * Pi / 180.0;

/* -------------------------------------------------------------------- */
/* Basic geometric parameters                                           */
/* -------------------------------------------------------------------- */
/* h: The characteristic length (max size) for outer (boundary) points  */
/* d: main length dimension in the negative y direction                 */
h = 40;
d = 400;

/* Compute the horizontal extent w based on the dip angle.              */
/* w is chosen so that the domain extends sufficiently in the x-dir.    */
w = d * Cos(dip_rad) / Sin(dip_rad);
w = w < d ? d : w;

/* Distances d1, d2, d3, d4 are used for intermediate points along the fault */
d1 = 15;
d2 = 16;
d3 = 18;
d4 = 40;

/* -------------------------------------------------------------------- */
/* Define points in 2D (x,y,z) with a characteristic length h           */
/* -------------------------------------------------------------------- */
/* Note: z=0 (2D plane), h is used as the max local element size for    */
/*       these points                                                   */
Point(1)  = {-w,  0, 0, h};   // Leftmost top corner
Point(2)  = { w,  0, 0, h};   // Rightmost top corner
Point(3)  = { w + d * Cos(dip_rad) / Sin(dip_rad),  -d, 0, h};
Point(4)  = {-w + d * Cos(dip_rad) / Sin(dip_rad),  -d, 0, h};
Point(5)  = { d * Cos(dip_rad) / Sin(dip_rad),      -d, 0, h};
Point(6)  = { 0,  0, 0, h};    // Center top // trench
Point(7)  = { d4 * Cos(dip_rad),    -d4 * Sin(dip_rad),    0, h};
Point(8)  = { d3 * Cos(dip_rad),    -d3 * Sin(dip_rad),    0, h};
Point(9)  = { d2 * Cos(dip_rad),    -d2 * Sin(dip_rad),    0, h};
Point(10) = { d1 * Cos(dip_rad),    -d1 * Sin(dip_rad),    0, h};

/* -------------------------------------------------------------------- */
/* Connect the points with lines                                        */
/* -------------------------------------------------------------------- */
/* Outer boundary lines:                                                */

/*
   Line(1): from left top corner (Point(1)) to center top (Point(6)).
   Top-left boundary segment.
*/
Line(1) = {1, 6};

/*
   Line(2): from center top (Point(6)) to right top corner (Point(2)).
   Top-right boundary segment.
*/
Line(2) = {6, 2};

/*
   Line(3): from right top corner (Point(2)) down to bottom right corner (Point(3)).
   Right boundary.
*/
Line(3) = {2, 3};

/*
   Line(4): from bottom right corner (Point(3)) to intersection near bottom (Point(5)).
   Part of bottom boundary on the right side.
*/
Line(4) = {3, 5};

/*
   Line(5): from intersection near bottom (Point(5)) to bottom left corner (Point(4)).
   Bottom boundary across the center.
*/
Line(5) = {5, 4};

/*
   Line(6): from bottom left corner (Point(4)) back to left top corner (Point(1)).
   Left boundary.
*/
Line(6) = {4, 1};


/* Fault lines (inside the domain):                                     */

/*
   Line(7): from fault point (Point(7)) to intersection near bottom (Point(5)).
   Internal fault segment.
*/
Line(7)  = {7, 5};

/*
   Line(8): from fault point (Point(8)) to fault point (Point(7)).
   Internal fault segment.
*/
Line(8)  = {8, 7};

/*
   Line(9): from fault point (Point(9)) to fault point (Point(8)).
   Internal fault segment.
*/
Line(9)  = {9, 8};

/*
   Line(10): from fault point (Point(10)) to fault point (Point(9)).
   Internal fault segment.
*/
Line(10) = {10, 9};

/*
   Line(11): from center top (Point(6)) to fault point (Point(10)).
   Internal fault segment connecting the top center to the fault.
*/
Line(11) = {6, 10};



/* -------------------------------------------------------------------- */
/* Define curve loops (closed loops) to form surfaces                   */
/* -------------------------------------------------------------------- */
/* Two surfaces:                                                        */
/*   1. Loop(1) = lines (11,10,9,8,7,-4,-3,-2)                           */
/*   2. Loop(2) = lines (11,10,9,8,7,5,6,1)                              */
Curve Loop(1) = {11, 10, 9, 8, 7, -4, -3, -2};
Curve Loop(2) = {11, 10, 9, 8, 7, 5, 6, 1};

/* Create the plane surfaces from these loops.                          */
Plane Surface(1) = {1};
Plane Surface(2) = {2};

/* -------------------------------------------------------------------- */
/* Physical groups: define boundary conditions / markers                */
/* -------------------------------------------------------------------- */
Physical Curve(3) = {8, 9, 10, 11};  // e.g., Tandem uses Physical Curve 3 to mark RSF fault elements
Physical Curve(1) = {1, 2, 4, 5};    // e.g., Tandem uses Physical Curve 1 to mark zero stress BD
Physical Curve(5) = {3, 6, 7};       // e.g., Tandem uses Physical Curve 5 to mark direchelt BD . Note that internal boundaries are treated as dislocations

/* Tag the plane surfaces */
Physical Surface(1) = {1};
Physical Surface(2) = {2};

/* -------------------------------------------------------------------- */
/* Define mesh size fields                                              */
/* -------------------------------------------------------------------- */

/* Field[1]: A MathEval expression that refines the mesh near the fault.
   ---------------------------------------------------------------
   Equation structure:
     Field[1].F = Lf
                  + 3e-2 * (x + y*(cos(dip_rad)/sin(dip_rad)))^2
                  + 2e-3 * (min(0, y/sin(dip_rad)+40))^2

   1) Base term:        Lf
      - The constant fault resolution (Fault resolution) you chose.

   2) The term:         3e-2 * (x + y*(cos(dip_rad)/sin(dip_rad)))^2
      - "x + y*(cos(dip_rad)/sin(dip_rad))" is essentially a coordinate
        transformation that measures distance parallel to the dip.
        refines the mesh near that “fault line”


   3) The term:         2e-3 * (min(0, y/sin(dip_rad)+40))^2
      - The expression "min(0, y/sin(dip_rad) + 40)" is zero or negative
        depending on y. Above a certain level, it becomes 0.
        Below that level, it goes negative, increasing this squared term.
      - This effectively makes the mesh size smaller shallower in the domain,

*/
Field[1] = MathEval;
Field[1].F = Sprintf(
  "%g + 3e-2*(x + y*%g)^2 + 2e-3*(min(0, y/%g+40))^2",
   Lf, Cos(dip_rad)/Sin(dip_rad), Sin(dip_rad)
);

/* Field[2]: Another MathEval that grows with the radial distance from (0,0).
   Here, "Ls" is the base resolution near the surface, and we add
   "0.1 * sqrt(x^2 + y^2)" to refine more near (0,0).
*/
Field[2] = MathEval;
Field[2].F = Sprintf(
  "%g + 0.1 * sqrt(x^2 + y^2)",
   Ls
);

/* Field[3]: Takes the minimum of Field[1] and Field[2], ensuring
   the mesh is as fine as dictated by either condition.
*/
Field[3] = Min;
Field[3].FieldsList = {1, 2};

/* Activate Field[3] as the global background field for meshing. */
Background Field = 3;

/* Set the output mesh version (2.2 is more widely supported). */
Mesh.MshFileVersion = 2.2;
