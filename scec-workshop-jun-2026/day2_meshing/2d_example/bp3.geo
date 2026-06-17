/**********************************************************************
 * BP3 — Gmsh 2D Fault Mesh
 * Converted from bp3 Jupyter notebook
 **********************************************************************/


/* -------------------------------------------------------------------- */
/* Parameters                                                           */
/* Tandem works in km — all lengths are in kilometers.                  */
/* -------------------------------------------------------------------- */

DefineConstant[ Lf  = {0.2,  Min 0, Max 10, Name "Fault resolution (km)" } ];
DefineConstant[ Ls  = {0.2,  Min 0, Max 10, Name "Trench resolution (km)" } ];
DefineConstant[ dip = {60.0, Min 0, Max 90, Name "Dipping angle (degrees)" } ];

dip_rad = dip * Pi / 180.0;


/* -------------------------------------------------------------------- */
/* Geometry                                                             */
/* -------------------------------------------------------------------- */
/* h:  coarse target element size at outer boundary points              */
/* d:  domain depth (y goes from 0 to -d)                              */
/* w:  horizontal half-width, computed from dip and clamped to >= d    */
/* max_depth_of_rsf_fault: down-dip depth of the RSF segment           */

h = 40;   // km
d = 400;  // km
max_depth_of_rsf_fault = 60;  // km

w = d * Cos(dip_rad) / Sin(dip_rad);
w = (w < d) ? d : w;


/* -------------------------------------------------------------------- */
/* Points                                                               */
/* addPoint(x, y, z, h)                                                 */
/*   x, y — coordinates in km                                           */
/*   z    — always 0 (2D mesh)                                          */
/*   h    — local target element size                                   */
/* -------------------------------------------------------------------- */

Point(1) = {-w,                           0,  0, h};  // pt_top_left
Point(2) = { w,                           0,  0, h};  // pt_top_right
Point(3) = { w + d * Cos(dip_rad)/Sin(dip_rad), -d, 0, h};  // pt_bottom_right
Point(4) = {-w + d * Cos(dip_rad)/Sin(dip_rad), -d, 0, h};  // pt_bottom_left
Point(5) = {     d * Cos(dip_rad)/Sin(dip_rad), -d, 0, h};  // pt_bottom_fault
Point(6) = { 0,                           0,  0, h};  // pt_trench
Point(7) = { max_depth_of_rsf_fault * Cos(dip_rad),
            -max_depth_of_rsf_fault * Sin(dip_rad),
             0, h};                                    // pt_fault_tip


/* -------------------------------------------------------------------- */
/* Lines                                                                */
/* -------------------------------------------------------------------- */

// Outer boundary
Line(1) = {1, 6};  // l_top_left     — top free surface, left of trench
Line(2) = {6, 2};  // l_top_right    — top free surface, right of trench
Line(3) = {2, 3};  // l_right        — right side boundary
Line(4) = {3, 5};  // l_bottom_right — bottom boundary, upper-plate side
Line(5) = {5, 4};  // l_bottom_left  — bottom boundary, downgoing-plate side
Line(6) = {4, 1};  // l_left         — left side boundary

// Fault
Line(7) = {6, 7};  // rsf_fault                 — rate-and-state friction segment
Line(8) = {7, 5};  // dislocation_creeping_fault — deeper creeping continuation


/* -------------------------------------------------------------------- */
/* Curve Loops & Surfaces                                               */
/* Negative tag = line traversed in reverse direction.                  */
/* -------------------------------------------------------------------- */

// loop_downgoing_plate
Curve Loop(1) = {7, 8, 5, 6, 1};
Plane Surface(1) = {1};  // downgoing_plate

// loop_upper_plate
Curve Loop(2) = {7, 8, -4, -3, -2};
Plane Surface(2) = {2};  // upper_plate


/* -------------------------------------------------------------------- */
/* Physical Groups                                                      */
/* Tags must match Tandem's expected values.                            */
/*   dim=1 (Curve): boundary lines                                      */
/*   dim=2 (Surface): domain volumes                                    */
/* The same tag number can appear for curves and surfaces without       */
/* conflict because they live in different dimensions.                  */
/* -------------------------------------------------------------------- */

Physical Curve(3) = {7};        // RSF_FAULT_TAG        — rate-and-state friction fault
Physical Curve(1) = {1, 2, 4, 5};  // ZERO_STRESS_TAG   — zero stress / free boundary
Physical Curve(5) = {3, 6, 8};  // DIRICHLET_TAG        — Dirichlet BC / dislocation

Physical Surface(1) = {1};  // downgoing_plate
Physical Surface(2) = {2};  // upper_plate


/* -------------------------------------------------------------------- */
/* Mesh Size Fields                                                     */
/* -------------------------------------------------------------------- */

fault_growth_rate  = 0.1;  // km/km — linear growth away from RSF fault
trench_growth_rate = 0.1;  // km/km — linear growth away from trench

// Field 1: distance from the RSF fault curve
Field[1] = Distance;
Field[1].CurvesList = {7};
Field[1].Sampling   = 1000;

// Field 2: Lf at the fault, growing linearly with distance
Field[2] = MathEval;
Field[2].F = Sprintf("%g + %g * F1", Lf, fault_growth_rate);

// Field 3: Ls at the trench (0,0), growing radially outward
Field[3] = MathEval;
Field[3].F = Sprintf("%g + %g * Sqrt(x^2 + y^2)", Ls, trench_growth_rate);

// Field 4: minimum of fault and trench fields
Field[4] = Min;
Field[4].FieldsList = {2, 3};

Background Field = 4;


/* -------------------------------------------------------------------- */
/* Mesh output format                                                   */
/* Version 2.2 required for Tandem compatibility.                       */
/* -------------------------------------------------------------------- */

Mesh.MshFileVersion = 2.2;
