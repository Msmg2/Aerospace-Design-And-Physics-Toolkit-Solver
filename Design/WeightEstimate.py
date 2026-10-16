import numpy as np
import warnings

_trapz = getattr(np, "trapezoid", None) or np.trapz   # np.trapz was removed in NumPy 2.4


'''
We will estimate weight of wing like bodies, fusalage Bodies, and propulsion
I would also need to make a way to add various compnents for the final weight. the best way is to use
the Assemble function: every component function returns (mass, CG) in its own frame, and Assemble
places the components on the aircraft with a position offset and sums up the mass moments.
All units will be in SI units

Terminology for Wing Function:
S = Wing area
b = Wingspan
lmb = Taper Ratio
AR = Aspect Ratio
rho_s = Strut density
N = no of ribs
rho_rib = rib density
Pc = Perimieter of Airfoil to Chord ratio
(Note: Taper Ratio and Aspect ratio along with wing area is used to calculate root chord and tip chord)
Currently the support is for linear taper ratio (elliptical support is coming soon)
rho_sk = skin density
t = thickness of skin
t_strut = thickness of strut
t_rib = thickness of rib
Ac = Area of support beam / chord^2 (Geometric Parameter)
rho_beam = density of support beam
Twist and Sweep Unsupported Currently
Currently only 1 support beam is supported, functionality of more will be added later.

Wing function calculates Total weight, centre of gravity (origin of wing like body starts from
leading edge of root chord)
Coordinate frame
-> Positive x is in chord direction
-> Positive y is along span
-> Positive z is along height

--------------------------------------------------------------------------------
Parameters ADDED while completing the Wing function (model assumptions are
documented in the Wing docstring):
Aac        = Airfoil enclosed Area / chord^2 (Geometric Parameter, used for the
             ribs; default 0.08 ~ a 12% thick airfoil like NACA 0012)
n_struts   = number of identical spanwise struts (default 1)
x_strut_c  = strut chordwise position as a fraction of the local chord (default 0.5)
n_beams    = number of identical support beams (default 1)
x_beam_c   = support beam chordwise position as a fraction of the local chord
             (default 0.35, ~ typical main spar position)
n_sections = number of spanwise stations used for the numerical integration
             (default 2001)

b may also be passed as None, in which case it is derived from the aspect
ratio:  b = sqrt(AR * S)

Note: Wing treats b as ONE panel from root to tip. The weight is the same
whether you pass the full wing or a half-wing (S/2, b/2) twice, but y_cg is
only meaningful for a half-wing. For the aircraft, pass the right half-wing
and use mirror=True in Assemble.

--------------------------------------------------------------------------------
Terminology for Fuselage Function:
L = fuselage length
D = maximum (cylinder) diameter
Ln = nose length
Lt = tail cone length
lmb_t = tail end diameter / D
N = no of frames
rho_sk = skin density
t = thickness of skin
rho_fr = frame density
t_fr = thickness of frame (along the fuselage axis)
h_fr = radial depth of frame (h_fr >= radius gives a full bulkhead)
rho_l = longeron density
A_l = longeron cross-section area
n_long = no of longerons

Fuselage function calculates Total weight, centre of gravity (origin of the
fuselage is the nose tip)
Coordinate frame
-> Positive x is along the fuselage axis, towards the tail
-> Positive y is to the right
-> Positive z is up

--------------------------------------------------------------------------------
Assemble function sums all components in the aircraft frame (origin at the
fuselage nose tip, same axes as the Fuselage frame). Each component is given
as (mass, cg) for a point mass, (mass, cg, offset) for a component placed at
an offset, or a dict that also allows rotation (e.g. VERTICAL for a fin) and
mirror (for left/right pairs).
'''


def Wing(S, b, lmb, N, rho_sk, t, Pc, rho_rib, t_rib,
         rho_s, t_strut, rho_beam, Ac,
         AR=None, Aac=0.08, taper='linear',
         n_struts=1, x_strut_c=0.5,
         n_beams=1, x_beam_c=0.35,
         n_sections=2001, verbose=False):
    '''
    Geometric build-up estimate of the weight (mass) and centre of gravity of
    a wing-like body.

    The wing is treated as one straight panel that starts at the root
    (y = 0) and ends at the tip (y = b). The origin of the coordinate frame
    sits at the leading edge of the root chord:
        +x : chord direction (from leading edge towards trailing edge)
        +y : span direction (root -> tip)
        +z : height (the wing plane is z = 0; everything is symmetric about it)

    Implementation of the pseudo code:
        1) wing dimensions c_root, c_tip
        2) rib positions and local chord lengths
        3) skin weight and CG
        4) ribs weight and CG
        5) struts weight and CG
        6) support beam weight and CG
        7) total weight
        8) total CG
        9) return (W_wing, CG)

    Parameters (all SI units)
    -------------------------
    S         : wing area [m^2] (area of the trapezoidal planform)
    b         : span of the panel [m]; pass None to derive it from AR
    lmb       : taper ratio c_tip / c_root (> 0, linear taper)
    N         : number of ribs, uniformly spaced from root to tip (both
                ends included)
    rho_sk    : skin density [kg/m^3]
    t         : skin thickness [m]
    Pc        : airfoil perimeter / chord ratio (skin wraps this perimeter)
    rho_rib   : rib density [kg/m^3]
    t_rib     : rib thickness [m]
    rho_s     : strut density [kg/m^3]
    t_strut   : strut thickness [m]
    rho_beam  : support beam density [kg/m^3]
    Ac        : support beam cross-section area / chord^2 (geometric parameter)
    AR        : optional aspect ratio. Used to derive b when b is None, and
                otherwise checked against b^2 / S (warning on >2% mismatch)
    Aac       : airfoil enclosed area / chord^2 (geometric parameter used for
                the ribs; default 0.08 ~ NACA 0012-like airfoil)
    taper     : 'linear' (elliptical support is coming soon)
    n_struts  : number of identical spanwise struts (default 1)
    x_strut_c : strut chordwise position as a fraction of local chord (0..1)
    n_beams   : number of identical support beams (default 1)
    x_beam_c  : beam chordwise position as a fraction of local chord (0..1)
    n_sections: number of spanwise stations for the numerical integration
    verbose   : print a per-component breakdown when True

    Model assumptions
    -----------------
    * straight, untwisted, unswept panel without dihedral (see notes above)
    * skin wraps the airfoil perimeter -> mass per unit span = rho_sk * t *
      Pc * c(y); total skin weight = rho_sk * t * Pc * S  (since INT c dy = S)
    * the airfoil is assumed chordwise symmetric, so the perimeter (skin) and
      the enclosed area (ribs) have their centroid at half of the local chord;
      every component is symmetric about z = 0, hence CG_z = 0
    * rib i is a flat plate of area Aac * c_i^2 and thickness t_rib
    * a strut is a square bar of side t_strut with constant cross-section,
      running the full span at x_strut_c * c(y); n_struts identical bars
    * the support beam cross-section scales with the local chord squared,
      A_beam(y) = Ac * c(y)^2, located at x_beam_c * c(y)
    * returned "weight" is the mass in kg (densities are mass densities);
      multiply by g = 9.81 m/s^2 if a force in Newtons is required

    Example
    -------
    W, CG = Wing(S=3.0, b=5.0, lmb=0.5, N=17,
                 rho_sk=1600, t=0.0015, Pc=2.1,
                 rho_rib=1600, t_rib=0.003,
                 rho_s=2700, t_strut=0.008,
                 rho_beam=2700, Ac=0.015, verbose=True)

    Returns
    -------
    W_wing : float           total wing weight (mass) [kg]
    CG     : numpy.ndarray   [x_cg, y_cg, z_cg] in m, measured from the
                             leading edge of the root chord
    '''
    # ---------------------------- 0) input checks ---------------------------
    if S <= 0:
        raise ValueError('Wing: S must be positive')
    if b is not None and b <= 0:
        raise ValueError('Wing: b must be positive')
    if lmb <= 0:
        raise ValueError('Wing: taper ratio lmb must be positive')
    if int(N) < 1:
        raise ValueError('Wing: at least one rib is needed (N >= 1)')
    if taper != 'linear':
        # elliptical taper support is coming soon
        raise NotImplementedError("Wing: taper='%s' is not supported yet" % str(taper))
    if AR is not None and AR <= 0:
        raise ValueError('Wing: AR must be positive')
    for name, val in (('t', t), ('t_rib', t_rib), ('t_strut', t_strut),
                      ('Pc', Pc), ('Ac', Ac), ('Aac', Aac),
                      ('rho_sk', rho_sk), ('rho_rib', rho_rib),
                      ('rho_s', rho_s), ('rho_beam', rho_beam)):
        if val <= 0:
            raise ValueError('Wing: %s must be positive' % name)
    if not (0.0 <= x_strut_c <= 1.0) or not (0.0 <= x_beam_c <= 1.0):
        raise ValueError('Wing: x_strut_c and x_beam_c must be in [0, 1]')
    if int(n_sections) < 2:
        raise ValueError('Wing: n_sections must be >= 2')

    # b can be derived from the aspect ratio: AR = b^2 / S -> b = sqrt(AR*S)
    if b is None:
        if AR is None:
            raise ValueError('Wing: provide either b or AR')
        b = float(np.sqrt(AR * S))

    AR_eff = b ** 2 / S
    if AR is not None and not np.isclose(AR, AR_eff, rtol=0.02):
        warnings.warn('Wing: AR=%.4f is inconsistent with b^2/S=%.4f '
                      '(b is used as given)' % (AR, AR_eff))

    # -------------------- 1) wing dimensions: c_root, c_tip -----------------
    # trapezoidal planform:  S = (c_root + c_tip)/2 * b  and  lmb = c_tip/c_root
    #   ->  c_root = 2 S / (b (1 + lmb)),   c_tip = lmb * c_root
    c_root = 2.0 * S / (b * (1.0 + lmb))
    c_tip = lmb * c_root

    # chord distribution along the span (linear taper, elliptical coming soon)
    def chord(y_local):
        return c_root * (1.0 - (1.0 - lmb) * y_local / b)

    # -------------------- 2) rib positions & chord lengths ------------------
    y_st = np.linspace(0.0, b, int(n_sections))   # spanwise integration stations
    c_st = chord(y_st)                            # local chord at each station

    y_rib = np.linspace(0.0, b, int(N))           # ribs from root to tip
    c_rib = chord(y_rib)                          # chord length at each rib

    # ------------------------ 3) skin weight & CG ---------------------------
    # skin wraps the airfoil perimeter, perimeter(y) = Pc * c(y):
    #   W_skin = rho_sk * t * Pc * INT[0,b] c dy = rho_sk * t * Pc * S
    # mass per unit span is proportional to the local chord; the perimeter is
    # chordwise symmetric, so every strip has its centroid at c(y) / 2
    S_int = _trapz(c_st, y_st)                    # = S (exact for a trapezoid)
    W_skin = rho_sk * t * Pc * S_int
    x_cg_skin = _trapz(0.5 * c_st * c_st, y_st) / S_int
    y_cg_skin = _trapz(y_st * c_st, y_st) / S_int

    # ------------------------ 4) ribs weight & CG ---------------------------
    # rib i is a flat plate of area Aac * c_i^2 (airfoil enclosed area) and
    # thickness t_rib  ->  W_i = rho_rib * t_rib * Aac * c_i^2
    W_ribs = rho_rib * t_rib * Aac * np.sum(c_rib ** 2)
    x_cg_ribs = np.sum(0.5 * c_rib * c_rib ** 2) / np.sum(c_rib ** 2)
    y_cg_ribs = np.sum(y_rib * c_rib ** 2) / np.sum(c_rib ** 2)

    # ------------------------ 5) struts weight & CG -------------------------
    # a strut is a square bar (side t_strut) of constant cross-section running
    # the full span, so its mass per unit length is constant
    W_struts = n_struts * rho_s * t_strut ** 2 * b
    y_cg_struts = 0.5 * b
    x_cg_struts = x_strut_c * S_int / b           # mean local chord * fraction

    # --------------------- 6) support beam weight & CG ----------------------
    # beam cross-section scales with local chord^2: A_beam(y) = Ac * c(y)^2,
    # located at x_beam_c * c(y); mass per unit span ~ c(y)^2
    I2 = _trapz(c_st ** 2, y_st)                  # INT c^2 dy
    W_beam = n_beams * rho_beam * Ac * I2
    y_cg_beam = _trapz(y_st * c_st ** 2, y_st) / I2
    x_cg_beam = x_beam_c * _trapz(c_st ** 3, y_st) / I2

    # ------------------------ 7) total weight -------------------------------
    components = {
        'skin':   (W_skin,   x_cg_skin,   y_cg_skin),
        'ribs':   (W_ribs,   x_cg_ribs,   y_cg_ribs),
        'struts': (W_struts, x_cg_struts, y_cg_struts),
        'beam':   (W_beam,   x_cg_beam,   y_cg_beam),
    }
    W_wing = sum(comp[0] for comp in components.values())

    # ------------------------ 8) total CG -----------------------------------
    # z = 0 for every component (symmetric airfoil, no dihedral / twist / sweep)
    x_cg = sum(comp[0] * comp[1] for comp in components.values()) / W_wing
    y_cg = sum(comp[0] * comp[2] for comp in components.values()) / W_wing
    CG = np.array([x_cg, y_cg, 0.0])

    # ------------------------ 9) return -------------------------------------
    if verbose:
        print('Wing weight breakdown (origin: leading edge of root chord)')
        print('  c_root = %.4f m   c_tip = %.4f m   AR = %.3f'
              % (c_root, c_tip, AR_eff))
        for name, (w, xc, yc) in components.items():
            print('  %-7s W = %10.4f kg   CG = (%8.4f, %8.4f, %6.2f) m'
                  % (name, w, xc, yc, 0.0))
        print('  %-7s W = %10.4f kg   CG = (%8.4f, %8.4f, %6.2f) m'
              % ('TOTAL', W_wing, x_cg, y_cg, 0.0))

    return W_wing, CG


def FuselageRadius(x, L, D, Ln, Lt, lmb_t=0.0, nose='parabolic'):
    '''
    Radius r(x) and slope dr/dx of the Fuselage body of revolution at the
    axial positions x (origin at the nose tip). Same shape and parameters as
    Fuselage() (see there), which does the input checks; also handy to draw
    the fuselage outline.

    Returns
    -------
    r, dr : numpy.ndarray   radius [m] and slope dr/dx [-] at each x
    '''
    R = 0.5 * D
    x_tail = L - Lt                               # start of the tail cone
    x = np.asarray(x, dtype=float)
    r = np.full_like(x, R)
    dr = np.zeros_like(x)

    # nose: parabolic r = R (1 - (1 - s)^2) has zero slope at s = 1
    s = np.clip(x / Ln, 0.0, 1.0)
    if nose == 'parabolic':
        r_n = R * (1.0 - (1.0 - s) ** 2)
        dr_n = 2.0 * R * (1.0 - s) / Ln
    else:
        r_n = R * s
        dr_n = np.full_like(x, R / Ln)
    in_nose = x < Ln
    r = np.where(in_nose, r_n, r)
    dr = np.where(in_nose, dr_n, dr)

    # tail: linear cone from R to lmb_t * R
    if Lt > 0:
        u = np.clip((x - x_tail) / Lt, 0.0, 1.0)
        in_tail = x > x_tail
        r = np.where(in_tail, R * (1.0 - (1.0 - lmb_t) * u), r)
        dr = np.where(in_tail, -R * (1.0 - lmb_t) / Lt, dr)
    return r, dr


def Fuselage(L, D, Ln, Lt, N, rho_sk, t, rho_fr, t_fr, h_fr,
             rho_l, A_l,
             lmb_t=0.0, nose='parabolic', n_long=4,
             n_sections=2001, verbose=False):
    '''
    Geometric build-up estimate of the weight (mass) and centre of gravity of
    a fuselage-like body (body of revolution with a circular cross-section).

    The fuselage axis is the x axis. The origin of the coordinate frame sits
    at the nose tip:
        +x : along the fuselage axis (from nose towards tail)
        +y : to the right (starboard)
        +z : up

    The radius distribution r(x) is made of three sections:
        nose     0 <= x <= Ln         : grows from 0 to R = D/2
        mid      Ln <= x <= L - Lt    : cylinder of radius R
        tail     L - Lt <= x <= L     : linear cone from R to lmb_t * R

    Implementation of the pseudo code:
        1) radius distribution r(x) and its slope dr/dx
        2) frame positions and local radii
        3) skin weight and CG
        4) frames weight and CG
        5) longerons weight and CG
        6) total weight
        7) total CG
        8) return (W_fus, CG)

    Parameters (all SI units)
    -------------------------
    L         : total fuselage length [m]
    D         : maximum (cylinder) diameter [m]
    Ln        : nose length [m] (> 0)
    Lt        : tail cone length [m] (>= 0, Ln + Lt <= L)
    N         : number of frames, uniformly spaced along the fuselage with
                both ends excluded (x_i = i L / (N+1)); N = 0 is allowed
    rho_sk    : skin density [kg/m^3]
    t         : skin thickness [m]
    rho_fr    : frame density [kg/m^3]
    t_fr      : frame thickness (along x) [m]
    h_fr      : frame radial depth [m]; a frame with h_fr >= r is a full
                bulkhead (solid disc)
    rho_l     : longeron density [kg/m^3]
    A_l       : longeron cross-section area [m^2]
    lmb_t     : tail end diameter / D (0 = pointed tail, default 0)
    nose      : 'parabolic' (default, smooth junction with the cylinder) or
                'conical'
    n_long    : number of identical longerons, evenly spaced around the
                circumference starting from the top (default 4, 0 allowed)
    n_sections: number of axial stations for the numerical integration
    verbose   : print a per-component breakdown when True

    Model assumptions
    -----------------
    * straight fuselage axis, circular cross-sections, no cut-outs
    * skin covers the lateral surface only (no end caps):
        mass per unit length = rho_sk * t * 2 pi r(x) * sqrt(1 + r'(x)^2)
    * frame i is an annular ring of outer radius r_i, inner radius
      max(r_i - h_fr, 0) and thickness t_fr
    * longerons follow the skin from nose to tail with constant cross-section
      A_l; their length includes the surface slope (arc length)
    * everything is axisymmetric about the x axis, so CG_y = CG_z = 0 except
      for the longerons when n_long = 1 (a single longeron sits on top)
    * returned "weight" is the mass in kg (densities are mass densities);
      multiply by g = 9.81 m/s^2 if a force in Newtons is required

    Example
    -------
    W, CG = Fuselage(L=2.5, D=0.3, Ln=0.4, Lt=0.9, N=8,
                     rho_sk=1600, t=0.0008,
                     rho_fr=1600, t_fr=0.003, h_fr=0.02,
                     rho_l=1600, A_l=4e-5, lmb_t=0.2, verbose=True)

    Returns
    -------
    W_fus : float           total fuselage weight (mass) [kg]
    CG    : numpy.ndarray   [x_cg, y_cg, z_cg] in m, measured from the nose tip
    '''
    # ---------------------------- 0) input checks ---------------------------
    if L <= 0 or D <= 0:
        raise ValueError('Fuselage: L and D must be positive')
    if Ln <= 0:
        raise ValueError('Fuselage: nose length Ln must be positive')
    if Lt < 0:
        raise ValueError('Fuselage: tail length Lt must be >= 0')
    if Ln + Lt > L:
        raise ValueError('Fuselage: Ln + Lt must not exceed L')
    if not (0.0 <= lmb_t <= 1.0):
        raise ValueError('Fuselage: lmb_t must be in [0, 1]')
    if int(N) < 0:
        raise ValueError('Fuselage: N must be >= 0')
    if int(n_long) < 0:
        raise ValueError('Fuselage: n_long must be >= 0')
    if nose not in ('parabolic', 'conical'):
        raise NotImplementedError("Fuselage: nose='%s' is not supported yet" % str(nose))
    checks = [('t', t), ('t_fr', t_fr), ('h_fr', h_fr),
              ('rho_sk', rho_sk), ('rho_fr', rho_fr)]
    if int(n_long) > 0:
        checks += [('rho_l', rho_l), ('A_l', A_l)]
    for name, val in checks:
        if val <= 0:
            raise ValueError('Fuselage: %s must be positive' % name)
    if int(n_sections) < 2:
        raise ValueError('Fuselage: n_sections must be >= 2')

    N = int(N)
    n_long = int(n_long)
    R = 0.5 * D

    # ------------------ 1) radius distribution r(x), dr/dx ------------------
    def radius(x):
        return FuselageRadius(x, L, D, Ln, Lt, lmb_t, nose)

    # ------------------- 2) frame positions & local radii -------------------
    x_st = np.linspace(0.0, L, int(n_sections))   # axial integration stations
    r_st, dr_st = radius(x_st)
    ds_st = np.sqrt(1.0 + dr_st ** 2)             # surface arc length per dx

    x_fr = np.linspace(0.0, L, N + 2)[1:-1]       # frames, ends excluded
    r_fr, _ = radius(x_fr)

    # ------------------------ 3) skin weight & CG ---------------------------
    # wetted (lateral) area element: dA = 2 pi r sqrt(1 + r'^2) dx
    dA_st = 2.0 * np.pi * r_st * ds_st
    A_wet = _trapz(dA_st, x_st)
    W_skin = rho_sk * t * A_wet
    x_cg_skin = _trapz(x_st * dA_st, x_st) / A_wet

    # ------------------------ 4) frames weight & CG -------------------------
    # annular ring: area = pi (r^2 - r_in^2), r_in = max(r - h_fr, 0)
    r_in = np.maximum(r_fr - h_fr, 0.0)
    A_fr = np.pi * (r_fr ** 2 - r_in ** 2)
    W_frames = rho_fr * t_fr * np.sum(A_fr)
    x_cg_frames = np.sum(x_fr * A_fr) / np.sum(A_fr) if N > 0 else 0.0

    # ----------------------- 5) longerons weight & CG -----------------------
    # each longeron follows the skin, so its length is the surface arc length;
    # angular positions theta_k measured from +z towards +y
    L_arc = _trapz(ds_st, x_st)
    W_long = n_long * rho_l * A_l * L_arc
    if n_long > 0:
        x_cg_long = _trapz(x_st * ds_st, x_st) / L_arc
        r_cg_long = _trapz(r_st * ds_st, x_st) / L_arc
        theta = 2.0 * np.pi * np.arange(n_long) / n_long
        y_cg_long = r_cg_long * np.mean(np.sin(theta))
        z_cg_long = r_cg_long * np.mean(np.cos(theta))
        # evenly spaced longerons (n_long >= 2) cancel out exactly
        y_cg_long = 0.0 if abs(y_cg_long) < 1e-12 * R else y_cg_long
        z_cg_long = 0.0 if abs(z_cg_long) < 1e-12 * R else z_cg_long
    else:
        x_cg_long = y_cg_long = z_cg_long = 0.0

    # ------------------------ 6) total weight -------------------------------
    components = {
        'skin':      (W_skin,   x_cg_skin,   0.0,       0.0),
        'frames':    (W_frames, x_cg_frames, 0.0,       0.0),
        'longerons': (W_long,   x_cg_long,   y_cg_long, z_cg_long),
    }
    W_fus = sum(comp[0] for comp in components.values())

    # ------------------------ 7) total CG -----------------------------------
    x_cg = sum(comp[0] * comp[1] for comp in components.values()) / W_fus
    y_cg = sum(comp[0] * comp[2] for comp in components.values()) / W_fus
    z_cg = sum(comp[0] * comp[3] for comp in components.values()) / W_fus
    CG = np.array([x_cg, y_cg, z_cg])

    # ------------------------ 8) return -------------------------------------
    if verbose:
        print('Fuselage weight breakdown (origin: nose tip)')
        print('  L = %.4f m   D = %.4f m   L/D = %.3f   A_wet = %.4f m^2'
              % (L, D, L / D, A_wet))
        for name, (w, xc, yc, zc) in components.items():
            print('  %-9s W = %10.4f kg   CG = (%8.4f, %8.4f, %8.4f) m'
                  % (name, w, xc, yc, zc))
        print('  %-9s W = %10.4f kg   CG = (%8.4f, %8.4f, %8.4f) m'
              % ('TOTAL', W_fus, x_cg, y_cg, z_cg))

    return W_fus, CG


# rotation matrix for a wing-like body used as a vertical fin: the local span
# axis (+y) is mapped to aircraft +z, the local +z to aircraft -y
VERTICAL = np.array([[1.0, 0.0, 0.0],
                     [0.0, 0.0, -1.0],
                     [0.0, 1.0, 0.0]])


def Assemble(components, verbose=False, breakdown=False):
    '''
    Total weight (mass) and centre of gravity of the whole aircraft, from the
    individual components.

    Aircraft frame (same orientation as the Fuselage frame):
        origin at the fuselage nose tip
        +x : aft along the fuselage axis
        +y : to the right (starboard)
        +z : up

    Every component function (Wing, Fuselage, ...) returns its CG in its own
    frame. Assemble places each component on the aircraft:
        p = offset + rotation @ cg
    and sums the mass moments:
        W = SUM m_i,   CG = SUM m_i p_i / W

    Parameters
    ----------
    components : dict  name -> entry, where entry is one of
        (mass, cg)              cg already in the aircraft frame (point mass)
        (mass, cg, offset)      cg in the component frame; offset = position
                                of the component origin in the aircraft frame
        dict(mass=..., cg=..., offset=(0, 0, 0), rotation=None, mirror=False)
            rotation : 3x3 matrix from the component frame to the aircraft
                       frame (None = same orientation). Use VERTICAL for a
                       fin built with Wing(); any linear map works, e.g. a
                       shear for a swept wing.
            mirror   : True when the entry is one side only (e.g. the right
                       wing panel from Wing() with S/2 and b/2); a mirror
                       image about the x-z plane (y -> -y) is added, which
                       doubles the mass and puts the pair's CG at y = 0
    verbose    : print a per-component breakdown when True
    breakdown  : also return the placed components when True

    Example
    -------
    W_w, CG_w = Wing(S=1.5, b=2.5, ...)         # right half-wing
    W_f, CG_f = Fuselage(L=2.5, D=0.3, ...)
    W, CG = Assemble({
        'fuselage': (W_f, CG_f),
        'wing':     dict(mass=W_w, cg=CG_w, offset=(0.9, 0.0, 0.1), mirror=True),
        'battery':  (3.0, (0.7, 0.0, 0.0)),
    }, verbose=True)

    Returns
    -------
    W_total : float           total weight (mass) [kg]
    CG      : numpy.ndarray   [x_cg, y_cg, z_cg] in m, aircraft frame
    rows    : list of (name, mass, position) for every placed component,
              mirrored sides included (only when breakdown=True)
    '''
    rows = []
    for name, entry in components.items():
        if isinstance(entry, dict):
            unknown = set(entry) - {'mass', 'cg', 'offset', 'rotation', 'mirror'}
            if unknown:
                raise ValueError('Assemble: %s has unknown keys %s'
                                 % (name, sorted(unknown)))
            mass = entry['mass']
            cg = entry['cg']
            offset = entry.get('offset', (0.0, 0.0, 0.0))
            rotation = entry.get('rotation', None)
            mirror = entry.get('mirror', False)
        elif len(entry) in (2, 3):
            mass, cg = entry[0], entry[1]
            offset = entry[2] if len(entry) == 3 else (0.0, 0.0, 0.0)
            rotation = None
            mirror = False
        else:
            raise ValueError('Assemble: %s must be (mass, cg), '
                             '(mass, cg, offset) or a dict' % name)

        mass = float(mass)
        cg = np.asarray(cg, dtype=float).reshape(3)
        offset = np.asarray(offset, dtype=float).reshape(3)
        if mass < 0:
            raise ValueError('Assemble: %s has a negative mass' % name)
        if rotation is not None:
            cg = np.asarray(rotation, dtype=float).reshape(3, 3) @ cg

        p = offset + cg
        if mirror:
            rows.append((name + ' (R)', mass, p))
            rows.append((name + ' (L)', mass, p * np.array([1.0, -1.0, 1.0])))
        else:
            rows.append((name, mass, p))

    W_total = sum(m for _, m, _ in rows)
    if W_total <= 0:
        raise ValueError('Assemble: total mass must be positive')
    CG = sum(m * p for _, m, p in rows) / W_total

    if verbose:
        print('Aircraft weight breakdown (origin: fuselage nose tip)')
        for name, m, p in rows:
            print('  %-16s W = %9.4f kg (%5.1f%%)   at (%8.4f, %8.4f, %8.4f) m'
                  % (name, m, 100.0 * m / W_total, p[0], p[1], p[2]))
        print('  %-16s W = %9.4f kg (100.0%%)   CG (%8.4f, %8.4f, %8.4f) m'
              % ('TOTAL', W_total, CG[0], CG[1], CG[2]))

    if breakdown:
        return W_total, CG, rows
    return W_total, CG


if __name__ == '__main__':
    # Example: tapered UAV wing panel, S = 3 m^2, b = 5 m (AR = 8.33), 17 ribs
    W_wing, CG_wing = Wing(S=3.0, b=5.0, lmb=0.5, N=17,
                           rho_sk=1600.0, t=0.0015, Pc=2.1,
                           rho_rib=1600.0, t_rib=0.003,
                           rho_s=2700.0, t_strut=0.008,
                           rho_beam=2700.0, Ac=0.015,
                           verbose=True)
    print('\nW_wing = %.3f kg   CG = %s m'
          % (W_wing, np.array2string(CG_wing, precision=4)))

    # Example: UAV fuselage, L = 2.5 m, D = 0.3 m, 8 frames, 4 longerons
    print()
    W_fus, CG_fus = Fuselage(L=2.5, D=0.3, Ln=0.4, Lt=0.9, N=8,
                             rho_sk=1600.0, t=0.0008,
                             rho_fr=1600.0, t_fr=0.003, h_fr=0.02,
                             rho_l=1600.0, A_l=4e-5, lmb_t=0.2,
                             verbose=True)

    # Example: whole aircraft. Wing-like bodies are built as the right half
    # (S/2, b/2) and mirrored; the fin is rotated so its span points up.
    # Ac = 0.001 is the beam material area (tube / I-section), not solid.
    print()
    wing_materials = dict(rho_sk=1600.0, t=0.001, Pc=2.1,
                          rho_rib=1600.0, t_rib=0.002,
                          rho_s=2700.0, t_strut=0.005,
                          rho_beam=2700.0, Ac=0.001)
    W_rw, CG_rw = Wing(S=1.5, b=2.5, lmb=0.5, N=9, **wing_materials)
    W_ht, CG_ht = Wing(S=0.2, b=0.5, lmb=0.6, N=4, **wing_materials)
    W_vt, CG_vt = Wing(S=0.15, b=0.4, lmb=0.5, N=4, **wing_materials)

    W_ac, CG_ac = Assemble({
        'fuselage': (W_fus, CG_fus),
        'wing':     dict(mass=W_rw, cg=CG_rw, offset=(0.9, 0.0, 0.1), mirror=True),
        'h-tail':   dict(mass=W_ht, cg=CG_ht, offset=(2.15, 0.0, 0.05), mirror=True),
        'fin':      dict(mass=W_vt, cg=CG_vt, offset=(2.1, 0.0, 0.1), rotation=VERTICAL),
        'motor':    (0.8, (0.05, 0.0, 0.0)),
        'battery':  (3.0, (0.7, 0.0, -0.03)),
        'payload':  (2.0, (1.0, 0.0, -0.05)),
    }, verbose=True)