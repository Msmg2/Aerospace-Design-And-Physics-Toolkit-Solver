import numpy as np
import warnings

_trapz = getattr(np, "trapezoid", np.trapz)


'''
We will estimate weight of wing like bodies, fusalage Bodies, and propulsion
I would also need to make a way to add various compnents for the final weight. the best way is to use
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


def Fuselage():
    '''
    Fuselage weight and CG estimation.
    TODO: to be implemented (kept as a placeholder so the module stays valid).
    '''
    pass


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