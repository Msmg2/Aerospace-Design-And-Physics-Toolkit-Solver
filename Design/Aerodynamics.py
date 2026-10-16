import warnings
import numpy as np

'''
We will estimate the aerodynamics of wing like bodies here. Fuselage
aerodynamics will be added to this same module later.
All units are SI, except angles, which are given in degrees at the interface
(radians are used internally). Lift slopes are per radian.

Terminology for wing-like bodies (WingAero and the three methods):
S        = Wing area, both sides (reference area, projected planform)
b        = Wingspan, tip to tip (projected)
lmb      = Taper Ratio c_tip / c_root (linear taper)
AR       = Aspect Ratio = b^2 / S
sweep    = leading edge sweep [deg]. 0 gives the same planform as Wing() in
           WeightEstimate (straight leading edge)
dihedral = dihedral angle [deg]
twist    = tip twist relative to the root [deg], linear along the span
           (negative = washout)
alpha    = angle of attack of the root chord [deg]
V        = flight speed [m/s]
h        = altitude [m] (International Standard Atmosphere, see ISA)
airfoil  = section data from Airfoil() (simple parameters) or AirfoilTable()
           (tabulated polars, placeholder for now)

Coordinate frame (same as Wing in WeightEstimate): origin at the leading edge
of the root chord
-> Positive x is in chord direction (aft)
-> Positive y is along span (right)
-> Positive z is up

Methods (choose with method= in WingAero):
'lifting_line'   Prandtl lifting line (Glauert series). Straight wings with
                 taper and twist; sweep and dihedral are ignored.
'semi_empirical' DATCOM / Helmbold lift slope, Nita-Scholz span efficiency,
                 Schrenk lift distribution. Handles sweep, quick.
'vlm'            Vortex lattice (horseshoe vortices). Handles taper, twist,
                 sweep and dihedral.
Every method is linear in alpha (no stall model). CL_max is estimated with
the critical section method: the wing stalls when the local cl first reaches
the airfoil Cl_max somewhere along the span.

Parasite drag (all methods): skin friction x form factor x wetted area
(Raymer build-up), see ParasiteDrag.

Compressibility: Prandtl-Glauert (lifting line, semi-empirical) and the
Goethert rule (vlm). Linear theory, only valid well below M = 0.7.
'''


# ------------------------------------------------------------------------------
# Atmosphere
# ------------------------------------------------------------------------------
def ISA(h):
    '''
    International Standard Atmosphere, troposphere and lower stratosphere
    (0 <= h <= 20000 m, geopotential altitude).

    Returns
    -------
    T [K], p [Pa], rho [kg/m^3], mu [Pa s] (Sutherland), a [m/s]
    '''
    if not (0.0 <= h <= 20000.0):
        raise ValueError('ISA: altitude must be between 0 and 20000 m')
    g0, R, gamma = 9.80665, 287.05287, 1.4
    if h <= 11000.0:
        T = 288.15 - 0.0065 * h
        p = 101325.0 * (T / 288.15) ** (g0 / (0.0065 * R))
    else:
        T = 216.65
        p = 22632.06 * np.exp(-g0 * (h - 11000.0) / (R * T))
    rho = p / (R * T)
    mu = 1.458e-6 * T ** 1.5 / (T + 110.4)
    a = np.sqrt(gamma * R * T)
    return T, p, rho, mu, a


# ------------------------------------------------------------------------------
# Airfoil (section) data
# ------------------------------------------------------------------------------
def Airfoil(a0=2.0 * np.pi, alpha_L0=0.0, Cm0=0.0, Cl_max=1.4,
            tc=0.12, x_tc=0.3, Cd0=0.01, Cd1=0.0, Cd2=0.5, name='simple'):
    '''
    Section data from simple parameters (linear lift curve).

    Parameters
    ----------
    a0       : section lift slope [1/rad] (thin airfoil theory: 2 pi)
    alpha_L0 : zero-lift angle [deg] (negative for positive camber)
    Cm0      : pitching moment coefficient about the quarter chord
    Cl_max   : maximum section lift coefficient
    tc       : thickness / chord
    x_tc     : chordwise position of max thickness / chord
    Cd0, Cd1, Cd2 : section drag polynomial Cd = Cd0 + Cd1 alpha + Cd2 alpha^2
               (alpha in rad); used by the rotor code (BEM.py), the wing
               methods use a skin friction build-up instead
    name     : label only

    Returns
    -------
    dict with kind = 'simple' and the parameters above
    '''
    for key, val in (('a0', a0), ('Cl_max', Cl_max), ('tc', tc)):
        if val <= 0:
            raise ValueError('Airfoil: %s must be positive' % key)
    if not (0.0 < x_tc < 1.0):
        raise ValueError('Airfoil: x_tc must be in (0, 1)')
    if Cd0 < 0 or Cd2 < 0:
        raise ValueError('Airfoil: Cd0 and Cd2 must be >= 0')
    return dict(kind='simple', name=name, a0=float(a0), alpha_L0=float(alpha_L0),
                Cm0=float(Cm0), Cl_max=float(Cl_max), tc=float(tc), x_tc=float(x_tc),
                Cd0=float(Cd0), Cd1=float(Cd1), Cd2=float(Cd2))


def AirfoilTable(alpha, Re, Cl, Cd, Cm, tc=0.12, x_tc=0.3, name='table'):
    '''
    PLACEHOLDER: section data from tabulated polars.

    The arrays are checked and stored, but the wing methods cannot use them
    yet (_section raises NotImplementedError for kind = 'table').

    Parameters
    ----------
    alpha        : angles of attack [deg], shape (n_alpha,)
    Re           : Reynolds numbers, shape (n_Re,)
    Cl, Cd, Cm   : coefficients, shape (n_Re, n_alpha)
    tc, x_tc     : thickness / chord and its chordwise position
    '''
    alpha = np.asarray(alpha, dtype=float)
    Re = np.atleast_1d(np.asarray(Re, dtype=float))
    shape = (Re.size, alpha.size)
    tables = {}
    for key, arr in (('Cl', Cl), ('Cd', Cd), ('Cm', Cm)):
        arr = np.asarray(arr, dtype=float)
        if arr.size != Re.size * alpha.size:
            raise ValueError('AirfoilTable: %s must have shape (n_Re, n_alpha) = %s'
                             % (key, shape))
        tables[key] = arr.reshape(shape)
    return dict(kind='table', name=name, alpha=alpha, Re=Re, tc=float(tc),
                x_tc=float(x_tc), **tables)


def LoadPolar(path):
    '''
    PLACEHOLDER: read a polar file (e.g. XFOIL output: alpha, Cl, Cd, Cdp, Cm)
    and return AirfoilTable(...).
    '''
    raise NotImplementedError('LoadPolar: reading polar files is not implemented yet')


def _section(airfoil, Re):
    '''
    Linear section properties used by every wing method:
        a0 [1/rad], alpha_L0 [rad], Cm0, Cl_max
    This is the single place where tabulated data has to be plugged in.
    '''
    if airfoil is None:
        airfoil = Airfoil()
    if airfoil['kind'] == 'simple':
        return (airfoil['a0'], np.deg2rad(airfoil['alpha_L0']),
                airfoil['Cm0'], airfoil['Cl_max'])
    if airfoil['kind'] == 'table':
        # TODO: interpolate the polars at Re and fit the linear part:
        #   a0 = dCl/dalpha, alpha_L0 = alpha at Cl = 0,
        #   Cm0 = Cm at alpha_L0, Cl_max = max(Cl)
        # and also return the profile drag Cd(Cl) for the drag polar
        raise NotImplementedError('tabulated airfoil data is not supported yet; '
                                  'use Airfoil() for now')
    raise ValueError('unknown airfoil kind %r' % airfoil.get('kind'))


def SectionCoefficients(airfoil, alpha, Re=None):
    '''
    Section Cl, Cd, Cm at the angles of attack alpha [rad] (any array shape,
    any angle: reverse flow on a rotor blade gives |alpha| up to pi).

    Simple airfoil model:
    * |alpha| <= 45 deg: linear lift a0 (alpha - alpha_L0), clipped to
      +-Cl_max (flat after stall); Cd = Cd0 + Cd1 alpha + Cd2 alpha^2,
      at most 2
    * |alpha| > 45 deg: flat plate, Cl = sin(2 alpha), Cd = max(Cd0, 2 sin^2 alpha)
    * Cm = Cm0
    This is the place where tabulated polars will be plugged in (placeholder).
    '''
    if airfoil is None:
        airfoil = Airfoil()
    if airfoil['kind'] == 'table':
        # TODO: interpolate Cl, Cd, Cm in alpha and Re
        raise NotImplementedError('tabulated airfoil data is not supported yet; '
                                  'use Airfoil() for now')
    alpha = np.angle(np.exp(1j * np.asarray(alpha, dtype=float)))   # wrap to [-pi, pi]
    attached = np.abs(alpha) <= 0.25 * np.pi
    Cl_lin = airfoil['a0'] * (alpha - np.deg2rad(airfoil['alpha_L0']))
    Cl = np.where(attached, np.clip(Cl_lin, -airfoil['Cl_max'], airfoil['Cl_max']),
                  np.sin(2.0 * alpha))
    Cd_poly = airfoil.get('Cd0', 0.01) + airfoil.get('Cd1', 0.0) * alpha \
        + airfoil.get('Cd2', 0.5) * alpha ** 2
    Cd = np.where(attached, np.minimum(Cd_poly, 2.0),
                  np.maximum(airfoil.get('Cd0', 0.01), 2.0 * np.sin(alpha) ** 2))
    Cm = np.full_like(alpha, airfoil['Cm0'])
    return Cl, Cd, Cm


# ------------------------------------------------------------------------------
# Planform and parasite drag
# ------------------------------------------------------------------------------
def _planform(S, b, lmb, sweep):
    '''Trapezoidal planform: chords, mean aerodynamic chord, sweep lines.'''
    AR = b ** 2 / S
    c_root = 2.0 * S / (b * (1.0 + lmb))
    tan_le = np.tan(np.deg2rad(sweep))
    MAC = 2.0 / 3.0 * c_root * (1.0 + lmb + lmb ** 2) / (1.0 + lmb)
    y_mac = b / 6.0 * (1.0 + 2.0 * lmb) / (1.0 + lmb)

    def tan_sweep(frac):
        # sweep of the line at a fraction frac of the local chord
        return tan_le - 4.0 * frac / AR * (1.0 - lmb) / (1.0 + lmb)

    return dict(AR=AR, c_root=c_root, c_tip=lmb * c_root, MAC=MAC, y_mac=y_mac,
                x_le_mac=y_mac * tan_le, tan_sweep=tan_sweep)


def ParasiteDrag(S, MAC, tc, x_tc, Re, M=0.0, sweep_m=0.0, Q=1.0, k_lam=0.0, Swet=None):
    '''
    Parasite (zero-lift) drag coefficient of a wing-like body, based on S.
    Raymer component build-up:  CD0 = Cf * FF * Q * Swet / S

    Parameters
    ----------
    S, MAC  : reference area [m^2] and mean aerodynamic chord [m]
    tc      : thickness / chord;  x_tc: position of max thickness / chord
    Re      : Reynolds number based on MAC
    M       : Mach number
    sweep_m : sweep of the max thickness line [deg]
    Q       : interference factor (1 = isolated wing)
    k_lam   : laminar fraction of the wetted area (0 = fully turbulent)
    Swet    : wetted area [m^2]; default S * (1.977 + 0.52 t/c) (Raymer)

    Model assumptions
    -----------------
    * Cf = k_lam * 1.328/sqrt(Re) + (1 - k_lam) * 0.455 / (log10(Re)^2.58
      (1 + 0.144 M^2)^0.65)
    * FF = (1 + 0.6/x_tc * t/c + 100 (t/c)^4) * cos(sweep_m)^0.28. Raymer's
      extra 1.34 M^0.18 factor is left out (it goes to zero at low Mach)
    * the whole wing is exposed (no fuselage yet)
    '''
    if Re <= 1.0:
        raise ValueError('ParasiteDrag: Reynolds number too small')
    if not (0.0 <= k_lam <= 1.0):
        raise ValueError('ParasiteDrag: k_lam must be in [0, 1]')
    Cf_turb = 0.455 / (np.log10(Re) ** 2.58 * (1.0 + 0.144 * M ** 2) ** 0.65)
    Cf_lam = 1.328 / np.sqrt(Re)
    Cf = k_lam * Cf_lam + (1.0 - k_lam) * Cf_turb
    FF = (1.0 + 0.6 / x_tc * tc + 100.0 * tc ** 4) * np.cos(np.deg2rad(sweep_m)) ** 0.28
    if Swet is None:
        Swet = S * (1.977 + 0.52 * tc)
    return Cf * FF * Q * Swet / S


# ------------------------------------------------------------------------------
# Wing methods
#
# Every method returns the same "linear solution" dict:
#   CL0, CLa          CL = CL0 + CLa * alpha            (alpha in rad)
#   CDi_coef          CDi = c0 + c1 * alpha + c2 * alpha^2
#   e                 span efficiency CLa^2 / (pi AR c2)
#   x_ac, Cm_ac       aerodynamic centre from root LE [m], Cm about it (ref MAC)
#   y, c              spanwise stations on the right half [m] and local chord
#   cl0, cla          local cl = cl0 + cla * alpha at those stations
# ------------------------------------------------------------------------------
def LiftingLine(S, b, lmb, sweep=0.0, dihedral=0.0, twist=0.0, airfoil=None,
                M=0.0, Re=None, n=50):
    '''
    Prandtl lifting line, solved with the symmetric Glauert series
        Gamma(theta) = 2 b V SUM A_k sin(k theta),  k = 1, 3, 5, ...
        y = (b/2) cos(theta)

    Model assumptions
    -----------------
    * straight lifting line: sweep and dihedral are ignored (warning)
    * section lift slope a0 / beta (Prandtl-Glauert)
    * aerodynamic centre at the quarter of the MAC; Cm_ac = airfoil Cm0
    n : number of spanwise stations on one half (= number of harmonics)
    '''
    if sweep != 0.0 or dihedral != 0.0:
        warnings.warn('LiftingLine: sweep and dihedral are ignored by the lifting '
                      'line; use method="vlm" or "semi_empirical" for them')
    p = _planform(S, b, lmb, sweep)
    a0, aL0, Cm0, _ = _section(airfoil, Re)
    a = a0 / np.sqrt(1.0 - M ** 2)
    AR = p['AR']

    k = np.arange(1, int(n) + 1)
    theta = k * np.pi / (2.0 * n)                 # (0, pi/2], pi/2 is the root
    y = 0.5 * b * np.cos(theta)
    c = p['c_root'] * (1.0 - (1.0 - lmb) * y / (0.5 * b))
    eps = np.deg2rad(twist) * y / (0.5 * b)       # local twist [rad]
    nn = 2 * k - 1                                # odd harmonics (symmetric load)

    # monoplane equation:  SUM A_n sin(n th) (1 + n mu / sin th) = mu (alpha + eps - aL0)
    mu = a * c / (4.0 * b)
    sin_nt = np.sin(np.outer(theta, nn))
    mat = sin_nt * (1.0 + np.outer(mu / np.sin(theta), nn))
    A_a = np.linalg.solve(mat, mu)                # per rad of alpha
    A_0 = np.linalg.solve(mat, mu * (eps - aL0))  # at alpha = 0

    CL0, CLa = np.pi * AR * A_0[0], np.pi * AR * A_a[0]
    CDi_coef = (np.pi * AR * np.sum(nn * A_0 ** 2),
                2.0 * np.pi * AR * np.sum(nn * A_0 * A_a),
                np.pi * AR * np.sum(nn * A_a ** 2))

    # local cl = 2 Gamma / (V c)
    cl0 = 4.0 * b * (sin_nt @ A_0) / c
    cla = 4.0 * b * (sin_nt @ A_a) / c
    order = np.argsort(y)                         # root -> tip
    return dict(CL0=CL0, CLa=CLa, CDi_coef=CDi_coef,
                e=CLa ** 2 / (np.pi * AR * CDi_coef[2]),
                x_ac=p['x_le_mac'] + 0.25 * p['MAC'], Cm_ac=Cm0,
                y=y[order], c=c[order], cl0=cl0[order], cla=cla[order])


def SemiEmpirical(S, b, lmb, sweep=0.0, dihedral=0.0, twist=0.0, airfoil=None,
                  M=0.0, Re=None, n=50):
    '''
    Closed-form estimates:
    * lift slope (DATCOM / Helmbold, as in Raymer):
        CLa = 2 pi A / (2 + sqrt(4 + (A beta/eta)^2 (1 + tan^2 L_c/2 / beta^2)))
        eta = a0 / (2 pi). Dihedral: S and b are projected, so the formula
        is applied to the unfolded wing, A = AR / cos(dihedral), and the
        result multiplied by cos(dihedral)
    * zero-lift angle: airfoil alpha_L0 minus the chord-weighted mean twist
        twist (1 + 2 lmb) / (3 (1 + lmb))
    * span efficiency (Nita & Scholz 2012, wing alone):
        e = 1 / (1 + f(lmb - dlmb) AR),  dlmb = -0.357 + 0.45 exp(-0.0375 L_c/4)
        f(x) = 0.0524 x^4 - 0.15 x^3 + 0.1659 x^2 - 0.0706 x + 0.0119
    * Cm_ac = Cm0 AR cos^2 L_c/4 / (AR + 2 cos L_c/4), x_ac at quarter MAC
    * spanwise lift: Schrenk approximation (mean of the planform and the
      elliptic chord); the twist only shifts it uniformly
    n : number of spanwise stations on one half (for the distribution)
    '''
    p = _planform(S, b, lmb, sweep)
    a0, aL0, Cm0, _ = _section(airfoil, Re)
    beta = np.sqrt(1.0 - M ** 2)
    eta = a0 / (2.0 * np.pi)
    AR = p['AR']
    tan_c2 = p['tan_sweep'](0.5)
    tan_c4 = p['tan_sweep'](0.25)

    cos_d = np.cos(np.deg2rad(dihedral))
    A_u = AR / cos_d                              # unfolded aspect ratio
    CLa = cos_d * 2.0 * np.pi * A_u / (2.0 + np.sqrt(4.0 + (A_u * beta / eta) ** 2
                                                     * (1.0 + tan_c2 ** 2 / beta ** 2)))
    eps_mean = np.deg2rad(twist) * (1.0 + 2.0 * lmb) / (3.0 * (1.0 + lmb))
    CL0 = -CLa * (aL0 - eps_mean)

    L_c4 = np.degrees(np.arctan(abs(tan_c4)))
    x = lmb - (-0.357 + 0.45 * np.exp(-0.0375 * L_c4))
    f = 0.0524 * x ** 4 - 0.15 * x ** 3 + 0.1659 * x ** 2 - 0.0706 * x + 0.0119
    e = 1.0 / (1.0 + f * AR)
    k = np.pi * AR * e
    CDi_coef = (CL0 ** 2 / k, 2.0 * CL0 * CLa / k, CLa ** 2 / k)

    cos_c4 = np.cos(np.arctan(tan_c4))
    Cm_ac = Cm0 * AR * cos_c4 ** 2 / (AR + 2.0 * cos_c4)

    # Schrenk: c_s = (c + c_elliptic) / 2, local cl = CL c_s / c
    y = np.linspace(0.0, 0.5 * b, int(n))
    c = p['c_root'] * (1.0 - (1.0 - lmb) * y / (0.5 * b))
    c_ell = 4.0 * S / (np.pi * b) * np.sqrt(np.clip(1.0 - (2.0 * y / b) ** 2, 0.0, None))
    ratio = 0.5 * (c + c_ell) / c
    return dict(CL0=CL0, CLa=CLa, CDi_coef=CDi_coef, e=e,
                x_ac=p['x_le_mac'] + 0.25 * p['MAC'], Cm_ac=Cm_ac,
                y=y, c=c, cl0=CL0 * ratio, cla=CLa * ratio)


def _segment(P, A, B, tol):
    '''
    Velocity at the points P (nP, 3) induced by unit-strength straight vortex
    segments A -> B (nS, 3), Biot-Savart. Returns (nP, nS, 3). Points closer
    than tol to a segment's line get zero.
    '''
    r1 = P[:, None, :] - A[None, :, :]
    r2 = P[:, None, :] - B[None, :, :]
    r0 = B - A
    cr = np.cross(r1, r2)
    cr2 = np.einsum('ijk,ijk->ij', cr, cr)
    n1 = np.linalg.norm(r1, axis=-1)[..., None]
    n2 = np.linalg.norm(r2, axis=-1)[..., None]
    dot = np.einsum('jk,ijk->ij', r0, r1 / n1 - r2 / n2)
    ok = cr2 > tol ** 2 * np.einsum('jk,jk->j', r0, r0)[None, :]
    k = np.where(ok, dot / (4.0 * np.pi * np.where(ok, cr2, 1.0)), 0.0)
    return cr * k[..., None]


def VortexLattice(S, b, lmb, sweep=0.0, dihedral=0.0, twist=0.0, airfoil=None,
                  M=0.0, Re=None, n_span=20, n_chord=6, spacing='cosine'):
    '''
    Vortex lattice method with horseshoe vortices on the (flat) mean surface.

    Model assumptions
    -----------------
    * bound vortex at 1/4 and control point at 3/4 of each panel, trailing
      legs parallel to the x axis to infinity
    * camber through the zero-lift line: every panel normal is tilted by
      (local twist - alpha_L0); the airfoil Cm0 is added to Cm_ac
    * small angles: freestream (1, 0, alpha) (linear in alpha)
    * Goethert rule: influence coefficients on the planform stretched by
      1/beta in x; loads are the same, so CL = CL_stretched / beta
    * induced drag in the Trefftz plane
    n_span  : spanwise strips on one half;  n_chord: chordwise panels
    spacing : 'cosine' (strips refined towards the tip) or 'uniform'
    '''
    if spacing not in ('cosine', 'uniform'):
        raise ValueError("VortexLattice: spacing must be 'cosine' or 'uniform'")
    p = _planform(S, b, lmb, sweep)
    _, aL0, Cm0, _ = _section(airfoil, Re)
    beta = np.sqrt(1.0 - M ** 2)
    half = 0.5 * b
    tan_le = np.tan(np.deg2rad(sweep))
    sin_d, cos_d = np.sin(np.deg2rad(dihedral)), np.cos(np.deg2rad(dihedral))
    tan_d = sin_d / cos_d

    # strip edges from left tip to right tip, panel edges along the chord.
    # With cosine spacing the control stations sit at the angular midpoint
    # of each strip (semicircle method), which converges much faster.
    if spacing == 'cosine':
        phi = np.linspace(0.0, 0.5 * np.pi, int(n_span) + 1)
        eta = np.sin(phi)
        eta_c = np.sin(0.5 * (phi[:-1] + phi[1:]))
    else:
        eta = np.linspace(0.0, 1.0, int(n_span) + 1)
        eta_c = 0.5 * (eta[:-1] + eta[1:])
    y_e = half * np.concatenate([-eta[::-1], eta[1:]])
    y_m = half * np.concatenate([-eta_c[::-1], eta_c])
    ns = y_e.size - 1
    xi = np.linspace(0.0, 1.0, int(n_chord) + 1)
    xi_q = xi[:-1] + 0.25 * np.diff(xi)           # bound vortices
    xi_c = xi[:-1] + 0.75 * np.diff(xi)           # control points

    def chord(y):
        return p['c_root'] * (1.0 - (1.0 - lmb) * np.abs(y) / half)

    def points(xi_, y):                           # -> (n_chord * ns, 3)
        yy = np.broadcast_to(y, (xi_.size, y.size))
        xx = np.abs(yy) * tan_le + xi_[:, None] * chord(yy)
        zz = np.abs(yy) * tan_d
        return np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)

    A = points(xi_q, y_e[:-1])
    B = points(xi_q, y_e[1:])
    P = points(xi_c, y_m)

    # panel normals: tilted nose-up by (twist - alpha_L0), rotated by dihedral
    side = np.sign(y_m)
    eps = np.deg2rad(twist) * np.abs(y_m) / half - aL0
    eps, side = (np.broadcast_to(v, (xi_c.size, ns)).ravel() for v in (eps, side))
    normal = np.stack([np.sin(eps), -side * sin_d * np.cos(eps),
                       cos_d * np.cos(eps)], axis=-1)

    # influence coefficients on the stretched planform (Goethert)
    st = np.array([1.0 / beta, 1.0, 1.0])
    As, Bs, Ps = A * st, B * st, P * st
    far = np.array([1e6 * b, 0.0, 0.0])
    tol = 1e-9 * b
    vel = (_segment(Ps, As + far, As, tol) + _segment(Ps, As, Bs, tol)
           + _segment(Ps, Bs, Bs + far, tol))
    aic = np.einsum('ijk,ik->ij', vel, normal)

    # flow tangency (unit V):  aic Gamma = -(1, 0, alpha) . n
    G_0 = np.linalg.solve(aic, -normal[:, 0])
    G_a = np.linalg.solve(aic, -normal[:, 2])

    # loads (Kutta-Joukowski on the bound vortices, original planform)
    dy = B[:, 1] - A[:, 1]
    x_b = 0.5 * (A[:, 0] + B[:, 0])
    CL0, CLa = (2.0 * np.sum(G * dy) / S for G in (G_0, G_a))
    Cm0_le, Cma_le = (-2.0 * np.sum(G * dy * x_b) / (S * p['MAC']) for G in (G_0, G_a))
    x_ac = -Cma_le / CLa * p['MAC']
    Cm_ac = Cm0_le + CL0 * x_ac / p['MAC'] + Cm0

    # strip circulation and local cl
    Gs_0 = G_0.reshape(xi_c.size, ns).sum(axis=0)
    Gs_a = G_a.reshape(xi_c.size, ns).sum(axis=0)
    c_m = chord(y_m)

    # Trefftz plane: trailing vortex at strip edge k has strength
    # Gs[k] - Gs[k-1]; normal wash at the strip centres
    z_e, z_m = np.abs(y_e) * tan_d, np.abs(y_m) * tan_d
    ry = y_m[:, None] - y_e[None, :]
    rz = z_m[:, None] - z_e[None, :]
    K = (rz * (-np.sign(y_m)[:, None] * sin_d) - ry * cos_d) / (2.0 * np.pi * (ry ** 2 + rz ** 2))
    D = np.eye(ns + 1, ns) - np.eye(ns + 1, ns, k=-1)
    ds = np.hypot(np.diff(y_e), np.diff(z_e))
    Qm = -(ds[:, None] * (K @ D)) / S             # CDi = Gs . Qm . Gs
    CDi_coef = (Gs_0 @ Qm @ Gs_0, Gs_0 @ (Qm + Qm.T) @ Gs_a, Gs_a @ Qm @ Gs_a)

    right = y_m > 0
    return dict(CL0=CL0, CLa=CLa, CDi_coef=CDi_coef,
                e=CLa ** 2 / (np.pi * p['AR'] * CDi_coef[2]),
                x_ac=x_ac, Cm_ac=Cm_ac,
                y=y_m[right], c=c_m[right],
                cl0=2.0 * Gs_0[right] / c_m[right], cla=2.0 * Gs_a[right] / c_m[right])


METHODS = {'lifting_line': LiftingLine,
           'semi_empirical': SemiEmpirical,
           'vlm': VortexLattice}
METHOD_LABELS = {'lifting_line': 'Lifting line',
                 'semi_empirical': 'Semi-empirical',
                 'vlm': 'Vortex lattice'}


def WingAero(S, b, lmb, alpha, V, h=0.0, sweep=0.0, dihedral=0.0, twist=0.0,
             airfoil=None, method='lifting_line', Q=1.0, k_lam=0.0,
             verbose=False, **method_options):
    '''
    Basic aerodynamics of a wing-like body (both sides, symmetric flight).

    Implementation of the pseudo code:
        1) flight condition: ISA, Mach, Reynolds (MAC), dynamic pressure
        2) planform: AR, chords, MAC, sweep lines
        3) lift and induced drag with the chosen method (linear in alpha)
        4) parasite drag (skin friction build-up)
        5) CL_max / stall angle (critical section)
        6) coefficients and forces for every alpha
        7) return results dict

    Parameters (SI units, angles in degrees)
    ----------------------------------------
    S, b, lmb : area (both sides), span (tip to tip), taper ratio
    alpha     : angle(s) of attack of the root chord [deg], scalar or array
    V, h      : flight speed [m/s] and altitude [m]
    sweep     : leading edge sweep [deg];  dihedral [deg]
    twist     : tip twist relative to root [deg] (negative = washout)
    airfoil   : Airfoil(...) dict (default: Airfoil(), thin symmetric section)
    method    : 'lifting_line', 'semi_empirical' or 'vlm'
    Q, k_lam  : interference factor and laminar fraction for ParasiteDrag
    verbose   : print a summary and the alpha table when True
    method_options : passed to the method, e.g. n=80 (lifting_line),
                     n_span=30, n_chord=8, spacing='uniform' (vlm)

    Model assumptions
    -----------------
    * isolated wing (no fuselage, no downwash on a tail)
    * linear lift curve; results past alpha_stall are not physical
    * CD = CD0 + CDi (profile drag change with CL not included yet; it will
      come with tabulated airfoil data)

    Example
    -------
    R = WingAero(S=3.0, b=5.0, lmb=0.5, alpha=np.arange(-4, 13), V=20.0,
                 h=500.0, airfoil=Airfoil(alpha_L0=-2.0, Cm0=-0.05),
                 method='vlm', verbose=True)

    Returns
    -------
    dict with (arrays have one value per alpha)
        method, alpha [deg], CL, CDi, CD0, CD, L_D, L [N], D [N]
        CL_alpha [1/rad], alpha_L0 [deg], e, CL_max, alpha_stall [deg]
        Cm_ac, x_ac [m] (from root LE), MAC [m], AR, c_root, c_tip [m]
        M, Re (MAC), q [Pa], rho [kg/m^3]
        y [m], c [m], cl0, cla [1/rad]: spanwise cl = cl0 + cla * alpha (right half)
        cl : local cl at every alpha, shape (n_alpha, n_y)
    '''
    # ---------------------------- 0) input checks ---------------------------
    if S <= 0 or b <= 0:
        raise ValueError('WingAero: S and b must be positive')
    if lmb <= 0:
        raise ValueError('WingAero: taper ratio lmb must be positive')
    if V <= 0:
        raise ValueError('WingAero: V must be positive')
    if method not in METHODS:
        raise ValueError('WingAero: method must be one of %s' % sorted(METHODS))
    if airfoil is None:
        airfoil = Airfoil()

    # ------------------- 1) flight condition --------------------------------
    T, p_atm, rho, mu, a_sound = ISA(h)
    M = V / a_sound
    if M >= 1.0:
        raise ValueError('WingAero: supersonic flight (M = %.2f) is not supported' % M)
    if M > 0.7:
        warnings.warn('WingAero: M = %.2f, linear compressibility corrections are '
                      'not reliable above about 0.7' % M)
    q = 0.5 * rho * V ** 2

    # ------------------- 2) planform ----------------------------------------
    p = _planform(S, b, lmb, sweep)
    Re = rho * V * p['MAC'] / mu

    # ------------------- 3) lift and induced drag ---------------------------
    sol = METHODS[method](S, b, lmb, sweep=sweep, dihedral=dihedral, twist=twist,
                          airfoil=airfoil, M=M, Re=Re, **method_options)
    alpha_deg = np.atleast_1d(np.asarray(alpha, dtype=float))
    a = np.deg2rad(alpha_deg)
    CL = sol['CL0'] + sol['CLa'] * a
    c0, c1, c2 = sol['CDi_coef']
    CDi = c0 + c1 * a + c2 * a ** 2

    # ------------------- 4) parasite drag -----------------------------------
    sweep_m = np.degrees(np.arctan(p['tan_sweep'](airfoil['x_tc'])))
    CD0 = ParasiteDrag(S, p['MAC'], airfoil['tc'], airfoil['x_tc'], Re, M=M,
                       sweep_m=sweep_m, Q=Q, k_lam=k_lam)
    CD = CD0 + CDi

    # ------------------- 5) CL_max (critical section) -----------------------
    _, _, _, Cl_max = _section(airfoil, Re)
    pos = sol['cla'] > 0
    a_stall = np.min((Cl_max - sol['cl0'][pos]) / sol['cla'][pos])
    CL_max = sol['CL0'] + sol['CLa'] * a_stall

    # ------------------- 6) results -----------------------------------------
    R = dict(method=method, alpha=alpha_deg, CL=CL, CDi=CDi, CD0=CD0, CD=CD,
             L_D=CL / CD, L=q * S * CL, D=q * S * CD,
             CL_alpha=sol['CLa'], alpha_L0=np.degrees(-sol['CL0'] / sol['CLa']),
             e=sol['e'], CL_max=CL_max, alpha_stall=np.degrees(a_stall),
             Cm_ac=sol['Cm_ac'], x_ac=sol['x_ac'], MAC=p['MAC'], AR=p['AR'],
             c_root=p['c_root'], c_tip=p['c_tip'],
             M=M, Re=Re, q=q, rho=rho,
             y=sol['y'], c=sol['c'], cl0=sol['cl0'], cla=sol['cla'],
             cl=sol['cl0'][None, :] + np.outer(a, sol['cla']))

    # ------------------- 7) return ------------------------------------------
    if verbose:
        print('Wing aerodynamics, %s (origin: leading edge of root chord)'
              % METHOD_LABELS[method])
        print('  S = %.4f m^2   b = %.4f m   AR = %.3f   lmb = %.3f   MAC = %.4f m'
              % (S, b, p['AR'], lmb, p['MAC']))
        print('  V = %.2f m/s   h = %.0f m   M = %.3f   Re_MAC = %.3e   q = %.2f Pa'
              % (V, h, M, Re, q))
        print('  CL_alpha = %.4f /rad (%.5f /deg)   alpha_L0 = %.3f deg   e = %.4f'
              % (R['CL_alpha'], np.deg2rad(R['CL_alpha']), R['alpha_L0'], R['e']))
        print('  CD0 = %.5f   Cm_ac = %.4f   x_ac = %.4f m   CL_max = %.3f at alpha = %.2f deg'
              % (CD0, R['Cm_ac'], R['x_ac'], CL_max, R['alpha_stall']))
        print('  %7s %8s %9s %9s %8s %9s %9s' % ('alpha', 'CL', 'CDi', 'CD', 'L/D', 'L [N]', 'D [N]'))
        for i in range(alpha_deg.size):
            flag = '  past stall' if alpha_deg[i] > R['alpha_stall'] else ''
            print('  %7.2f %8.4f %9.5f %9.5f %8.3f %9.2f %9.3f%s'
                  % (alpha_deg[i], CL[i], CDi[i], CD[i], R['L_D'][i], R['L'][i], R['D'][i], flag))

    return R


if __name__ == '__main__':
    # Example: the WeightEstimate wing (S = 3 m^2, b = 5 m, lmb = 0.5) at
    # 20 m/s, 500 m, slightly cambered airfoil, with every method
    foil = Airfoil(alpha_L0=-2.0, Cm0=-0.05, Cl_max=1.4)
    for m in METHODS:
        WingAero(S=3.0, b=5.0, lmb=0.5, alpha=np.arange(-4.0, 14.0, 2.0),
                 V=20.0, h=500.0, airfoil=foil, method=m, verbose=True)
        print()
