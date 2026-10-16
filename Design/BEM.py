import warnings
import numpy as np

from Aerodynamics import Airfoil, SectionCoefficients
from MomentumCode import AxialInflow, ForwardInflow, FigureOfMerit

_trapz = getattr(np, "trapezoid", None) or np.trapz   # np.trapz was removed in NumPy 2.4

'''
Blade element (momentum) theory for rotors and propellers.

Plan (from the original notes):
In BEM code I want 2 modes
1) Uniform Inflow
2) Momentum theory with tip loss
For design purposes the 1st one is enough, for detailed simulation the second.
We would also need a coordinate transformation. Keep in mind we have torque,
we need the gyroscopic moment, which allows us to work out angular
acceleration. Cyclic pitches are added too.
Two sets of outputs: Design mode and Simulator mode (simulator mode is only
for the advanced code).
    UniformInflow()    -> CT, CL, CM, CX, CY, CQ
    NonUniformInflow() -> CT, CL, CM, CX, CY, CQ
    Simulation()       -> C, lambda, beta
Cl, Cd, Cm of the airfoil will be arrays in alpha and Re, so really precise
data can be imported (for now simple parameters, see Aerodynamics.Airfoil;
tables are a placeholder in Aerodynamics.SectionCoefficients).
"BEM requires us to input the inflow": momentum theory (MomentumCode.py)
provides it -- one uniform value for UniformInflow, one per annulus (BEMT)
for NonUniformInflow, Glauert or Drees for the Simulation.

What is implemented
-------------------
Design mode (axial flow: hover, climb, propeller in forward flight)
    UniformInflow     blade element theory + one uniform momentum inflow
    NonUniformInflow  blade element momentum theory per annulus,
                      param = 1 momentum, 2 momentum + Prandtl tip loss
    Design            either of the above by model name
    DesignTrim        collective that gives a required thrust
Simulator mode (edgewise forward flight, first version)
    Simulation        cyclic pitch, blade flapping (time marched until
                      periodic), uniform (Glauert) or Drees inflow, mean hub
                      forces and moments
    GyroscopicMoment  moment of the spinning rotor on a rotating airframe
Not yet: angular acceleration / full rotor dynamics, unsteady loads (N/rev
vibration), compressibility on the blades, swirl in the wake.

Terminology (SI units, angles in degrees at the interface):
W       = rotor speed [rad/s] (Omega)
theta   = collective pitch at the root cutout [deg]; in the simulator
          th = (theta0, theta1s, theta1c) adds cyclic pitch:
          theta(psi, r) = theta0 + theta1s sin(psi) + theta1c cos(psi)
                          + twist (r - rcut) / (R - rcut)
n       = number of radial stations (from rcut to R);  N = number of blades
airfoil = section data, Aerodynamics.Airfoil(...)
c       = chord at the root cutout [m];  tap = taper ratio c_tip / c_root
twist   = total linear twist from root cutout to tip [deg] (negative = washout)
R       = rotor radius [m];  rcut = root cutout radius [m]
V       = flight speed [m/s]: along the shaft in design mode (climb speed of
          a lift rotor, flight speed of a propeller), along the flight path
          in the simulator
rho, a_sound = air density [kg/m^3] and speed of sound [m/s] (Aerodynamics.ISA)

Coefficients use the disk area A = pi R^2 and the tip speed W R:
CT = T / (rho A (W R)^2),   CQ = CP = Q / (rho A (W R)^2 R)
CX, CY = in-plane hub forces (like CT), CL, CM = hub roll and pitch
moments (like CQ). In axial flow they are zero by symmetry.
Propeller coefficients are also returned: J = V / (n D),
kT = T / (rho n^2 D^4), kQ = Q / (rho n^2 D^5), kP = P / (rho n^3 D^5).

Shaft (hub) frame: z along the shaft (thrust direction), x aft
(downstream, azimuth psi = 0), y to the right. Blades turn counter-clockwise
seen from above, so the advancing side is psi = 90 deg (right).
Roll moment positive right side down, pitch moment positive nose up.
'''


# ------------------------------------------------------------------------------
# Blade geometry and section loads
# ------------------------------------------------------------------------------
def _blade(n, N, c, tap, twist, R, rcut):
    if R <= 0 or not (0.0 <= rcut < R):
        raise ValueError('Rotor: need R > 0 and 0 <= rcut < R')
    if c <= 0 or tap <= 0:
        raise ValueError('Rotor: chord c and taper tap must be positive')
    if int(N) < 1 or int(n) < 3:
        raise ValueError('Rotor: need N >= 1 blades and n >= 3 stations')
    r = np.linspace(rcut, R, int(n))
    s = (r - rcut) / (R - rcut)
    chord = c * (1.0 - (1.0 - tap) * s)
    return dict(r=r, x=r / R, chord=chord, twist=np.deg2rad(twist) * s,
                sigma_r=int(N) * chord / (np.pi * R),
                sigma=int(N) * _trapz(chord, r) / (np.pi * R ** 2))


def _axial_section(lam, x, theta, sigma_r, airfoil):
    '''
    Blade element in axial flow (no swirl): U_T = x, U_P = lam (both / W R)
        dCT/dx = sigma_r / 2 U^2 (Cl cos phi - Cd sin phi)
        dCQ/dx = sigma_r / 2 U^2 (Cl sin phi + Cd cos phi) x
    '''
    phi = np.arctan2(lam, x)
    alpha = theta - phi
    Cl, Cd, _ = SectionCoefficients(airfoil, alpha)
    U2 = x ** 2 + lam ** 2
    dCT = 0.5 * sigma_r * U2 * (Cl * np.cos(phi) - Cd * np.sin(phi))
    dCQ = 0.5 * sigma_r * U2 * (Cl * np.sin(phi) + Cd * np.cos(phi)) * x
    return phi, alpha, Cl, Cd, dCT, dCQ


def _tip_loss(N, x, phi):
    '''Prandtl tip loss F = 2/pi arccos(exp(-N/2 (1 - x) / (x sin phi)))'''
    f = 0.5 * N * (1.0 - x) / (np.maximum(x, 1e-6) * np.maximum(np.abs(np.sin(phi)), 1e-6))
    return 2.0 / np.pi * np.arccos(np.exp(-f))


def _design_results(method, W, R, rho, a_sound, N, b, theta, V, lam, F, airfoil, verbose):
    '''Integrate the radial loads and build the design mode results dict.'''
    x = b['x']
    th = np.deg2rad(theta) + b['twist']
    phi, alpha, Cl, Cd, dCT, dCQ = _axial_section(lam, x, th, b['sigma_r'], airfoil)
    A, WR = np.pi * R ** 2, W * R
    CT, CQ = _trapz(dCT, x), _trapz(dCQ, x)
    T, Q = CT * rho * A * WR ** 2, CQ * rho * A * WR ** 2 * R
    P = Q * W
    n_rps, D = W / (2.0 * np.pi), 2.0 * R
    x0 = x[0]
    M_tip = WR / a_sound
    if M_tip > 0.7:
        warnings.warn('Rotor: tip Mach number %.2f, compressibility is not modelled' % M_tip)

    res = dict(method=method, theta=theta,
               theta_75=theta + np.degrees(np.interp(0.75, x, b['twist'])),
               V=V, W=W, rpm=W * 60.0 / (2.0 * np.pi), R=R, N=int(N),
               CT=CT, CQ=CQ, CP=CQ, CL=0.0, CM=0.0, CX=0.0, CY=0.0,
               T=T, Q=Q, P=P, sigma=b['sigma'], CT_sigma=CT / b['sigma'],
               FM=FigureOfMerit(CT, CQ) if (V == 0 and CT > 0 and CQ > 0) else np.nan,
               eta=T * V / P if (V > 0 and P > 0 and T > 0) else np.nan,
               J=V / (n_rps * D), kT=T / (rho * n_rps ** 2 * D ** 4),
               kQ=Q / (rho * n_rps ** 2 * D ** 5), kP=P / (rho * n_rps ** 3 * D ** 5),
               lam_c=V / WR, lam_mean=2.0 * _trapz(lam * x, x) / (1.0 - x0 ** 2),
               M_tip=M_tip, tip_speed=WR,
               r=b['r'], x=x, lam_r=lam, alpha_r=np.degrees(alpha), phi_r=np.degrees(phi),
               Cl_r=Cl, Cd_r=Cd, dCT=dCT, dCQ=dCQ, F=F)

    if verbose:
        print('Rotor design mode, %s (axial flow)' % DESIGN_MODELS[method])
        print('  R = %.4f m   N = %d   sigma = %.4f   rpm = %.0f   tip speed = %.1f m/s   M_tip = %.3f'
              % (R, N, b['sigma'], res['rpm'], WR, M_tip))
        print('  theta0 = %.3f deg (theta_75 = %.3f deg)   V = %.2f m/s   lam_c = %.5f   mean inflow = %.5f'
              % (theta, res['theta_75'], V, res['lam_c'], res['lam_mean']))
        print('  CT = %.6f   CQ = %.7f   CT/sigma = %.4f   FM = %.4f   eta = %.4f'
              % (CT, CQ, res['CT_sigma'], res['FM'], res['eta']))
        print('  T = %.3f N   Q = %.4f N m   P = %.2f W'
              % (T, Q, P))
        print('  propeller coefficients: J = %.4f   kT = %.5f   kQ = %.6f   kP = %.5f'
              % (res['J'], res['kT'], res['kQ'], res['kP']))
    return res


# ------------------------------------------------------------------------------
# Design mode (axial flow)
# ------------------------------------------------------------------------------
def UniformInflow(W, theta, n, N, airfoil, c, twist, tap, R, rcut,
                  V=0.0, rho=1.225, a_sound=340.3, kappa=1.0, verbose=False):
    '''
    Design mode: blade element theory with one uniform inflow over the disk,
    taken from momentum theory (MomentumCode.AxialInflow) for the thrust it
    produces. Solved by bisection on the inflow, so it also works in descent
    (vortex ring fit) and for windmilling propellers.

    kappa : induced power factor applied to the induced inflow (1 = ideal)

    Returns
    -------
    dict with CT, CL, CM, CX, CY, CQ (the latter four are 0 in axial flow),
    CP, T [N], Q [N m], P [W], FM, eta, J, kT, kQ, kP, CT_sigma, M_tip,
    radial distributions r, x, lam_r, alpha_r [deg], Cl_r, Cd_r, dCT, dCQ, F
    '''
    if airfoil is None:
        airfoil = Airfoil()
    b = _blade(n, N, c, tap, twist, R, rcut)
    WR = W * R
    lam_c = V / WR
    th = np.deg2rad(theta) + b['twist']

    def residual(lam):
        CT = _trapz(_axial_section(np.full_like(b['x'], lam), b['x'], th,
                                   b['sigma_r'], airfoil)[4], b['x'])
        return lam - AxialInflow(CT, lam_c, kappa)

    # residual grows with lam (more inflow -> less thrust -> less induced)
    lo, hi = lam_c - 0.5, lam_c + 0.5
    if residual(lo) > 0 or residual(hi) < 0:
        raise RuntimeError('UniformInflow: could not bracket the inflow')
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if residual(mid) < 0:
            lo = mid
        else:
            hi = mid
    lam = np.full_like(b['x'], 0.5 * (lo + hi))
    return _design_results('uniform', W, R, rho, a_sound, N, b, theta, V, lam,
                           np.ones_like(lam), airfoil, verbose)


def NonUniformInflow(W, th, n, N, airfoil, c, twist, tap, R, rcut, param=2,
                     V=0.0, rho=1.225, a_sound=340.3, verbose=False):
    '''
    Design mode: blade element momentum theory (BEMT). Each annulus has its
    own inflow lam(x), from thrust balance between momentum and blade element:
        4 F lam (lam - lam_c) x = sigma_r / 2 U^2 (Cl cos phi - Cd sin phi)
    solved per station by bisection (exact angles, any airfoil model).

    param : 1 = momentum theory inflow (F = 1)
            2 = momentum theory + Prandtl tip loss
    th    : collective pitch at the root cutout [deg]

    Model assumptions
    -----------------
    * no swirl, annuli independent of each other
    * climb, hover and propellers only (descent: use UniformInflow); sections
      that would windmill beyond the momentum limit get lam = lam_c / 2
    '''
    if param not in (1, 2):
        raise ValueError('NonUniformInflow: param must be 1 or 2')
    if airfoil is None:
        airfoil = Airfoil()
    b = _blade(n, N, c, tap, twist, R, rcut)
    x, sr = b['x'], b['sigma_r']
    WR = W * R
    lam_c = V / WR
    if lam_c < 0:
        raise ValueError('NonUniformInflow: descent (V < 0) is not supported, '
                         'use UniformInflow')
    pitch = np.deg2rad(th) + b['twist']

    def tip(lam):
        phi = np.arctan2(lam, x)
        return _tip_loss(N, x, phi) if param == 2 else np.ones_like(x)

    def g(lam):
        dCT = _axial_section(lam, x, pitch, sr, airfoil)[4]
        return 4.0 * tip(lam) * lam * (lam - lam_c) * x - dCT

    lo = np.full_like(x, 0.5 * lam_c)
    hi = np.full_like(x, lam_c + 0.5)
    clamped = g(lo) > 0
    # widen the bracket where needed (e.g. the tip, where F = 0, only
    # balances once the blade itself stops lifting)
    for _ in range(12):
        short = g(hi) < 0
        if not np.any(short):
            break
        hi = np.where(short, lam_c + 2.0 * (hi - lam_c), hi)
    else:
        raise RuntimeError('NonUniformInflow: could not bracket the inflow')
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        low = g(mid) < 0
        lo = np.where(low, mid, lo)
        hi = np.where(low, hi, mid)
    lam = np.where(clamped, 0.5 * lam_c, 0.5 * (lo + hi))

    inner = clamped & (x > 0) & (x < 1)
    if np.any(inner):
        warnings.warn('NonUniformInflow: %d stations windmill beyond the momentum '
                      'theory limit; their inflow is clamped' % np.sum(inner))
    method = 'bemt' if param == 1 else 'bemt_tip'
    return _design_results(method, W, R, rho, a_sound, N, b, th, V, lam, tip(lam),
                           airfoil, verbose)


DESIGN_MODELS = {'uniform': 'Uniform inflow (BET + momentum)',
                 'bemt': 'BEMT (momentum)',
                 'bemt_tip': 'BEMT + Prandtl tip loss'}


def Design(model, W, theta, n, N, airfoil, c, twist, tap, R, rcut, **kw):
    '''Design mode with the model chosen by name (see DESIGN_MODELS).'''
    if model == 'uniform':
        return UniformInflow(W, theta, n, N, airfoil, c, twist, tap, R, rcut, **kw)
    if model in ('bemt', 'bemt_tip'):
        return NonUniformInflow(W, theta, n, N, airfoil, c, twist, tap, R, rcut,
                                param=1 if model == 'bemt' else 2, **kw)
    raise ValueError('Design: model must be one of %s' % sorted(DESIGN_MODELS))


def DesignTrim(T_req, model, W, n, N, airfoil, c, twist, tap, R, rcut,
               theta_bounds=(-10.0, 45.0), verbose=False, **kw):
    '''
    Collective (at the root cutout) that gives the thrust T_req [N], found by
    scanning theta_bounds in 2 deg steps and refining the first crossing by
    bisection. Raises ValueError if the rotor cannot reach T_req (stall).

    Returns the Design results at the trimmed collective.
    '''
    def thrust(theta):
        return Design(model, W, theta, n, N, airfoil, c, twist, tap, R, rcut, **kw)['T']

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        grid = np.arange(theta_bounds[0], theta_bounds[1] + 1e-9, 2.0)
        T_grid = np.array([thrust(th) for th in grid])
        above = np.nonzero(T_grid >= T_req)[0]
        if above.size == 0 or above[0] == 0:
            i = int(np.argmax(T_grid))
            raise ValueError('DesignTrim: the rotor cannot produce T = %.2f N between '
                             '%.0f and %.0f deg collective (max %.2f N at %.0f deg)'
                             % (T_req, theta_bounds[0], theta_bounds[1], T_grid[i], grid[i]))
        lo, hi = grid[above[0] - 1], grid[above[0]]
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if thrust(mid) < T_req:
                lo = mid
            else:
                hi = mid
    return Design(model, W, 0.5 * (lo + hi), n, N, airfoil, c, twist, tap, R, rcut,
                  verbose=verbose, **kw)


# ------------------------------------------------------------------------------
# Simulator mode (edgewise forward flight)
# ------------------------------------------------------------------------------
INFLOW_MODELS = {'uniform': 'Uniform (Glauert)', 'drees': 'Drees (linear)'}


def Simulation(W, th, n, N, airfoil, c, twist, tap, R, rcut, V, alpha_s=0.0,
               rho=1.225, a_sound=340.3, inflow='drees', gamma=8.0, nu_beta=1.0,
               n_psi=72, tol=1e-6, max_rev=300, verbose=False):
    '''
    Simulator mode: rotor in edgewise forward flight with cyclic pitch.

    The blade flapping equation (rigid blade, flap hinge at the shaft, flap
    spring through nu_beta)
        beta'' + nu_beta^2 beta = gamma / (2 a0 c_75) INT x U^2 c (Cl cos phi - Cd sin phi) dx
    is marched in azimuth (RK4) until the motion repeats every revolution,
    while the inflow is updated from the rotor thrust each revolution. Mean
    hub loads are the average over one revolution times N.

    Parameters (besides the blade geometry, see the module notes)
    ----------
    th       : (theta0, theta1s, theta1c) [deg]
    V        : flight speed [m/s];  alpha_s: shaft angle of attack [deg],
               positive tilted aft (nose up); -90 is axial flow (propeller /
               climbing rotor), 0 is a shaft perpendicular to the flight path
    inflow   : 'uniform' (Glauert) or 'drees' (linear: more inflow at the
               back of the disk, less on the advancing side)
    gamma    : Lock number (blade aero / inertia forces, ~ 5 to 10)
    nu_beta  : flap frequency / rotor speed (1 = articulated; > 1 hingeless,
               which gives hub moments)
    n_psi    : azimuth steps per revolution

    Model assumptions
    -----------------
    * steady flight, identical blades, quasi-steady aerodynamics, no radial
      flow effects, no compressibility; reverse flow handled by the airfoil
      model (flat plate beyond 45 deg)
    * blade flap inertia from the Lock number: I_b = rho a0 c_75 R^4 / gamma

    Returns
    -------
    dict with coefficients CT, CX (H force, aft), CY (side), CQ, CP, CL (roll),
    CM (pitch); forces T, H, Y [N], Q [N m], P [W], roll, pitch [N m];
    inflow mu, lam (mean total), lam_i (mean induced); flapping beta0,
    beta1c, beta1s [deg] and beta(psi); maps psi [deg], x, alpha_map [deg]
    (n_psi x n); I_b, I_rotor [kg m^2]; M_tip, M_adv
    '''
    if inflow not in INFLOW_MODELS:
        raise ValueError('Simulation: inflow must be one of %s' % sorted(INFLOW_MODELS))
    if gamma <= 0 or nu_beta <= 0:
        raise ValueError('Simulation: gamma and nu_beta must be positive')
    if int(n_psi) < 12:
        raise ValueError('Simulation: n_psi must be >= 12')
    if airfoil is None:
        airfoil = Airfoil()
    b = _blade(n, N, c, tap, twist, R, rcut)
    x, chord = b['x'], b['chord']
    WR = W * R
    A = np.pi * R ** 2
    th0, th1s, th1c = np.deg2rad(np.asarray(th, dtype=float))
    pitch_r = th0 + b['twist']
    a_s = np.deg2rad(alpha_s)
    mu = V * np.cos(a_s) / WR
    lam_fs = -V * np.sin(a_s) / WR                # freestream through the disk
    c_75 = np.interp(0.75 * R, b['r'], chord)
    I_b = rho * airfoil['a0'] * c_75 * R ** 4 / gamma
    K_flap = I_b * W ** 2 * (nu_beta ** 2 - 1.0)
    k_aero = gamma / (2.0 * airfoil['a0'] * c_75)
    n_psi = int(n_psi)
    dpsi = 2.0 * np.pi / n_psi

    def sections(psi, beta, dbeta, lam_i, kx, ky):
        # broadcasts: psi, beta, dbeta scalars or (m, 1) columns; x is (n,)
        uT = x + mu * np.sin(psi)
        lam = lam_fs + lam_i * (1.0 + kx * x * np.cos(psi) + ky * x * np.sin(psi))
        uP = lam + x * dbeta + mu * beta * np.cos(psi)
        phi = np.arctan2(uP, uT)
        alpha = pitch_r + th1s * np.sin(psi) + th1c * np.cos(psi) - phi
        Cl, Cd, _ = SectionCoefficients(airfoil, alpha)
        U2 = uT ** 2 + uP ** 2
        fz = U2 * chord * (Cl * np.cos(phi) - Cd * np.sin(phi))   # x rho (W R)^2 / 2
        fx = U2 * chord * (Cl * np.sin(phi) + Cd * np.cos(phi))
        return alpha, fz, fx

    def rhs(psi, y, infl):
        fz = sections(psi, y[0], y[1], *infl)[1]
        return np.array([y[1], -nu_beta ** 2 * y[0] + k_aero * _trapz(x * fz, x)])

    def skew(lam_i):
        if inflow == 'uniform' or mu == 0.0:
            return 0.0, 0.0
        chi = np.arctan2(mu, lam_fs + lam_i)      # wake skew angle
        return 4.0 / 3.0 * (1.0 - np.cos(chi) - 1.8 * mu ** 2) / np.sin(chi), -2.0 * mu

    psi_k = np.arange(n_psi) * dpsi
    y = np.array([0.0, 0.0])
    lam_i = ForwardInflow(0.005, mu, lam_fs) - lam_fs
    for rev in range(int(max_rev)):
        infl = (lam_i,) + skew(lam_i)
        y_start = y.copy()
        states = np.empty((n_psi, 2))
        for k in range(n_psi):
            states[k] = y
            p = psi_k[k]
            k1 = rhs(p, y, infl)
            k2 = rhs(p + 0.5 * dpsi, y + 0.5 * dpsi * k1, infl)
            k3 = rhs(p + 0.5 * dpsi, y + 0.5 * dpsi * k2, infl)
            k4 = rhs(p + dpsi, y + dpsi * k3, infl)
            y = y + dpsi / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # mean thrust of this revolution -> new inflow (relaxed)
        alpha, fz, fx = sections(psi_k[:, None], states[:, :1], states[:, 1:], *infl)
        Fz = _trapz(fz, x, axis=1)
        CT = N * np.mean(Fz * np.cos(states[:, 0])) * 0.5 * R / A
        lam_new = ForwardInflow(CT, mu, lam_fs) - lam_fs
        change = max(np.max(np.abs(y - y_start)), abs(lam_new - lam_i))
        lam_i += 0.5 * (lam_new - lam_i)
        if change < tol:
            break
    else:
        raise RuntimeError('Simulation: no periodic solution after %d revolutions' % max_rev)

    # mean hub loads over the last revolution (shaft frame, see module notes)
    q_tip = 0.5 * rho * WR ** 2
    beta, cpsi, spsi = states[:, 0], np.cos(psi_k), np.sin(psi_k)
    Fz = q_tip * R * _trapz(fz, x, axis=1)        # per blade, per azimuth
    Fx = q_tip * R * _trapz(fx, x, axis=1)
    Qb = q_tip * R ** 2 * _trapz(fx * x, x, axis=1)
    T = N * np.mean(Fz * np.cos(beta))
    H = N * np.mean(-Fz * np.sin(beta) * cpsi + Fx * spsi)
    Y = N * np.mean(-Fz * np.sin(beta) * spsi - Fx * cpsi)
    Q = N * np.mean(Qb)
    Mx = N * np.mean(K_flap * beta * spsi)
    My = N * np.mean(-K_flap * beta * cpsi)
    roll, pitch = -Mx, My
    f_ref, m_ref = rho * A * WR ** 2, rho * A * WR ** 2 * R

    res = dict(method='simulation', inflow=inflow, th=tuple(np.degrees([th0, th1s, th1c])),
               V=V, alpha_s=alpha_s, W=W, rpm=W * 60.0 / (2.0 * np.pi), R=R, N=int(N),
               mu=mu, lam_fs=lam_fs, lam_i=lam_i, lam=lam_fs + lam_i,
               CT=T / f_ref, CX=H / f_ref, CY=Y / f_ref, CQ=Q / m_ref, CP=Q / m_ref,
               CL=roll / m_ref, CM=pitch / m_ref,
               T=T, H=H, Y=Y, Q=Q, P=Q * W, roll=roll, pitch=pitch,
               beta0=np.degrees(np.mean(beta)),
               beta1c=np.degrees(2.0 * np.mean(beta * cpsi)),
               beta1s=np.degrees(2.0 * np.mean(beta * spsi)),
               psi=np.degrees(psi_k), beta=np.degrees(beta), x=x, r=b['r'],
               alpha_map=np.degrees(alpha), sigma=b['sigma'], CT_sigma=T / f_ref / b['sigma'],
               I_b=I_b, I_rotor=int(N) * I_b, revolutions=rev + 1,
               M_tip=WR / a_sound, M_adv=(WR + V * np.cos(a_s)) / a_sound)

    if verbose:
        print('Rotor simulator mode, %s inflow (edgewise flight)' % INFLOW_MODELS[inflow])
        print('  V = %.2f m/s   alpha_s = %.2f deg   mu = %.4f   rpm = %.0f   M_adv = %.3f'
              % (V, alpha_s, mu, res['rpm'], res['M_adv']))
        print('  theta0 = %.3f   theta1s = %.3f   theta1c = %.3f deg   gamma = %.2f   nu_beta = %.3f'
              % (res['th'] + (gamma, nu_beta)))
        print('  lam = %.5f (freestream %.5f, induced %.5f)   revolutions to converge = %d'
              % (res['lam'], lam_fs, lam_i, res['revolutions']))
        print('  beta0 = %.3f   beta1c = %.3f   beta1s = %.3f deg'
              % (res['beta0'], res['beta1c'], res['beta1s']))
        print('  CT = %.6f   CX = %.7f   CY = %.7f   CQ = %.7f   CL = %.7f   CM = %.7f'
              % (res['CT'], res['CX'], res['CY'], res['CQ'], res['CL'], res['CM']))
        print('  T = %.3f N   H = %.3f N   Y = %.3f N   Q = %.4f N m   P = %.2f W'
              % (T, H, Y, Q, res['P']))
        print('  roll = %.4f N m   pitch = %.4f N m' % (roll, pitch))
    return res


def GyroscopicMoment(I_rotor, W, omega_body, axis=(0.0, 0.0, 1.0)):
    '''
    Gyroscopic moment of a spinning rotor on the airframe [N m].

    I_rotor    : polar moment of inertia of the rotor [kg m^2]
    W          : rotor speed [rad/s], positive counter-clockwise about axis
    omega_body : airframe angular velocity [rad/s] (same frame as axis)
    The rotor angular momentum is H = I_rotor W axis; turning it with the
    airframe needs dH/dt = omega x H, so the moment on the airframe is
    -omega x H.
    '''
    axis = np.asarray(axis, dtype=float)
    H = I_rotor * W * axis / np.linalg.norm(axis)
    return -np.cross(np.asarray(omega_body, dtype=float), H)


if __name__ == '__main__':
    # Example: a 0.38 m two-bladed lift rotor at 3500 rpm, sea level
    foil = Airfoil(a0=5.73, alpha_L0=-2.0, Cl_max=1.3, Cd0=0.011, Cd2=0.6)
    W = 3500.0 * 2.0 * np.pi / 60.0
    blade = dict(n=50, N=2, airfoil=foil, c=0.055, twist=-15.0, tap=0.6, R=0.38, rcut=0.06)
    for model in DESIGN_MODELS:
        Design(model, W, 15.0, verbose=True, **blade)
        print()
    DesignTrim(80.0, 'bemt_tip', W, verbose=True, **blade)
    print()
    Simulation(W, (15.0, 0.0, 0.0), V=15.0, alpha_s=-5.0, verbose=True, **blade)
