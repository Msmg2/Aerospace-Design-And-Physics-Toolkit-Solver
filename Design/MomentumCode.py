import numpy as np

'''
Momentum theory (actuator disk) for rotors and propellers.
It gives the inflow that the blade element code (BEM.py) needs, plus quick
power estimates. All quantities are non-dimensional with the tip speed W R
and the disk area A = pi R^2.

Terminology:
CT    = thrust coefficient T / (rho A (W R)^2)
CP    = power coefficient P / (rho A (W R)^3)  (= torque coefficient CQ)
lam   = total inflow ratio through the disk, positive when the flow goes
        through the disk opposite to the thrust (downwards in hover)
lam_c = freestream (climb) part of the inflow, V_climb / (W R); for a
        propeller this is V / (W R)
lam_i = induced part, v_i / (W R)
lam_h = hover induced inflow sqrt(CT / 2)
mu    = edgewise advance ratio V cos(alpha_s) / (W R)
kappa = induced power factor (1 = ideal, ~1.15 for real rotors)
sigma = rotor solidity (blade area / disk area)
Cd0   = mean profile drag coefficient of the blade sections
'''


def HoverInflow(CT):
    '''Induced inflow in hover, lam_h = sqrt(CT / 2) (keeps the sign of CT).'''
    return np.sign(CT) * np.sqrt(abs(CT) / 2.0)


def _induced_axial(CT, lam_c):
    '''Induced inflow lam_i for CT >= 0 in axial flight (climb or descent).'''
    lam_h = HoverInflow(CT)
    if lam_h == 0.0:
        return 0.0
    x = lam_c / lam_h
    if x >= 0.0:
        # normal working state (hover, climb, propeller)
        return -0.5 * lam_c + np.sqrt((0.5 * lam_c) ** 2 + 0.5 * CT)
    if x <= -2.0:
        # windmill brake state
        return -0.5 * lam_c - np.sqrt((0.5 * lam_c) ** 2 - 0.5 * CT)
    # vortex ring / turbulent wake state: momentum theory is invalid here,
    # empirical fit (Leishman) with kappa = 1 so it joins the two branches
    k1, k2, k3, k4 = -1.125, -1.372, -1.718, -0.655
    return lam_h * (1.0 + k1 * x + k2 * x ** 2 + k3 * x ** 3 + k4 * x ** 4)


def AxialInflow(CT, lam_c=0.0, kappa=1.0):
    '''
    Total inflow lam = lam_c + kappa * lam_i in axial flight (flow along the
    shaft): hover, climb, descent and propellers.

    * climb / hover / propeller: lam_i = -lam_c/2 + sqrt((lam_c/2)^2 + CT/2)
    * descent faster than 2 v_h (windmill brake): the other momentum root
    * vortex ring state (-2 < V_c / v_h < 0): empirical curve fit, since
      momentum theory does not apply there
    * negative CT: same curves with thrust and flow reversed
    '''
    if CT >= 0.0:
        lam_i = _induced_axial(CT, lam_c)
    else:
        lam_i = -_induced_axial(-CT, -lam_c)
    return lam_c + kappa * lam_i


def ForwardInflow(CT, mu, lam_fs=0.0, kappa=1.0, tol=1e-10, max_iter=100):
    '''
    Glauert's forward flight inflow, solved with Newton-Raphson:
        lam = lam_fs + kappa CT / (2 sqrt(mu^2 + lam^2))
    lam_fs is the freestream part through the disk (positive down), e.g.
    -mu tan(alpha_s) for a shaft tilted aft by alpha_s. With mu = 0 it gives
    the axial (climb) result. Not valid in the vortex ring state.
    '''
    if CT == 0.0:
        return lam_fs
    lam = lam_fs + HoverInflow(CT)
    for _ in range(max_iter):
        s = np.sqrt(mu ** 2 + lam ** 2)
        f = lam - lam_fs - kappa * CT / (2.0 * s)
        df = 1.0 + kappa * CT * lam / (2.0 * s ** 3)
        step = f / df
        lam -= step
        if abs(step) < tol:
            return lam
    raise RuntimeError('ForwardInflow: Newton iteration did not converge')


def MomentumPower(CT, sigma, Cd0, mu=0.0, lam_fs=0.0, kappa=1.15):
    '''
    Power coefficient from momentum theory plus blade profile drag (energy
    method, Leishman):
        CP = kappa CT lam_i + CT lam_fs + sigma Cd0 / 8 (1 + 4.65 mu^2)
    induced + climb (or propulsive) + profile power
    '''
    lam = ForwardInflow(CT, mu, lam_fs, kappa=1.0)
    lam_i = lam - lam_fs
    return kappa * CT * lam_i + CT * lam_fs + sigma * Cd0 / 8.0 * (1.0 + 4.65 * mu ** 2)


def FigureOfMerit(CT, CP):
    '''Hover figure of merit: ideal power / actual power = CT^1.5 / (sqrt(2) CP).'''
    return CT ** 1.5 / (np.sqrt(2.0) * CP)


def IdealHoverPower(T, rho, R):
    '''Ideal (momentum) hover power [W] of a rotor of radius R carrying thrust T.'''
    return T ** 1.5 / np.sqrt(2.0 * rho * np.pi * R ** 2)


def FroudeEfficiency(T, V, rho, R):
    '''Ideal propulsive efficiency of a propeller (momentum theory).'''
    Tc = T / (0.5 * rho * V ** 2 * np.pi * R ** 2)
    return 2.0 / (1.0 + np.sqrt(1.0 + Tc))


if __name__ == '__main__':
    # Example: hover and climb inflow, and the power of a small lift rotor
    CT, sigma = 0.008, 0.06
    print('lam_h = %.5f' % HoverInflow(CT))
    for Vc_vh in (1.0, 0.0, -1.0, -3.0):
        lam_c = Vc_vh * HoverInflow(CT)
        print('V_c/v_h = %5.1f   lam_i/lam_h = %.4f'
              % (Vc_vh, (AxialInflow(CT, lam_c) - lam_c) / HoverInflow(CT)))
    CP = MomentumPower(CT, sigma, Cd0=0.011)
    print('hover CP = %.6f   FM = %.3f' % (CP, FigureOfMerit(CT, CP)))
