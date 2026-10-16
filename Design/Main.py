import io
import sys
import argparse
import warnings
from contextlib import redirect_stdout

import numpy as np

import WeightEstimate as WE
import Aerodynamics as AE
import BEM

'''
Master file of the BlackBird design code -- this is the file to run.
    python Main.py                      opens the GUI
    python Main.py --cli                runs every analysis on AIRCRAFT and
                                        prints the results
    python Main.py --cli --method vlm --rotor-model uniform
                                        same, with other aerodynamics / rotor
                                        models

AIRCRAFT holds the whole aircraft definition in one place. The run_*
functions take that dict and route every piece to the module that needs it:
    run_weight -> WeightEstimate (Fuselage, Wing, Assemble)
    run_aero   -> Aerodynamics (WingAero)
    run_rotor  -> BEM (Design, DesignTrim, Simulation), which uses
                  MomentumCode for the inflow
The GUI builds the same dict from its input fields (its defaults come from
AIRCRAFT, FLIGHT, SETTINGS and ROTOR below) and calls these same functions,
so the GUI and the console always run the same code.

Adding a module: put its inputs in AIRCRAFT (or in its own settings dict),
write a run_<module>(aircraft, ...) function here returning a dict with a
'log' text, and call it from run_all() and from a GUI tab.
Not connected yet: AeroElasticity, MissionPlanner, Material_Properties.

Aircraft frame: origin at the fuselage nose tip, +x aft, +y right, +z up.

AIRCRAFT['surfaces'] (wing-like bodies):
kind     = 'mirrored' : left/right pair, S and b are for both sides (b tip to tip)
           'vertical' : single fin, b is its height
include  = False leaves the surface out of every analysis
N        = no of ribs (per side for mirrored surfaces)
x, y, z  = root leading edge position in the aircraft frame
sweep    = leading edge sweep [deg]; dihedral [deg]; twist = tip twist [deg]
           (all used by the aerodynamics and the drawing. In the weight
           estimate, sweep and dihedral shift the surface CG (sheared
           planform) but the masses still come from the straight Wing())
airfoil  = keyword arguments of Aerodynamics.Airfoil

AIRCRAFT['rotors'] (groups of identical rotors):
axis      = 'up' (lift rotor: hover / climb) or 'forward' (propeller)
positions = hub positions [(x, y, z), ...] in the aircraft frame, one per rotor
R, rcut   = radius and root cutout [m];  N = blades;  c = chord at the root
            cutout [m];  tap = taper ratio;  twist = root to tip twist [deg]
rpm       = rotor speed;  theta = collective at the root cutout [deg] (lift
            rotors use it only when ROTOR['trim'] is False)
mass      = mass of one rotor with its motor [kg] (goes into the weight)
airfoil   = keyword arguments of Aerodynamics.Airfoil (Cd0, Cd1, Cd2 used)
'''


AIRCRAFT = {
    'fuselage': dict(L=2.5, D=0.3, Ln=0.4, Lt=0.9, lmb_t=0.2, N=8,
                     rho_sk=1600.0, t=0.0008,
                     rho_fr=1600.0, t_fr=0.003, h_fr=0.02,
                     n_long=4, rho_l=1600.0, A_l=4e-5, nose='parabolic'),
    'surfaces': {
        'wing':   dict(kind='mirrored', include=True,
                       S=3.0, b=5.0, lmb=0.5, N=9, x=0.9, y=0.0, z=0.1,
                       sweep=0.0, dihedral=0.0, twist=0.0,
                       airfoil=dict(a0=2 * np.pi, alpha_L0=-2.0, Cm0=-0.05,
                                    Cl_max=1.4, tc=0.12, x_tc=0.3)),
        'h-tail': dict(kind='mirrored', include=True,
                       S=0.4, b=1.0, lmb=0.6, N=4, x=2.15, y=0.0, z=0.05,
                       sweep=0.0, dihedral=0.0, twist=0.0,
                       airfoil=dict(a0=2 * np.pi, alpha_L0=0.0, Cm0=0.0,
                                    Cl_max=1.2, tc=0.10, x_tc=0.3)),
        'fin':    dict(kind='vertical', include=True,
                       S=0.15, b=0.4, lmb=0.5, N=4, x=2.1, y=0.0, z=0.1,
                       sweep=0.0, dihedral=0.0, twist=0.0,
                       airfoil=dict(a0=2 * np.pi, alpha_L0=0.0, Cm0=0.0,
                                    Cl_max=1.2, tc=0.10, x_tc=0.3)),
    },
    # structure shared by every wing-like body (keyword arguments of Wing)
    'structure': dict(rho_sk=1600.0, t=0.001, Pc=2.1,
                      rho_rib=1600.0, t_rib=0.002,
                      rho_s=2700.0, t_strut=0.005,
                      rho_beam=2700.0, Ac=0.001),
    'rotors': {
        'lift': dict(include=True, axis='up',
                     positions=[(0.45, 0.9, 0.2), (0.45, -0.9, 0.2),
                                (2.0, 0.9, 0.2), (2.0, -0.9, 0.2)],
                     R=0.38, rcut=0.06, N=2, c=0.055, tap=0.6, twist=-15.0,
                     rpm=3500.0, theta=20.0, mass=0.6,
                     airfoil=dict(a0=5.73, alpha_L0=-2.0, Cl_max=1.3,
                                  Cd0=0.011, Cd1=0.0, Cd2=0.6)),
        'prop': dict(include=True, axis='forward', positions=[(-0.05, 0.0, 0.0)],
                     R=0.25, rcut=0.05, N=2, c=0.04, tap=0.6, twist=-28.0,
                     rpm=5000.0, theta=36.0, mass=0.3,
                     airfoil=dict(a0=5.73, alpha_L0=-2.0, Cl_max=1.3,
                                  Cd0=0.011, Cd1=0.0, Cd2=0.6)),
    },
    # (name, mass [kg], (x, y, z) [m]) in the aircraft frame
    'point_masses': [('motor',   0.8, (0.05, 0.0, 0.0)),
                     ('battery', 3.0, (0.7, 0.0, -0.03)),
                     ('payload', 2.0, (1.0, 0.0, -0.05))],
}

# flight condition; alpha = (start, stop, step) [deg]
FLIGHT = dict(V=20.0, h=500.0, alpha=(-4.0, 14.0, 1.0))

SETTINGS = dict(aero_method='lifting_line')

# rotor analysis settings
# model       : design mode model, see BEM.DESIGN_MODELS
# n           : radial stations per blade
# trim        : lift rotors carry the total weight in hover / climb
# climb       : climb speed of the lift rotors [m/s] (propellers fly at FLIGHT['V'])
# theta_sweep : collective sweep (start, stop, step) [deg] for the curves
# simulate    : rotor group run in simulator mode (None = no simulation)
# sim         : simulator inputs; theta0 = None uses the design collective
ROTOR = dict(model='bemt_tip', n=40, trim=True, climb=0.0,
             theta_sweep=(0.0, 40.0, 2.0), simulate='lift',
             sim=dict(V=15.0, alpha_s=-5.0, theta0=None, theta1s=0.0, theta1c=0.0,
                      inflow='drees', gamma=8.0, nu_beta=1.0))

G = 9.80665


def _warning_text(caught):
    return ''.join('WARNING: %s\n' % w.message for w in caught)


def alpha_range(alpha):
    '''(start, stop, step) [deg] -> array including stop'''
    start, stop, step = (float(v) for v in alpha)
    if step <= 0 or stop < start:
        raise ValueError('alpha range: need step > 0 and stop >= start')
    return np.arange(start, stop + 0.5 * step, step)


def run_weight(aircraft, verbose=False):
    '''
    Weight and CG of the whole aircraft.

    Mirrored surfaces are passed to Wing() as one half (S/2, b/2) and
    mirrored by Assemble(); vertical surfaces are rotated with VERTICAL.

    Returns
    -------
    dict: W [kg], CG [m] (aircraft frame), rows (placed components, see
          Assemble), log (breakdown text printed by the weight functions)
    '''
    log = io.StringIO()
    with redirect_stdout(log), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        W_f, CG_f = WE.Fuselage(verbose=True, **aircraft['fuselage'])
        components = {'fuselage': (W_f, CG_f)}

        for name, s in aircraft['surfaces'].items():
            if not s.get('include', True):
                continue
            if s['kind'] not in ('mirrored', 'vertical'):
                raise ValueError("surface %s: kind must be 'mirrored' or 'vertical'" % name)
            mirrored = s['kind'] == 'mirrored'
            S, b = (s['S'] / 2, s['b'] / 2) if mirrored else (s['S'], s['b'])
            print('\n[%s]' % name)
            W, CG = WE.Wing(S=S, b=b, lmb=s['lmb'], N=s['N'], verbose=True,
                            **aircraft['structure'])
            # sweep and dihedral shear the panel: every point moves by
            # y tan(sweep) in x and y tan(dihedral) in z (masses unchanged)
            shear = np.array([[1.0, np.tan(np.deg2rad(s.get('sweep', 0.0))), 0.0],
                              [0.0, 1.0, 0.0],
                              [0.0, np.tan(np.deg2rad(s.get('dihedral', 0.0))), 1.0]])
            components[name] = dict(mass=W, cg=CG, offset=(s['x'], s['y'], s['z']),
                                    mirror=mirrored,
                                    rotation=shear if mirrored else WE.VERTICAL @ shear)

        # every rotor (with its motor) is a point mass at its hub
        for name, rot in aircraft.get('rotors', {}).items():
            if rot.get('include', True):
                for i, p in enumerate(rot['positions']):
                    components['%s #%d' % (name, i + 1)] = (rot['mass'], p)

        for name, m, p in aircraft['point_masses']:
            if name in components:
                raise ValueError('Point mass name "%s" is already used by a '
                                 'component; please rename it' % name)
            components[name] = (m, p)

        print()
        W, CG, rows = WE.Assemble(components, verbose=True, breakdown=True)

    text = log.getvalue() + _warning_text(caught)
    if verbose:
        print(text, end='')
    return dict(W=W, CG=CG, rows=rows, log=text)


def run_aero(aircraft, flight, method='lifting_line', surfaces=None, verbose=False):
    '''
    Aerodynamics of every included mirrored surface, each on its own
    (isolated wing, no fuselage or downwash yet).

    Parameters
    ----------
    flight   : dict(V=..., h=..., alpha=(start, stop, step))
    method   : 'lifting_line', 'semi_empirical' or 'vlm'
    surfaces : names to analyse (default: all included mirrored surfaces)

    Returns
    -------
    dict: surfaces = {name: WingAero results}, log (text)
    '''
    alpha = alpha_range(flight['alpha'])
    results = {}
    log = io.StringIO()
    with redirect_stdout(log), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for name, s in aircraft['surfaces'].items():
            if surfaces is not None and name not in surfaces:
                continue
            if not s.get('include', True):
                continue
            if s['kind'] != 'mirrored':
                print('[%s] vertical surfaces are not analysed yet\n' % name)
                continue
            print('[%s]' % name)
            results[name] = AE.WingAero(
                S=s['S'], b=s['b'], lmb=s['lmb'], alpha=alpha,
                V=flight['V'], h=flight['h'],
                sweep=s.get('sweep', 0.0), dihedral=s.get('dihedral', 0.0),
                twist=s.get('twist', 0.0), airfoil=AE.Airfoil(**s['airfoil']),
                method=method, verbose=True)
            print()

    text = log.getvalue() + _warning_text(caught)
    if verbose:
        print(text, end='')
    return dict(surfaces=results, log=text)


def run_rotor(aircraft, flight, settings=ROTOR, weight=None, verbose=False):
    '''
    Rotor performance of every included rotor group.

    Design mode (axial flow), per group:
    * lift rotors (axis 'up'): climb speed settings['climb']; with
      settings['trim'] and a weight result, the collective is trimmed so all
      lift rotors together carry the weight (W g / number of lift rotors
      each), otherwise rotor['theta'] is used
    * propellers (axis 'forward'): flight speed FLIGHT['V'] at rotor['theta']
    * a collective sweep (settings['theta_sweep']) for the curves
    Simulator mode: group settings['simulate'] in edgewise flight with the
    settings['sim'] inputs (theta0 = None uses the design collective).
    Air density and speed of sound from ISA at flight['h'].

    Returns
    -------
    dict: rotors = {name: dict(op=design results at the operating point,
                               sweep=dict(theta, T, P, FM, eta, CT_sigma),
                               axis, count, T_req, T_total, P_total)},
          sim = Simulation results or None, log (text)
    '''
    _, _, rho, _, a_sound = AE.ISA(flight['h'])
    rotors = {name: r for name, r in aircraft.get('rotors', {}).items()
              if r.get('include', True)}
    n_lift = sum(len(r['positions']) for r in rotors.values() if r['axis'] == 'up')
    model = settings['model']
    results, sim = {}, None

    log = io.StringIO()
    with redirect_stdout(log), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for name, rot in rotors.items():
            if rot['axis'] not in ('up', 'forward'):
                raise ValueError("rotor %s: axis must be 'up' or 'forward'" % name)
            if len(rot['positions']) == 0:
                raise ValueError('rotor %s: needs at least one position' % name)
            W = rot['rpm'] * 2.0 * np.pi / 60.0
            blade = dict(n=settings['n'], N=rot['N'], airfoil=AE.Airfoil(**rot['airfoil']),
                         c=rot['c'], twist=rot['twist'], tap=rot['tap'], R=rot['R'],
                         rcut=rot['rcut'])
            air = dict(rho=rho, a_sound=a_sound)
            lift = rot['axis'] == 'up'
            V = settings['climb'] if lift else flight['V']
            T_req = None
            print('[%s] %d x %s' % (name, len(rot['positions']),
                                    'lift rotor' if lift else 'propeller'))
            if lift and settings['trim'] and weight is not None:
                T_req = weight['W'] * G / n_lift
                print('  trimmed to carry the weight: %.2f kg x g / %d lift rotors = %.2f N each'
                      % (weight['W'], n_lift, T_req))
                op = BEM.DesignTrim(T_req, model, W, V=V, verbose=True, **air, **blade)
            else:
                op = BEM.Design(model, W, rot['theta'], V=V, verbose=True, **air, **blade)

            thetas = alpha_range(settings['theta_sweep'])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                runs = [BEM.Design(model, W, th, V=V, **air, **blade) for th in thetas]
            sweep = {key: np.array([r[key] for r in runs])
                     for key in ('T', 'P', 'FM', 'eta', 'CT_sigma')}
            sweep['theta'] = thetas
            count = len(rot['positions'])
            results[name] = dict(op=op, sweep=sweep, axis=rot['axis'], count=count,
                                 T_req=T_req, T_total=op['T'] * count, P_total=op['P'] * count)
            print('  all %d: T = %.2f N   P = %.1f W\n' % (count, op['T'] * count, op['P'] * count))

        target = settings.get('simulate')
        if target is not None and target in results:
            rot, s = rotors[target], settings['sim']
            theta0 = s['theta0'] if s['theta0'] is not None else results[target]['op']['theta']
            print('[%s] simulator' % target)
            sim = BEM.Simulation(rot['rpm'] * 2.0 * np.pi / 60.0,
                                 (theta0, s['theta1s'], s['theta1c']),
                                 n=settings['n'], N=rot['N'], airfoil=AE.Airfoil(**rot['airfoil']),
                                 c=rot['c'], twist=rot['twist'], tap=rot['tap'], R=rot['R'],
                                 rcut=rot['rcut'], V=s['V'], alpha_s=s['alpha_s'],
                                 rho=rho, a_sound=a_sound, inflow=s['inflow'],
                                 gamma=s['gamma'], nu_beta=s['nu_beta'], verbose=True)
            sim['rotor'] = target
            print()

    text = log.getvalue() + _warning_text(caught)
    if verbose:
        print(text, end='')
    return dict(rotors=results, sim=sim, log=text)


def run_all(aircraft=AIRCRAFT, flight=FLIGHT, method=SETTINGS['aero_method'],
            rotor=ROTOR, verbose=True):
    '''Every connected analysis on one aircraft definition.'''
    out = {}
    if verbose:
        print('=' * 30 + ' WEIGHT & CG ' + '=' * 30)
    out['weight'] = run_weight(aircraft, verbose=verbose)
    if verbose:
        print('\n' + '=' * 30 + ' AERODYNAMICS ' + '=' * 29)
    out['aero'] = run_aero(aircraft, flight, method=method, verbose=verbose)
    if verbose:
        print('\n' + '=' * 30 + ' ROTORS ' + '=' * 35)
    out['rotor'] = run_rotor(aircraft, flight, rotor, weight=out['weight'], verbose=verbose)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description='BlackBird design code')
    parser.add_argument('--cli', action='store_true',
                        help='run the analyses in the console instead of opening the GUI')
    parser.add_argument('--method', choices=sorted(AE.METHODS),
                        default=SETTINGS['aero_method'],
                        help='aerodynamics method (default: %(default)s)')
    parser.add_argument('--rotor-model', choices=sorted(BEM.DESIGN_MODELS),
                        default=ROTOR['model'],
                        help='rotor design mode model (default: %(default)s)')
    args = parser.parse_args(argv)

    if args.cli:
        run_all(AIRCRAFT, FLIGHT, method=args.method,
                rotor=dict(ROTOR, model=args.rotor_model), verbose=True)
    else:
        # the GUI does "import Main"; register this running script under that
        # name so it is not loaded a second time
        sys.modules.setdefault('Main', sys.modules[__name__])
        import GUI_AIHELPHELPME
        GUI_AIHELPHELPME.main()


if __name__ == '__main__':
    main()
