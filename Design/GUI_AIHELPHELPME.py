import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import Main
from WeightEstimate import FuselageRadius
from Aerodynamics import METHOD_LABELS
from BEM import DESIGN_MODELS, INFLOW_MODELS

'''
Placeholder GUI for the BlackBird design code (Tkinter + matplotlib).
Start it through the master file:  python Main.py

Layout
-> Left  : the aircraft definition (fuselage, wing-like bodies with their
           airfoils, structure, rotors, point masses). It has the same
           structure as Main.AIRCRAFT, and its defaults come from there.
-> Right : result tabs
   Weight & CG    : top / side view with every component CG, the rotor
                    disks and the total CG, plus the breakdown text
   Aerodynamics   : flight condition and method, CL-alpha, drag polar, L/D,
                    spanwise cl, plus the summary text (mirrored surfaces)
   Rotors         : design mode (thrust, power, FM / efficiency against
                    collective, local angle of attack along the blade) and
                    simulator mode (angle of attack over the disk, flapping)
   AeroElasticity, Mission, Materials : empty placeholders

Calculate builds the aircraft dict from the inputs and calls Main.run_weight,
Main.run_aero and Main.run_rotor, so the GUI never calls the analysis modules
directly. The rotors use the altitude (and the propellers the speed) of the
Aerodynamics tab.

To add an input: add a line to the matching *_FIELDS list below and the
default value to Main.AIRCRAFT (the key must be the keyword argument name).
'''


# (key, label, type) -- key is the keyword argument of the analysis function
FUSELAGE_FIELDS = [
    ('L',      'Length L [m]',               float),
    ('D',      'Max diameter D [m]',         float),
    ('Ln',     'Nose length Ln [m]',         float),
    ('Lt',     'Tail cone length Lt [m]',    float),
    ('lmb_t',  'Tail end dia. / D',          float),
    ('N',      'Frames N',                   int),
    ('rho_sk', 'Skin density [kg/m³]',       float),
    ('t',      'Skin thickness [m]',         float),
    ('rho_fr', 'Frame density [kg/m³]',      float),
    ('t_fr',   'Frame thickness [m]',        float),
    ('h_fr',   'Frame depth [m]',            float),
    ('n_long', 'Longerons',                  int),
    ('rho_l',  'Longeron density [kg/m³]',   float),
    ('A_l',    'Longeron area [m²]',         float),
]

# shared by every wing-like body
STRUCTURE_FIELDS = [
    ('rho_sk',   'Skin density [kg/m³]',     float),
    ('t',        'Skin thickness [m]',       float),
    ('Pc',       'Perimeter / chord',        float),
    ('rho_rib',  'Rib density [kg/m³]',      float),
    ('t_rib',    'Rib thickness [m]',        float),
    ('rho_s',    'Strut density [kg/m³]',    float),
    ('t_strut',  'Strut side [m]',           float),
    ('rho_beam', 'Beam density [kg/m³]',     float),
    ('Ac',       'Beam area / chord²',       float),
]

AIRFOIL_FIELDS = [
    ('a0',       'Lift slope a0 [1/rad]',    float),
    ('alpha_L0', 'Zero-lift angle [deg]',    float),
    ('Cm0',      'Cm0 (quarter chord)',      float),
    ('Cl_max',   'Cl_max',                   float),
    ('tc',       'Thickness t/c',            float),
    ('x_tc',     'Max thickness at x/c',     float),
]


ROTOR_FIELDS = [
    ('R',     'Radius R [m]',             float),
    ('rcut',  'Root cutout [m]',          float),
    ('N',     'Blades N',                 int),
    ('c',     'Chord at root cutout [m]', float),
    ('tap',   'Taper c_tip/c_root',       float),
    ('twist', 'Twist root to tip [deg]',  float),
    ('rpm',   'Speed [rpm]',              float),
    ('theta', 'Collective at root [deg]', float),
    ('mass',  'Mass each, with motor [kg]', float),
]

ROTOR_AIRFOIL_FIELDS = [
    ('a0',       'Lift slope a0 [1/rad]',  float),
    ('alpha_L0', 'Zero-lift angle [deg]',  float),
    ('Cl_max',   'Cl_max',                 float),
    ('Cd0',      'Cd0',                    float),
    ('Cd1',      'Cd1 [1/rad]',            float),
    ('Cd2',      'Cd2 [1/rad²]',           float),
]


def fmt_positions(positions):
    return '; '.join(', '.join(fmt(float(v)) for v in p) for p in positions)


def parse_positions(text, where):
    '''"x, y, z; x, y, z" -> [(x, y, z), ...]'''
    try:
        points = [tuple(float(v) for v in item.split(','))
                  for item in text.split(';') if item.strip()]
    except ValueError:
        points = None
    if not points or any(len(p) != 3 for p in points):
        raise ValueError('%s: write the positions as "x, y, z; x, y, z; ..."' % where)
    return points


def surface_fields(kind):
    if kind == 'mirrored':
        labels = ('Area S, both sides [m²]', 'Span b, tip to tip [m]', 'Ribs per side')
    else:
        labels = ('Area S [m²]', 'Height b [m]', 'Ribs N')
    return [
        ('S',        labels[0],                  float),
        ('b',        labels[1],                  float),
        ('lmb',      'Taper ratio c_tip/c_root', float),
        ('N',        labels[2],                  int),
        ('x',        'Root LE x [m]',            float),
        ('y',        'Root LE y [m]',            float),
        ('z',        'Root LE z [m]',            float),
        ('sweep',    'LE sweep [deg]',           float),
        ('dihedral', 'Dihedral [deg]',           float),
        ('twist',    'Tip twist [deg]',          float),
    ]


def fmt(value):
    return str(value) if isinstance(value, int) else '%.10g' % value


def parse(text, typ, where):
    try:
        return typ(text.strip())
    except ValueError:
        kind = 'whole number' if typ is int else 'number'
        raise ValueError('%s: "%s" is not a valid %s' % (where, text.strip(), kind))


class FieldGroup(ttk.LabelFrame):
    '''A titled block of labelled entries; values() parses them.'''

    def __init__(self, parent, title, specs, defaults, optional=False):
        super().__init__(parent, text=title, padding=6)
        self.title = title
        self.specs = specs
        self.defaults = defaults
        self.vars = {}
        row = 0
        self.include = tk.BooleanVar(value=defaults.get('include', True))
        if optional:
            ttk.Checkbutton(self, text='Include', variable=self.include).grid(
                row=row, column=0, columnspan=2, sticky='w')
            row += 1
        for key, label, _ in specs:
            ttk.Label(self, text=label).grid(row=row, column=0, sticky='w', padx=(0, 8), pady=1)
            var = tk.StringVar(value=fmt(defaults[key]))
            ttk.Entry(self, textvariable=var, width=10).grid(row=row, column=1, sticky='e', pady=1)
            self.vars[key] = var
            row += 1
        self.next_row = row                       # for extra widgets below

    def values(self):
        return {key: parse(self.vars[key].get(), typ, '%s -> %s' % (self.title, label))
                for key, label, typ in self.specs}

    def reset(self):
        self.include.set(self.defaults.get('include', True))
        for key, _, _ in self.specs:
            self.vars[key].set(fmt(self.defaults[key]))


class PointMassEditor(ttk.LabelFrame):
    '''Editable list of point masses (name, mass, position).'''

    COLUMNS = ('name', 'mass', 'x', 'y', 'z')

    def __init__(self, parent, defaults):
        super().__init__(parent, text='Point masses (aircraft frame)', padding=6)
        self.defaults = defaults
        self.tree = ttk.Treeview(self, columns=self.COLUMNS, show='headings',
                                 height=5, selectmode='browse')
        for col, head, width in zip(self.COLUMNS,
                                    ('Name', 'Mass [kg]', 'x [m]', 'y [m]', 'z [m]'),
                                    (80, 65, 50, 50, 50)):
            self.tree.heading(col, text=head)
            self.tree.column(col, width=width, anchor='w' if col == 'name' else 'e')
        self.tree.grid(row=0, column=0, columnspan=5, sticky='ew')
        self.tree.bind('<<TreeviewSelect>>', self._on_select)

        self.vars = {}
        for i, col in enumerate(self.COLUMNS):
            ttk.Label(self, text=col).grid(row=1, column=i, sticky='w')
            var = tk.StringVar()
            ttk.Entry(self, textvariable=var, width=9 if col == 'name' else 6).grid(
                row=2, column=i, sticky='ew', padx=1)
            self.vars[col] = var

        buttons = ttk.Frame(self)
        buttons.grid(row=3, column=0, columnspan=5, sticky='w', pady=(4, 0))
        ttk.Button(buttons, text='Add', command=self.add).pack(side='left')
        ttk.Button(buttons, text='Update', command=self.update_selected).pack(side='left', padx=4)
        ttk.Button(buttons, text='Remove', command=self.remove).pack(side='left')
        self.reset()

    def reset(self):
        self.tree.delete(*self.tree.get_children())
        for name, m, (x, y, z) in self.defaults:
            self.tree.insert('', 'end', values=(name, m, x, y, z))

    def _on_select(self, _event=None):
        sel = self.tree.selection()
        if sel:
            for col, val in zip(self.COLUMNS, self.tree.item(sel[0], 'values')):
                self.vars[col].set(val)

    def _read_entries(self, skip=None):
        name = self.vars['name'].get().strip()
        if not name:
            raise ValueError('Point mass needs a name')
        for item in self.tree.get_children():
            if item != skip and self.tree.item(item, 'values')[0] == name:
                raise ValueError('Point mass "%s" already exists' % name)
        try:
            nums = [float(self.vars[col].get()) for col in self.COLUMNS[1:]]
        except ValueError:
            raise ValueError('Point mass "%s": mass, x, y, z must be numbers' % name)
        if nums[0] < 0:
            raise ValueError('Point mass "%s": mass must be >= 0' % name)
        return [name] + nums

    def add(self):
        try:
            self.tree.insert('', 'end', values=self._read_entries())
        except ValueError as err:
            messagebox.showerror('Point masses', str(err))

    def update_selected(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo('Point masses', 'Select a point mass in the list first')
            return
        try:
            self.tree.item(sel[0], values=self._read_entries(skip=sel[0]))
        except ValueError as err:
            messagebox.showerror('Point masses', str(err))

    def remove(self):
        for item in self.tree.selection():
            self.tree.delete(item)

    def items(self):
        '''list of (name, mass, (x, y, z)), the Main.AIRCRAFT format'''
        out = []
        for item in self.tree.get_children():
            name, m, x, y, z = self.tree.item(item, 'values')
            out.append((name, float(m), (float(x), float(y), float(z))))
        return out


class ScrollableFrame(ttk.Frame):
    '''Vertically scrollable container; put widgets in self.inner.'''

    def __init__(self, parent):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0)
        bar = ttk.Scrollbar(self, orient='vertical', command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=bar.set)
        self.inner = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.inner, anchor='nw')
        self.inner.bind('<Configure>', lambda e: self.canvas.configure(
            scrollregion=self.canvas.bbox('all'), width=self.inner.winfo_reqwidth()))
        self.canvas.grid(row=0, column=0, sticky='nsew')
        bar.grid(row=0, column=1, sticky='ns')
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.bind_all('<MouseWheel>', self._on_wheel, add='+')

    def _on_wheel(self, event):
        # scroll only while the pointer is over this panel
        widget = self.winfo_containing(event.x_root, event.y_root)
        if widget is not None and str(widget).startswith(str(self)):
            self.canvas.yview_scroll(int(-event.delta / 120), 'units')


class AircraftPanel(ScrollableFrame):
    '''Inputs for the whole aircraft; read() returns a Main.AIRCRAFT-like dict.'''

    def __init__(self, parent):
        super().__init__(parent)
        defaults = Main.AIRCRAFT
        panel = self.inner

        self.fuselage = FieldGroup(panel, 'Fuselage', FUSELAGE_FIELDS, defaults['fuselage'])
        ttk.Label(self.fuselage, text='Nose shape').grid(
            row=self.fuselage.next_row, column=0, sticky='w')
        self.nose = tk.StringVar(value=defaults['fuselage']['nose'])
        ttk.Combobox(self.fuselage, textvariable=self.nose, width=9, state='readonly',
                     values=('parabolic', 'conical')).grid(
            row=self.fuselage.next_row, column=1, sticky='e')
        self.fuselage.pack(fill='x', pady=(0, 6))

        self.surfaces = {}
        for name, s in defaults['surfaces'].items():
            kind = s['kind']
            title = '%s (%s)' % (name, 'mirrored left/right' if kind == 'mirrored' else 'vertical')
            group = FieldGroup(panel, title, surface_fields(kind), s, optional=True)
            foil = FieldGroup(group, 'Airfoil', AIRFOIL_FIELDS, s['airfoil'])
            foil.title = '%s airfoil' % name
            foil.grid(row=group.next_row, column=0, columnspan=2, sticky='ew', pady=(4, 0))
            group.pack(fill='x', pady=(0, 6))
            self.surfaces[name] = (kind, group, foil)

        self.structure = FieldGroup(panel, 'Wing-like body structure (all surfaces)',
                                    STRUCTURE_FIELDS, defaults['structure'])
        self.structure.pack(fill='x', pady=(0, 6))

        self.rotors = {}
        for name, r in defaults.get('rotors', {}).items():
            group = FieldGroup(panel, 'rotor: %s' % name, ROTOR_FIELDS, r, optional=True)
            row = group.next_row
            ttk.Label(group, text='Axis').grid(row=row, column=0, sticky='w')
            axis = tk.StringVar(value=r['axis'])
            ttk.Combobox(group, textvariable=axis, width=9, state='readonly',
                         values=('up', 'forward')).grid(row=row, column=1, sticky='e')
            ttk.Label(group, text='Hub positions  x, y, z; x, y, z; ...').grid(
                row=row + 1, column=0, columnspan=2, sticky='w', pady=(4, 0))
            positions = tk.StringVar(value=fmt_positions(r['positions']))
            ttk.Entry(group, textvariable=positions, width=34).grid(
                row=row + 2, column=0, columnspan=2, sticky='ew')
            foil = FieldGroup(group, 'Airfoil', ROTOR_AIRFOIL_FIELDS, r['airfoil'])
            foil.title = 'rotor %s airfoil' % name
            foil.grid(row=row + 3, column=0, columnspan=2, sticky='ew', pady=(4, 0))
            group.pack(fill='x', pady=(0, 6))
            self.rotors[name] = (group, axis, positions, foil)

        self.point_masses = PointMassEditor(panel, defaults['point_masses'])
        self.point_masses.pack(fill='x', pady=(0, 6))

    def read(self):
        fuselage = self.fuselage.values()
        fuselage['nose'] = self.nose.get()
        surfaces = {}
        for name, (kind, group, foil) in self.surfaces.items():
            surfaces[name] = dict(kind=kind, include=group.include.get(),
                                  airfoil=foil.values(), **group.values())
        rotors = {}
        for name, (group, axis, positions, foil) in self.rotors.items():
            rotors[name] = dict(include=group.include.get(), axis=axis.get(),
                                positions=parse_positions(positions.get(),
                                                          'rotor %s -> positions' % name),
                                airfoil=foil.values(), **group.values())
        return dict(fuselage=fuselage, surfaces=surfaces,
                    structure=self.structure.values(), rotors=rotors,
                    point_masses=self.point_masses.items())

    def reset(self):
        self.fuselage.reset()
        self.nose.set(Main.AIRCRAFT['fuselage']['nose'])
        for _, group, foil in self.surfaces.values():
            group.reset()
            foil.reset()
        self.structure.reset()
        for name, (group, axis, positions, foil) in self.rotors.items():
            group.reset()
            foil.reset()
            axis.set(Main.AIRCRAFT['rotors'][name]['axis'])
            positions.set(fmt_positions(Main.AIRCRAFT['rotors'][name]['positions']))
        self.point_masses.reset()


def text_box(parent, height):
    '''Monospace text widget with scrollbars; returns (frame, text).'''
    frame = ttk.Frame(parent)
    frame.columnconfigure(0, weight=1)
    frame.rowconfigure(0, weight=1)
    text = tk.Text(frame, font=('Consolas', 9), wrap='none', height=height)
    ybar = ttk.Scrollbar(frame, orient='vertical', command=text.yview)
    xbar = ttk.Scrollbar(frame, orient='horizontal', command=text.xview)
    text.configure(yscrollcommand=ybar.set, xscrollcommand=xbar.set)
    text.grid(row=0, column=0, sticky='nsew')
    ybar.grid(row=0, column=1, sticky='ns')
    xbar.grid(row=1, column=0, sticky='ew')
    return frame, text


class ResultTab(ttk.Frame):
    '''Summary line, matplotlib figure with toolbar, and a text box.'''

    def __init__(self, parent, top=None, figsize=(7, 5)):
        super().__init__(parent, padding=6)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=3)
        self.rowconfigure(4, weight=2)
        if top is not None:
            top(self).grid(row=0, column=0, sticky='ew', pady=(0, 4))
        self.summary = tk.StringVar(value='Press Calculate')
        ttk.Label(self, textvariable=self.summary, font=('Segoe UI', 10, 'bold'),
                  wraplength=900, justify='left').grid(row=1, column=0, sticky='w', pady=(0, 4))
        self.fig = Figure(figsize=figsize, layout='constrained')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().grid(row=2, column=0, sticky='nsew')
        toolbar_frame = ttk.Frame(self)
        toolbar_frame.grid(row=3, column=0, sticky='ew')
        NavigationToolbar2Tk(self.canvas, toolbar_frame).update()
        frame, self.text = text_box(self, height=10)
        frame.grid(row=4, column=0, sticky='nsew', pady=(4, 0))

    def set_text(self, text):
        self.text.delete('1.0', 'end')
        self.text.insert('1.0', text)


def planform(s):
    '''
    Outline of one wing-like body (one side) in the aircraft frame: root LE,
    root TE, tip TE, tip LE as (x, y, z) arrays. Same trapezoid as the
    analyses: chords from S and b, LE sweep, dihedral (mirrored surfaces).
    '''
    mirrored = s['kind'] == 'mirrored'
    S, b = (s['S'] / 2, s['b'] / 2) if mirrored else (s['S'], s['b'])
    c_root = 2.0 * S / (b * (1.0 + s['lmb']))
    c_tip = s['lmb'] * c_root
    dx = b * np.tan(np.deg2rad(s['sweep']))
    x = s['x'] + np.array([0.0, c_root, dx + c_tip, dx])
    if mirrored:
        y = s['y'] + np.array([0.0, 0.0, b, b])
        z = s['z'] + np.array([0.0, 0.0, 1.0, 1.0]) * b * np.tan(np.deg2rad(s['dihedral']))
    else:
        y = np.full(4, s['y'])
        z = s['z'] + np.array([0.0, 0.0, b, b])
    return x, y, z


class WeightTab(ResultTab):
    def __init__(self, parent):
        super().__init__(parent)
        self.ax_top = self.fig.add_subplot(2, 1, 1)
        self.ax_side = self.fig.add_subplot(2, 1, 2)

    def show(self, aircraft, result):
        W, CG = result['W'], result['CG']
        self.summary.set('Total mass %.3f kg     CG = (%.4f, %.4f, %.4f) m'
                         % (W, CG[0], CG[1], CG[2]))
        self.set_text(result['log'])
        self.draw(aircraft, result)

    def draw(self, aircraft, result):
        '''
        Top view is drawn nose-up (span across the plot, x downwards), so a
        long wing fits the wide plot; side view is nose-left.
        '''
        top, side = self.ax_top, self.ax_side
        top.clear()
        side.clear()
        for leg in self.fig.legends:
            leg.remove()

        # fuselage outline: same r(x) as the weight model
        fus = aircraft['fuselage']
        x = np.linspace(0.0, fus['L'], 200)
        r, _ = FuselageRadius(x, fus['L'], fus['D'], fus['Ln'], fus['Lt'],
                              fus['lmb_t'], fus['nose'])
        top.fill_betweenx(x, -r, r, color='0.85', ec='0.4', lw=1)
        side.fill_between(x, -r, r, color='0.85', ec='0.4', lw=1)

        # wing-like bodies
        style = dict(alpha=0.3, fc='tab:blue', ec='tab:blue')
        for s in aircraft['surfaces'].values():
            if not s['include']:
                continue
            px, py, pz = planform(s)
            if s['kind'] == 'mirrored':
                top.fill(py, px, **style)
                top.fill(-py, px, **style)
                side.fill(px, pz, lw=1.5, **style)
            else:
                side.fill(px, pz, **style)
                top.plot([s['y'], s['y']], [px.min(), px.max()], color='tab:blue', lw=2.5)

        # rotor disks: circles seen along the shaft, lines seen edge-on
        t = np.linspace(0.0, 2.0 * np.pi, 72)
        disk = dict(color='tab:purple', lw=1.2)
        for rot in aircraft.get('rotors', {}).values():
            if not rot['include']:
                continue
            R = rot['R']
            for x0, y0, z0 in rot['positions']:
                if rot['axis'] == 'up':
                    top.plot(y0 + R * np.cos(t), x0 + R * np.sin(t), **disk)
                    side.plot([x0 - R, x0 + R], [z0, z0], **disk)
                else:
                    top.plot([y0 - R, y0 + R], [x0, x0], **disk)
                    side.plot([x0, x0], [z0 - R, z0 + R], **disk)

        # component CGs (one colour per component, mirrored pairs and rotor
        # groups share it)
        colours = {}
        for name, m, p in result['rows']:
            base = name.replace(' (R)', '').replace(' (L)', '').split(' #')[0]
            first = base not in colours
            if first:
                colours[base] = 'C%d' % (len(colours) % 10)
            mark = dict(marker='o', ls='', ms=6, mec='k', mew=0.5, color=colours[base])
            top.plot(p[1], p[0], label=base if first else None, **mark)
            side.plot(p[0], p[2], **mark)
        CG = result['CG']
        top.plot(CG[1], CG[0], 'X', color='crimson', mec='k', ms=13, label='total CG')
        side.plot(CG[0], CG[2], 'X', color='crimson', mec='k', ms=13)

        top.set_title('Top view (nose up)', fontsize=10)
        top.set_xlabel('y (right) [m]')
        top.set_ylabel('x (aft) [m]')
        top.invert_yaxis()
        side.set_title('Side view', fontsize=10)
        side.set_xlabel('x (aft, from nose tip) [m]')
        side.set_ylabel('z (up) [m]')
        for ax in (top, side):
            ax.set_aspect('equal', adjustable='datalim')
            ax.grid(True, lw=0.4, alpha=0.6)
        self.fig.legend(loc='outside right center', fontsize=8, title='CG of')
        self.canvas.draw_idle()


class AeroTab(ResultTab):
    '''Flight condition and method on top; results of one surface below.'''

    def __init__(self, parent):
        self.method_keys = {label: key for key, label in METHOD_LABELS.items()}
        super().__init__(parent, top=self._controls, figsize=(7, 5.5))
        axes = self.fig.subplots(2, 2)
        self.ax_cl, self.ax_polar, self.ax_ld, self.ax_span = axes.ravel()
        self.aero, self.others, self.aircraft = None, {}, None

    def _controls(self, parent):
        frame = ttk.Frame(parent)
        flight = Main.FLIGHT
        self.method = tk.StringVar(value=METHOD_LABELS[Main.SETTINGS['aero_method']])
        self.vars = {}
        row1 = ttk.Frame(frame)
        row1.pack(fill='x')
        ttk.Label(row1, text='Method').pack(side='left')
        ttk.Combobox(row1, textvariable=self.method, state='readonly', width=15,
                     values=list(METHOD_LABELS.values())).pack(side='left', padx=(4, 12))
        for key, label, value in (('V', 'V [m/s]', flight['V']), ('h', 'h [m]', flight['h']),
                                  ('a0', 'α from [deg]', flight['alpha'][0]),
                                  ('a1', 'to', flight['alpha'][1]),
                                  ('da', 'step', flight['alpha'][2])):
            ttk.Label(row1, text=label).pack(side='left')
            self.vars[key] = tk.StringVar(value=fmt(value))
            ttk.Entry(row1, textvariable=self.vars[key], width=7).pack(side='left', padx=(4, 10))

        row2 = ttk.Frame(frame)
        row2.pack(fill='x', pady=(4, 0))
        ttk.Label(row2, text='Show surface').pack(side='left')
        self.surface = tk.StringVar()
        self.surface_box = ttk.Combobox(row2, textvariable=self.surface, state='readonly', width=10)
        self.surface_box.pack(side='left', padx=(4, 12))
        self.surface_box.bind('<<ComboboxSelected>>', lambda e: self.redraw())
        ttk.Label(row2, text='cl(y) at α [deg]').pack(side='left')
        self.alpha_show = tk.StringVar(value='4')
        entry = ttk.Entry(row2, textvariable=self.alpha_show, width=6)
        entry.pack(side='left', padx=(4, 12))
        entry.bind('<Return>', lambda e: self.redraw())
        self.overlay = tk.BooleanVar(value=False)
        ttk.Checkbutton(row2, text='Overlay all methods', variable=self.overlay).pack(side='left')
        return frame

    def settings(self):
        '''(flight dict, method key, overlay) from the controls'''
        v = {key: parse(var.get(), float, 'Aerodynamics -> %s' % key)
             for key, var in self.vars.items()}
        flight = dict(V=v['V'], h=v['h'], alpha=(v['a0'], v['a1'], v['da']))
        return flight, self.method_keys[self.method.get()], self.overlay.get()

    def show(self, aircraft, aero, others):
        self.aircraft, self.aero, self.others = aircraft, aero, others
        names = list(aero['surfaces'])
        self.surface_box['values'] = names
        if self.surface.get() not in names:
            self.surface.set(names[0] if names else '')
        self.set_text(aero['log'])
        self.redraw()

    def redraw(self):
        axes = (self.ax_cl, self.ax_polar, self.ax_ld, self.ax_span)
        for ax in axes:
            ax.clear()
        name = self.surface.get()
        if self.aero is None or name not in self.aero['surfaces']:
            self.summary.set('No mirrored surface is included')
            self.canvas.draw_idle()
            return
        try:
            a_show = parse(self.alpha_show.get(), float, 'cl(y) at α')
        except ValueError as err:
            messagebox.showerror('Aerodynamics', str(err))
            return

        R = self.aero['surfaces'][name]
        runs = [(R, '-', 2.0)] + [(o['surfaces'][name], '--', 1.0) for o in self.others.values()]
        for res, ls, lw in runs:
            label = METHOD_LABELS[res['method']]
            self.ax_cl.plot(res['alpha'], res['CL'], ls=ls, lw=lw, label=label)
            self.ax_polar.plot(res['CD'], res['CL'], ls=ls, lw=lw)
            self.ax_ld.plot(res['alpha'], res['L_D'], ls=ls, lw=lw)
            cl = res['cl0'] + np.deg2rad(a_show) * res['cla']
            self.ax_span.plot(res['y'], cl, ls=ls, lw=lw)

        Cl_max = self.aircraft['surfaces'][name]['airfoil']['Cl_max']
        self.ax_cl.plot(R['alpha_stall'], R['CL_max'], 'o', color='crimson', label='CL_max (est.)')
        self.ax_span.axhline(Cl_max, color='crimson', ls=':', lw=1, label='airfoil Cl_max')
        i = int(np.argmax(R['L_D']))
        self.ax_ld.plot(R['alpha'][i], R['L_D'][i], 'o', color='crimson')

        self.ax_cl.set(xlabel='α [deg]', ylabel='CL', title='Lift curve')
        self.ax_polar.set(xlabel='CD', ylabel='CL', title='Drag polar')
        self.ax_ld.set(xlabel='α [deg]', ylabel='L/D', title='Lift to drag')
        self.ax_span.set(xlabel='y [m]', ylabel='local cl', title='Spanwise cl at α = %g°' % a_show)
        self.ax_cl.legend(fontsize=8)
        self.ax_span.legend(fontsize=8, loc='lower left')
        for ax in axes:
            ax.grid(True, lw=0.4, alpha=0.6)

        self.summary.set(
            '%s, %s:  CLα = %.3f /rad   α_L0 = %.2f°   e = %.3f   CD0 = %.4f   '
            'CL_max ≈ %.2f at %.1f°   (L/D)max = %.1f at %.1f°   |   M = %.3f   Re = %.2e'
            % (name, METHOD_LABELS[R['method']], R['CL_alpha'], R['alpha_L0'], R['e'],
               R['CD0'], R['CL_max'], R['alpha_stall'], R['L_D'][i], R['alpha'][i],
               R['M'], R['Re']))
        self.canvas.draw_idle()


class RotorTab(ResultTab):
    '''Design mode curves of one rotor group, plus its simulator results.'''

    def __init__(self, parent, on_change):
        self.on_change = on_change
        self.model_keys = {label: key for key, label in DESIGN_MODELS.items()}
        self.inflow_keys = {label: key for key, label in INFLOW_MODELS.items()}
        super().__init__(parent, top=self._controls, figsize=(7, 5.5))
        self.ax_T = self.fig.add_subplot(2, 3, 1)
        self.ax_P = self.fig.add_subplot(2, 3, 2)
        self.ax_eff = self.fig.add_subplot(2, 3, 3)
        self.ax_alpha = self.fig.add_subplot(2, 3, 4)
        self.ax_disk = self.fig.add_subplot(2, 3, 5, projection='polar')
        self.ax_beta = self.fig.add_subplot(2, 3, 6)
        self.cbar = None
        self.result, self.aircraft = None, None

    def _entries(self, row, items, width=6):
        for key, label, value in items:
            ttk.Label(row, text=label).pack(side='left')
            self.vars[key] = tk.StringVar(value='' if value is None else fmt(value))
            ttk.Entry(row, textvariable=self.vars[key], width=width).pack(side='left', padx=(4, 10))

    def _controls(self, parent):
        frame = ttk.Frame(parent)
        s, sim = Main.ROTOR, Main.ROTOR['sim']
        names = list(Main.AIRCRAFT.get('rotors', {}))
        self.vars = {}

        row1 = ttk.Frame(frame)
        row1.pack(fill='x')
        ttk.Label(row1, text='Rotor').pack(side='left')
        self.rotor = tk.StringVar(value=s['simulate'] if s['simulate'] in names
                                  else (names[0] if names else ''))
        box = ttk.Combobox(row1, textvariable=self.rotor, state='readonly', width=8, values=names)
        box.pack(side='left', padx=(4, 12))
        box.bind('<<ComboboxSelected>>', lambda e: self.on_change())
        ttk.Label(row1, text='Model').pack(side='left')
        self.model = tk.StringVar(value=DESIGN_MODELS[s['model']])
        ttk.Combobox(row1, textvariable=self.model, state='readonly', width=27,
                     values=list(DESIGN_MODELS.values())).pack(side='left', padx=(4, 12))
        self.trim = tk.BooleanVar(value=s['trim'])
        ttk.Checkbutton(row1, text='Trim lift rotors to weight',
                        variable=self.trim).pack(side='left', padx=(0, 12))
        self._entries(row1, (('climb', 'Climb [m/s]', s['climb']),))

        row2 = ttk.Frame(frame)
        row2.pack(fill='x', pady=(4, 0))
        self._entries(row2, (('t0', 'Collective sweep [deg]  from', s['theta_sweep'][0]),
                             ('t1', 'to', s['theta_sweep'][1]),
                             ('dt', 'step', s['theta_sweep'][2])))

        row3 = ttk.Frame(frame)
        row3.pack(fill='x', pady=(4, 0))
        self.simulate = tk.BooleanVar(value=s['simulate'] is not None)
        ttk.Checkbutton(row3, text='Simulator (edgewise flight)',
                        variable=self.simulate).pack(side='left', padx=(0, 12))
        self._entries(row3, (('V', 'V [m/s]', sim['V']),
                             ('alpha_s', 'Shaft α [deg]', sim['alpha_s']),
                             ('theta0', 'θ0 (blank = design)', sim['theta0']),
                             ('theta1s', 'θ1s', sim['theta1s']),
                             ('theta1c', 'θ1c', sim['theta1c'])))
        row4 = ttk.Frame(frame)
        row4.pack(fill='x', pady=(4, 0))
        ttk.Label(row4, text='', width=29).pack(side='left')   # line up under the simulator
        self._entries(row4, (('gamma', 'Lock number γ', sim['gamma']),
                             ('nu_beta', 'Flap frequency ν_β', sim['nu_beta'])))
        ttk.Label(row4, text='Inflow').pack(side='left')
        self.inflow = tk.StringVar(value=INFLOW_MODELS[sim['inflow']])
        ttk.Combobox(row4, textvariable=self.inflow, state='readonly', width=16,
                     values=list(INFLOW_MODELS.values())).pack(side='left', padx=(4, 0))
        return frame

    def settings(self):
        '''Main.ROTOR-like dict from the controls'''
        def num(key):
            return parse(self.vars[key].get(), float, 'Rotors -> %s' % key)
        theta0 = self.vars['theta0'].get().strip()
        return dict(model=self.model_keys[self.model.get()], n=Main.ROTOR['n'],
                    trim=self.trim.get(), climb=num('climb'),
                    theta_sweep=(num('t0'), num('t1'), num('dt')),
                    simulate=self.rotor.get() if self.simulate.get() else None,
                    sim=dict(V=num('V'), alpha_s=num('alpha_s'),
                             theta0=None if theta0 == '' else num('theta0'),
                             theta1s=num('theta1s'), theta1c=num('theta1c'),
                             inflow=self.inflow_keys[self.inflow.get()],
                             gamma=num('gamma'), nu_beta=num('nu_beta')))

    def show(self, aircraft, result):
        self.aircraft, self.result = aircraft, result
        self.set_text(result['log'])
        self.redraw()

    def redraw(self):
        if self.cbar is not None:                 # before clearing its axes
            self.cbar.remove()
            self.cbar = None
        axes = (self.ax_T, self.ax_P, self.ax_eff, self.ax_alpha, self.ax_disk, self.ax_beta)
        for ax in axes:
            ax.clear()
        name = self.rotor.get()
        res = self.result['rotors'].get(name) if self.result else None
        if res is None:
            self.summary.set('Rotor "%s" is not included' % name)
            self.canvas.draw_idle()
            return

        op, sw = res['op'], res['sweep']
        lift = res['axis'] == 'up'
        mark = dict(marker='o', color='crimson', ls='')
        self.ax_T.plot(sw['theta'], sw['T'])
        self.ax_T.plot(op['theta'], op['T'], **mark)
        if res['T_req'] is not None:
            self.ax_T.axhline(res['T_req'], color='crimson', ls=':', lw=1)
        self.ax_P.plot(sw['theta'], sw['P'] / 1000.0)
        self.ax_P.plot(op['theta'], op['P'] / 1000.0, **mark)
        key, label = ('FM', 'Figure of merit') if lift else ('eta', 'Efficiency η')
        self.ax_eff.plot(sw['theta'], sw[key])
        self.ax_eff.plot(op['theta'], op[key], **mark)
        foil = self.aircraft['rotors'][name]['airfoil']
        self.ax_alpha.plot(op['r'], op['alpha_r'])
        self.ax_alpha.axhline(foil['alpha_L0'] + np.degrees(foil['Cl_max'] / foil['a0']),
                              color='crimson', ls=':', lw=1, label='stall')
        self.ax_alpha.legend(fontsize=8)
        self.ax_T.set(xlabel='θ0 [deg]', ylabel='T per rotor [N]', title='Thrust')
        self.ax_P.set(xlabel='θ0 [deg]', ylabel='P per rotor [kW]', title='Power')
        self.ax_eff.set(xlabel='θ0 [deg]', ylabel=label, title=label)
        self.ax_alpha.set(xlabel='r [m]', ylabel='α [deg]', title='Blade α at operating point')
        for ax in (self.ax_T, self.ax_P, self.ax_eff, self.ax_alpha, self.ax_beta):
            ax.grid(True, lw=0.4, alpha=0.6)

        text = ('%s: %d x %s, %s:   θ0 = %.2f° (θ.75 = %.2f°)   T = %.2f N each (%.1f N total)   '
                'P = %.0f W each (%.2f kW total)   %s   CT/σ = %.3f   M_tip = %.2f'
                % (name, res['count'], 'lift rotor' if lift else 'propeller',
                   DESIGN_MODELS[op['method']], op['theta'], op['theta_75'], op['T'],
                   res['T_total'], op['P'], res['P_total'] / 1000.0,
                   'FM = %.3f' % op['FM'] if lift else 'η = %.3f at J = %.3f' % (op['eta'], op['J']),
                   op['CT_sigma'], op['M_tip']))

        sim = self.result['sim']
        if sim is not None and sim.get('rotor') == name:
            self.ax_disk.set_axis_on()
            self.ax_beta.set_axis_on()
            psi = np.radians(np.append(sim['psi'], 360.0))
            amap = np.vstack([sim['alpha_map'], sim['alpha_map'][:1]])
            TH, RR = np.meshgrid(psi, sim['r'])
            lo, hi = np.percentile(amap, [1, 99])
            mesh = self.ax_disk.pcolormesh(TH, RR, amap.T, shading='gouraud', cmap='viridis',
                                           vmin=lo, vmax=hi)
            self.ax_disk.set_theta_zero_location('S')      # psi = 0 aft, nose up
            self.ax_disk.set_theta_direction(1)            # counter-clockwise
            self.ax_disk.set_yticklabels([])
            self.ax_disk.set_xticks(np.radians([0, 90, 180, 270]),
                                    ['ψ=0 aft', '90', 'nose', '270'], fontsize=7)
            self.ax_disk.set_title('Disk α [deg], advancing side right', fontsize=9)
            self.cbar = self.fig.colorbar(mesh, ax=self.ax_disk, shrink=0.9, pad=0.12)
            self.ax_beta.plot(sim['psi'], sim['beta'])
            self.ax_beta.set(xlabel='ψ [deg]', ylabel='β [deg]', title='Blade flapping',
                             xticks=[0, 90, 180, 270, 360])
            text += ('\nSimulator  V = %.1f m/s, μ = %.3f:   T = %.1f N   H = %.2f N   Y = %.2f N   '
                     'P = %.0f W   β0 = %.2f°   β1c = %.2f°   β1s = %.2f°   roll = %.2f N m   pitch = %.2f N m'
                     % (sim['V'], sim['mu'], sim['T'], sim['H'], sim['Y'], sim['P'], sim['beta0'],
                        sim['beta1c'], sim['beta1s'], sim['roll'], sim['pitch']))
        else:
            for ax in (self.ax_disk, self.ax_beta):
                ax.set_axis_off()
                ax.text(0.5, 0.5, 'Simulator off', transform=ax.transAxes,
                        ha='center', va='center', color='gray')
        self.summary.set(text)
        self.canvas.draw_idle()


# (tab title, module, what will go there)
PLACEHOLDERS = [
    ('AeroElasticity', 'AeroElasticity.py',      'Aeroelastic analysis.'),
    ('Mission',        'MissionPlanner.py',      'Mission planning.'),
    ('Materials',      'Material_Properties.py', 'Material properties (densities could feed the structure inputs).'),
]


def placeholder_tab(parent, title, module, note):
    frame = ttk.Frame(parent, padding=24)
    ttk.Label(frame, text=title, font=('Segoe UI', 16, 'bold')).pack(anchor='w')
    ttk.Label(frame, text=note).pack(anchor='w', pady=(6, 0))
    ttk.Label(frame, text='Not implemented yet. Connect it to %s through Main.py here.' % module,
              foreground='gray').pack(anchor='w', pady=(12, 0))
    return frame


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('BlackBird Design Code')
        self.geometry('1500x900')
        self.minsize(1100, 700)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left = ttk.Frame(self, padding=(6, 6, 0, 6))
        left.grid(row=0, column=0, sticky='ns')
        left.rowconfigure(0, weight=1)
        self.panel = AircraftPanel(left)
        self.panel.grid(row=0, column=0, sticky='ns')
        buttons = ttk.Frame(left)
        buttons.grid(row=1, column=0, sticky='ew', pady=(6, 0))
        ttk.Button(buttons, text='Calculate', command=self.calculate).pack(side='left')
        ttk.Button(buttons, text='Reset defaults', command=self.reset).pack(side='left', padx=6)

        notebook = ttk.Notebook(self)
        notebook.grid(row=0, column=1, sticky='nsew', padx=6, pady=6)
        self.weight_tab = WeightTab(notebook)
        self.aero_tab = AeroTab(notebook)
        self.rotor_tab = RotorTab(notebook, on_change=self.calculate)
        notebook.add(self.weight_tab, text='Weight & CG')
        notebook.add(self.aero_tab, text='Aerodynamics')
        notebook.add(self.rotor_tab, text='Rotors')
        for title, module, note in PLACEHOLDERS:
            notebook.add(placeholder_tab(notebook, title, module, note), text=title)

        self.calculate()

    def calculate(self):
        '''Read every input, run the analyses through Main, show the results.'''
        try:
            aircraft = self.panel.read()
            flight, method, overlay = self.aero_tab.settings()
            rotor_settings = self.rotor_tab.settings()
            weight = Main.run_weight(aircraft)
            aero = Main.run_aero(aircraft, flight, method=method)
            others = {m: Main.run_aero(aircraft, flight, method=m)
                      for m in METHOD_LABELS if overlay and m != method}
            rotor = Main.run_rotor(aircraft, flight, rotor_settings, weight=weight)
        except (ValueError, NotImplementedError, RuntimeError) as err:
            messagebox.showerror('Cannot calculate', str(err))
            return
        self.weight_tab.show(aircraft, weight)
        self.aero_tab.show(aircraft, aero, others)
        self.rotor_tab.show(aircraft, rotor)

    def reset(self):
        self.panel.reset()
        self.calculate()


def main():
    try:
        # sharp text on high-DPI Windows screens
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    App().mainloop()


if __name__ == '__main__':
    main()
