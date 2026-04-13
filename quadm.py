import pyvista as pv
import numpy as np
import time
import sys

# =========================================================================
# [CORE PHYSICS MODULES - UPGRADED TO FULL META-BEM] 
# =========================================================================
class FluidEnvironment:
    def __init__(self, fluid_type="air"):
        self.fluid_type = fluid_type
        if fluid_type == "water": self.rho = 1000.0
        elif fluid_type == "air": self.rho = 1.225

class MagnusAeroPolars:
    def __init__(self):
        self.sr_base = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0])
        self.cl_base = np.array([0.0, 1.1, 2.6, 4.1, 5.2, 6.0, 6.6, 7.5, 8.2, 8.8])
        self.cd_base = np.array([1.1, 0.9, 0.8, 1.1, 1.5, 2.0, 2.6, 3.8, 5.2, 6.8])
        self.ct_bem = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5])
        self.a_bem  = np.array([0.0, 0.05, 0.11, 0.18, 0.27, 0.38, 0.50, 0.65, 0.82, 1.0, 1.0, 1.0])

class Kinematics6DoF:
    def __init__(self, mass, I_pitch, I_yaw, I_roll, initial_pos=None):
        self.mass = mass
        self.I = np.array([I_pitch, I_yaw, I_roll], dtype=float)
        self.pos = np.array(initial_pos if initial_pos is not None else [0.0, 0.0, 0.0], dtype=float)
        self.vel = np.array([0.0, 0.0, 0.0], dtype=float)
        self.angles = np.array([0.0, 0.0, 0.0], dtype=float)
        self.omega = np.array([0.0, 0.0, 0.0], dtype=float)
        
    def step(self, dt, net_force, net_torque, Rx, limit_y_min=-400.0):
        accel = np.clip(np.nan_to_num(net_force / self.mass), -100.0, 100.0)
        self.vel += accel * dt
        self.pos += self.vel * dt
        
        local_torque = np.linalg.inv(Rx).dot(net_torque)
        local_alpha = np.clip(np.nan_to_num(local_torque / self.I), -100.0, 100.0)
        alpha_global = Rx.dot(local_alpha)
        
        self.omega += alpha_global * dt
        self.angles += self.omega * dt
        
        self.vel *= 0.999
        self.omega *= 0.99

        if self.pos[1] <= limit_y_min: 
            self.pos[1] = limit_y_min
            self.vel[1] = max(self.vel[1], 0.0)
            self.vel[0] *= 0.85 
            self.vel[2] *= 0.85 
            self.angles[0] += (-np.pi / 2.0 - self.angles[0]) * 0.1 
            self.angles[1] *= 0.90 
            self.angles[2] *= 0.90 
            self.omega *= 0.50     

# --- 0. CONFIG ---
pv.global_theme.allow_empty_mesh = True
pv.global_theme.font.color = 'black'

# --- 1. MATH ENGINE ---
def get_align_matrix(p0, p1, scale_x=1.0, scale_y=1.0, scale_z=1.0):
    scale_x = max(abs(scale_x), 0.001)
    scale_y = max(abs(scale_y), 0.001)
    scale_z = max(abs(scale_z), 0.001)
    
    p0 = np.array(p0, dtype=float); p1 = np.array(p1, dtype=float)
    v = p1 - p0; mag = np.linalg.norm(v)
    if mag < 1e-6 or np.isnan(mag): mag = 0.001; v = np.array([0,0,1], dtype=float)
    else: v = v / mag
    up = np.array([0, 0, 1], dtype=float)
    if np.abs(np.dot(v, up)) > 0.99: up = np.array([0, 1, 0], dtype=float)
    vec_x = np.cross(v, up); vec_x /= np.linalg.norm(vec_x)
    vec_y = np.cross(v, vec_x)
    m = np.eye(4)
    m[0:3, 0] = vec_x * scale_x; m[0:3, 1] = vec_y * scale_y; m[0:3, 2] = v * (mag * scale_z)
    m[0:3, 3] = (p0 + p1) / 2.0
    return m

def math_pts_cyl(p0, p1, r0, r1, res=24, rot=0.0):
    p0 = np.array(p0, dtype=float); p1 = np.array(p1, dtype=float)
    v = p1 - p0; mag = np.linalg.norm(v)
    if mag < 1e-6: mag = 0.001; v = np.array([0,0,1], dtype=float)
    else: v = v / mag
    not_v = np.array([0, 0, 1], dtype=float)
    if np.abs(np.dot(v, not_v)) > 0.99: not_v = np.array([0, 1, 0], dtype=float)
    n1 = np.cross(v, not_v); n1 /= np.linalg.norm(n1); n2 = np.cross(v, n1)
    num_h = 2; h = np.linspace(0, mag, num_h); u = np.linspace(0, 2*np.pi, res) + rot
    H, U = np.meshgrid(h, u, indexing='ij')
    Radii = np.linspace(r0, r1, num_h)[:, np.newaxis]
    H_ = H[..., np.newaxis]; U_ = U[..., np.newaxis]; R_ = Radii[..., np.newaxis]
    Points = (p0 + v * H_ + R_ * np.cos(U_) * n1 + R_ * np.sin(U_) * n2)
    return Points.reshape(-1, 3), [res, num_h, 1]

def math_pts_bellows(p0, p1, r0, r1, fold_factor, res=24, rot_phase=0.0):
    p0 = np.array(p0, dtype=float); p1 = np.array(p1, dtype=float)
    v = p1 - p0; mag = np.linalg.norm(v)
    if mag < 1e-6: mag = 0.001; v = np.array([0,0,1], dtype=float)
    else: v = v / mag
    not_v = np.array([0, 0, 1], dtype=float)
    if np.abs(np.dot(v, not_v)) > 0.99: not_v = np.array([0, 1, 0], dtype=float)
    n1 = np.cross(v, not_v); n1 /= np.linalg.norm(n1); n2 = np.cross(v, n1)
    num_pleats = 12; num_h = num_pleats * 2 + 1
    h = np.linspace(0, mag, num_h); u = np.linspace(0, 2*np.pi, res) + rot_phase 
    pleat_amp = r0 * 0.6 * fold_factor 
    r_vals = []
    for i in range(num_h):
        base_r = r0 + (r1 - r0) * (i / (num_h - 1))
        if i % 2 == 0: r_vals.append(base_r + pleat_amp)      
        else:          r_vals.append(base_r - pleat_amp*0.8) 
    H, U = np.meshgrid(h, u, indexing='ij')
    Radii = np.array(r_vals)[:, np.newaxis]
    H_ = H[..., np.newaxis]; U_ = U[..., np.newaxis]; R_ = Radii[..., np.newaxis]
    Points = (p0 + v * H_ + R_ * np.cos(U_) * n1 + R_ * np.sin(U_) * n2)
    return Points.reshape(-1, 3), [res, num_h, 1]

def math_pts_strip(p0, p1, r0, r1, angle_c, rot_phase):
    p0 = np.array(p0, dtype=float); p1 = np.array(p1, dtype=float); v = p1 - p0
    mag = np.linalg.norm(v)
    if mag < 1e-6: mag = 0.001; v = np.array([0,0,1], dtype=float)
    else: v = v / mag
    not_v = np.array([0, 0, 1], dtype=float)
    if np.abs(np.dot(v, not_v)) > 0.99: not_v = np.array([0, 1, 0], dtype=float)
    n1 = np.cross(v, not_v); n1 /= np.linalg.norm(n1); n2 = np.cross(v, n1)
    strip_width = np.radians(15.0); res_w = 12 
    u_strip = np.linspace(angle_c - strip_width, angle_c + strip_width, res_w) + rot_phase
    h_grid = np.linspace(0, mag, 2)
    H, U = np.meshgrid(h_grid, u_strip, indexing='ij')
    base_r = r0 + (r1 - r0) * (H / mag); Radii = base_r * 1.05 
    H_ = H[..., np.newaxis]; U_ = U[..., np.newaxis]; R_ = Radii[..., np.newaxis]
    Points = (p0 + v * H_ + R_ * np.cos(U_) * n1 + R_ * np.sin(U_) * n2)
    return Points.reshape(-1, 3), [res_w, 2, 1]

# --- 2. FACTORY FUNCTIONS ---
def create_solid_template(radius):
    return pv.Cylinder(center=(0,0,0), direction=(0,0,1), radius=radius, height=1.0, resolution=24, capping=True)
def create_arrow_template(scale=1.0):
    return pv.Arrow(start=(0,0,-0.5), direction=(0,0,1), tip_length=0.25, tip_radius=0.1, shaft_radius=0.05, scale=scale)
def create_grid_mesh(p0, p1, r0, r1, res=24):
    pts, dims = math_pts_cyl(p0, p1, r0, r1, res, 0.0)
    grid = pv.StructuredGrid(); grid.points = pts; grid.dimensions = dims
    return grid
def create_bellows_mesh_init(p0, p1, r0, r1, res=24):
    pts, dims = math_pts_bellows(p0, p1, r0, r1, 0.0, res, 0.0)
    grid = pv.StructuredGrid(); grid.points = pts; grid.dimensions = dims
    return grid
def create_strip_mesh_init(p0, p1, r0, r1, angle):
    pts, dims = math_pts_strip(p0, p1, r0, r1, angle, 0.0)
    grid = pv.StructuredGrid(); grid.points = pts; grid.dimensions = dims
    return grid

# --- 3. SCENE PART WRAPPER ---
class ScenePart:
    def __init__(self, plotter, mesh, color, opacity=1.0, wireframe=False):
        self.actor = plotter.add_mesh(mesh, color=color, opacity=opacity, style='wireframe' if wireframe else 'surface', 
                                      smooth_shading=True, specular=0.5, name=None)
        self.mesh = self.actor.mapper.dataset
        self.base_color = color 
    def set_matrix(self, matrix): self.actor.user_matrix = matrix
    def update_transform(self, p0, p1, scale_z=1.0):
        if np.any(np.isnan(p0)) or np.any(np.isnan(p1)) or np.isnan(scale_z): return
        m = get_align_matrix(p0, p1, scale_z=scale_z)
        self.actor.user_matrix = m
    def update_mesh(self, new_mesh): self.actor.mapper.dataset.DeepCopy(new_mesh)
    def set_visibility(self, visible): self.actor.visibility = visible
    def set_color(self, color): self.actor.prop.color = color

# --- 4. MAIN APP ---
class QuadMagnusApp:
    def __init__(self):
        pv.global_theme.allow_empty_mesh = True
        self.p = pv.Plotter(title="Quad-Magnus: FULL FLIGHT SIMULATOR (META-BEM INTEGRATION)", window_size=(1600, 1000))
        self.p.set_background('white')
        self.prev_net_power = 0.0
        self.optimization_timer = 0.0
        self.best_pitch_found = 22.0 
        self.main_line_reeled_in = 0.0
        
        def _on_close(*args):
            try: self.p.iren.TerminateApp()
            except: pass
            sys.exit(0)
            
        if hasattr(self.p, 'iren') and self.p.iren is not None:
            self.p.iren.add_observer("ExitEvent", _on_close)
            self.p.iren.add_observer("WindowCloseEvent", _on_close)
            
        self.spinning = False
        self.was_spinning = False 
        self.folding = False
        self.structural_failure = False 
        self.auto_pilot = False 
        
        self.flight_phase = "HARVEST" 
        self.governor_status = "IDLE"
        self.sim_time = 0.0
        self.para_inflation = 0.0 

        self.env = FluidEnvironment("air")
        self.polars = MagnusAeroPolars()
        
        total_mass = 1700.0 
        I_pitch = 10000.0
        I_yaw = 10000.0
        I_roll = 10000.0
        self.kin = Kinematics6DoF(total_mass, I_pitch, I_yaw, I_roll, initial_pos=[0.0, 0.0, 0.0])
        
        self.lut_load_pct = np.array([0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0, 1.2])
        self.lut_eff_gen  = np.array([0.0, 0.65, 0.85, 0.92, 0.95, 0.96, 0.94, 0.90]) 
        self.lut_eff_mot  = np.array([0.0, 0.60, 0.82, 0.90, 0.94, 0.95, 0.93, 0.88]) 
        
        self.ground_level = -400.0
        
        self.val_wind = 10.0         
        self.val_pitch = 0.0         
        self.val_spin_drive = 3500.0 
        self.val_spin_blue = 180.0   
        self.val_gen_load = 18.0     
        self.val_target_payload = 1500.0 
        
        self.rotor_rpm_top = 0.0
        self.rotor_rpm_bot = 0.0
        self.current_total_drag = 0.0 
        self.drag_limit = 250000.0 
        
        self.telemetry = {
            'gen_top_kw': 0.0, 'gen_bot_kw': 0.0,
            'motor_cone_top_kw': 0.0, 'motor_cone_bot_kw': 0.0, 
            'motor_tube_top_kw': 0.0, 'motor_tube_bot_kw': 0.0,
            'net_power_kw': 0.0,
            'lift_total_kg': 0.0, 'betz_limit_kw': 0.0,
            'tube_rpm_top': 0.0, 'tube_rpm_bot': 0.0,
            'cone_rpm_top': 0.0, 'cone_rpm_bot': 0.0,
            'tether_state': 'PARKED (SETUP)',
            'diag_f_net_y': 0.0, 'diag_drag_total': 0.0,
            'diag_drag_tubes': 0.0, 'diag_drag_cones': 0.0, 'diag_drag_frame': 0.0,
            'diag_thrust_vtol': 0.0, 'diag_parachute_drag': 0.0
        }
        
        self.show_tube_air = False   
        self.show_tube_force = False 
        self.show_cone_air_res = False  
        self.show_cone_air_comp = False 
        self.show_cone_force_res = False 
        self.show_cone_force_comp = False 
        
        self.fold_direction = 1 
        self.fold_factor = 0.0
        
        self.rotor_angle_top = 0.0
        self.rotor_angle_bot = 0.0
        self.spoke_spin_phase_top = 0.0
        self.spoke_spin_phase_bot = 0.0
        self.tube_spin_phase_top = 0.0
        self.tube_spin_phase_bot = 0.0
        
        self.Max_W = 9.5; self.Max_H = 9.0; self.Beam_Len = np.sqrt(self.Max_W**2 + self.Max_H**2)
        self.Min_W = 1.2; self.Z_F = 0.25; self.Z_B = -0.25
        self.Max_Tube_L = 19.0; self.Min_Tube_L = 4.0
        
        self.frame_parts = {}; self.strut_parts = []; self.ropes = {}; self.blue_tubes = []
        self.dynamic_spokes = []; self.rotors_matrix_parts = []
        self.tube_air_parts = []; self.tube_force_parts = []; self.cone_vector_parts = {} 
        self.winch_part = None
        self.lbl_actors = [] 
        
        self.para_canopy = None
        self.para_ropes = []
        self.sea_part = None
        self.pod_part = None
        self.buoy_part = None
        
        self.sl_pitch = None; self.sl_spin_lift = None; self.sl_spin_drive = None
        self.sl_gen_load = None; self.sl_wind = None
        
        self.setup_ui()     
        self.setup_hud()    
        self.build_scene()  
        
        self.p.camera.position = (250, -200, -500)
        self.p.camera.focal_point = (0, -200, -200)
        self.p.camera.up = (0, 1, 0)
        self.p.camera.zoom(1.1)

    def compute_strip_aero_forces(self, v_inf, a, a_prime, r_local_flat, cyl_omega, rotor_omega, B_blades, Hub_R, Cyl_L, Cyl_r, safe_v_app_perp, v_app_perp_dir, cyl_vec_dir, area_proj_strip, dr_strip, is_rotor=True):
        if not self.spinning or abs(cyl_omega) < 0.5:
            CD_static = 1.15
            q = 0.5 * self.env.rho * (safe_v_app_perp**2) * area_proj_strip
            dF_total = v_app_perp_dir * (q * CD_static)
            return dF_total, 0.0, 0.0, 0.0

        CL = 0.0; CD_total = 0.6; spin_ratio = 0.01
        a_new = a; a_prime_new = a_prime
        
        effective_v = safe_v_app_perp * (1.0 - a) if is_rotor else safe_v_app_perp
        
        if self.spinning:
            v_ax = max(effective_v, 0.01) 
            v_tg = abs(rotor_omega * r_local_flat * (1 + a_prime)) if is_rotor else 0.01
            phi = np.arctan2(v_ax, max(v_tg, 0.01))
            v_app_bem = np.sqrt(v_ax**2 + v_tg**2) if is_rotor else effective_v
            
            F_prandtl = 1.0
            if is_rotor and np.sin(phi) > 0.01:
                f_tip = (B_blades / 2.0) * (Hub_R + Cyl_L - r_local_flat) / (r_local_flat * np.sin(phi))
                F_prandtl = np.clip(np.nan_to_num((2.0 / np.pi) * np.arccos(np.clip(np.exp(-f_tip), -1.0, 1.0))), 0.001, 1.0)
            
            spin_ratio = abs(cyl_omega * Cyl_r) / max(v_app_bem, 0.01)
            
            mean_CL = np.interp(spin_ratio, self.polars.sr_base, self.polars.cl_base)
            mean_CD = np.interp(spin_ratio, self.polars.sr_base, self.polars.cd_base)

            wake_damping = max(0.0, 1.0 - (spin_ratio / 2.0))
            f_vortex = (0.20 * safe_v_app_perp) / (2.0 * Cyl_r)
            CL = mean_CL + (0.15 * mean_CL * wake_damping) * np.sin(2 * np.pi * f_vortex * self.sim_time)
            CD_profile = mean_CD + (0.10 * mean_CD * wake_damping) * np.sin(2 * np.pi * (2.0 * f_vortex) * self.sim_time)
            
            AR_effective = (Cyl_L / (2.0 * Cyl_r)) * (1.0 + 1.9 * ((Cyl_r * 2.5 - Cyl_r) / Cyl_r))
            CD_induced = (CL**2) / (np.pi * AR_effective * 0.45)
            CD_total = CD_profile + CD_induced
            
            if is_rotor:
                sigma = (B_blades * 2 * Cyl_r) / (2 * np.pi * max(r_local_flat, 0.1))
                Cx_mean = mean_CL * np.cos(phi) + (mean_CD + CD_induced) * np.sin(phi)
                Cy_mean = mean_CL * np.sin(phi) - (mean_CD + CD_induced) * np.cos(phi)
                
                CT_modified = ((sigma * Cx_mean * (1 - a)**2) / max(np.sin(phi)**2, 1e-4)) / max(F_prandtl, 0.001)
                a_new_val = np.interp(CT_modified, self.polars.ct_bem, self.polars.a_bem)
                a_prime_new_val = 1.0 / ((4 * F_prandtl * np.sin(phi) * np.cos(phi)) / (sigma * max(Cy_mean, 1e-4)) - 1)
                a_new = 0.5 * a + 0.5 * np.clip(np.nan_to_num(a_new_val), 0.0, 0.85)
                a_prime_new = 0.5 * a_prime + 0.5 * np.clip(np.nan_to_num(a_prime_new_val), -0.5, 0.5)

        spin_axis_local = cyl_vec_dir * np.sign(cyl_omega) if cyl_omega != 0 else cyl_vec_dir
        mag_lift_dir = np.cross(v_app_perp_dir, spin_axis_local)
        if np.linalg.norm(mag_lift_dir) > 0.001: mag_lift_dir /= np.linalg.norm(mag_lift_dir)
        
        q = 0.5 * self.env.rho * (effective_v**2) * area_proj_strip
        dF_total = mag_lift_dir * (q * CL) + v_app_perp_dir * (q * CD_total)

        v_surface_out = abs(cyl_omega * Cyl_r)
        dP_skin_friction = self.env.rho * 0.005 * (2 * np.pi * Cyl_r * dr_strip) * (v_surface_out**3)
        C_M = 0.012 * spin_ratio + 0.002 * (spin_ratio**2)
        dP_skin_friction += C_M * q * Cyl_r * abs(cyl_omega)

        return dF_total, dP_skin_friction, a_new, a_prime_new

    def draw_button_labels(self):
        base_y = 60
        self.lbl_actors = [
            self.p.add_text("START", position=(50, base_y+5), color='black', font_size=10),
            self.p.add_text("FOLD", position=(50, base_y+45), color='black', font_size=10),
            self.p.add_text("AUTO PILOT", position=(200, base_y+5), color='black', font_size=10),
            self.p.add_text("RESET CRASH", position=(200, base_y+45), color='black', font_size=10),
            self.p.add_text("TUBE AIR", position=(390, base_y+5), color='black', font_size=10),
            self.p.add_text("TUBE LIFT", position=(390, base_y+45), color='black', font_size=10),
            self.p.add_text("AIR (RES)", position=(570, base_y+5), color='black', font_size=10),
            self.p.add_text("FORCE (RES)", position=(570, base_y+45), color='black', font_size=10),
            self.p.add_text("AIR (COMP)", position=(750, base_y+5), color='black', font_size=10),
            self.p.add_text("FORCE (ALL)", position=(750, base_y+45), color='black', font_size=10)
        ]

    def set_labels_color(self, color_str):
        c = pv.Color(color_str).float_rgb
        for actor in self.lbl_actors:
            actor.GetTextProperty().SetColor(c)

    def setup_ui(self):
        self.sl_wind = self.p.add_slider_widget(self.set_wind, [0, 25], title="Wind Speed (m/s)", value=10.0, pointa=(0.03, 0.95), pointb=(0.20, 0.95), style='modern')
        self.sl_pitch = self.p.add_slider_widget(self.set_pitch, [-90, 90], title="Target Winch Pitch (deg)", value=0.0, pointa=(0.03, 0.82), pointb=(0.20, 0.82), style='modern')
        self.p.add_slider_widget(self.set_target_payload, [0, 15000], title="Target Payload (kg)", value=1500.0, pointa=(0.03, 0.69), pointb=(0.20, 0.69), style='modern')
        self.sl_spin_lift = self.p.add_slider_widget(self.set_spin_lift, [0, 300], title="Tube Base RPM", value=180.0, pointa=(0.03, 0.56), pointb=(0.20, 0.56), style='modern')
        self.sl_spin_drive = self.p.add_slider_widget(self.set_spin_drive, [0, 5000], title="Cone RPM Limit", value=3500.0, pointa=(0.03, 0.43), pointb=(0.20, 0.43), style='modern')
        self.sl_gen_load = self.p.add_slider_widget(self.set_gen_load, [-100, 100], title="Gen Load (%)", value=18.0, pointa=(0.03, 0.30), pointb=(0.20, 0.30), style='modern')
        
        base_y = 60
        self.btn_spin = self.p.add_checkbox_button_widget(self.toggle_spin, value=False, position=(10, base_y), size=30, color_on='green', color_off='grey')
        self.btn_fold = self.p.add_checkbox_button_widget(self.toggle_fold, value=False, position=(10, base_y+40), size=30, color_on='orange', color_off='grey')
        self.btn_ap = self.p.add_checkbox_button_widget(self.toggle_auto_pilot, value=False, position=(160, base_y), size=30, color_on='cyan', color_off='grey')
        self.btn_reset = self.p.add_checkbox_button_widget(self.trigger_reset, value=False, position=(160, base_y+40), size=30, color_on='red', color_off='lightgray')
        
        self.btn_tube_air = self.p.add_checkbox_button_widget(self.toggle_tube_air, value=False, position=(350, base_y), size=30, color_on='cyan', color_off='grey')
        self.btn_tube_force = self.p.add_checkbox_button_widget(self.toggle_tube_force, value=False, position=(350, base_y+40), size=30, color_on='red', color_off='grey')
        self.btn_cone_air_res = self.p.add_checkbox_button_widget(self.toggle_cone_air_res, value=False, position=(530, base_y), size=30, color_on='orange', color_off='grey')
        self.btn_cone_force_res = self.p.add_checkbox_button_widget(self.toggle_cone_force_res, value=False, position=(530, base_y+40), size=30, color_on='purple', color_off='grey')
        self.btn_cone_air_comp = self.p.add_checkbox_button_widget(self.toggle_cone_air_comp, value=False, position=(710, base_y), size=30, color_on='yellow', color_off='grey')
        self.btn_cone_force_comp = self.p.add_checkbox_button_widget(self.toggle_cone_force_comp, value=False, position=(710, base_y+40), size=30, color_on='blue', color_off='grey')
        
        self.draw_button_labels()

    def setup_hud(self):
        self.update_hud()

    def update_hud(self):
        t = self.telemetry
        drag_curr = self.current_total_drag
        stress_pct = min(100, max(0, (drag_curr / self.drag_limit) * 100))
        bars = int(stress_pct / 5)
        stress_bar_visual = "[" + "#" * bars + "-" * (20 - bars) + "]"
        
        ap_status = "OFF (MANUAL)"
        ap_color = "black"
        status_msg = "SYSTEM NORMAL"
        status_col = "black"
        
        if self.auto_pilot:
            ap_status = f"ON ({self.governor_status})"
            ap_color = "green"
            if "DEFENSE" in self.governor_status:
                status_msg = "ACTIVE DEPOWER LIMITER"
                status_col = "orange"
            elif "GLIDE" in self.governor_status:
                status_msg = "AERODYNAMIC BRAKE (GLIDING)"
                status_col = "purple"
            elif "PARACHUTE" in self.governor_status:
                status_msg = "EMERGENCY FLARE (AIRBRAKE DEPLOYED)"
                status_col = "purple"
            elif "VTOL" in self.governor_status:
                status_msg = "DRONE MODE (VTOL)"
                status_col = "teal"
            elif "LANDED" in self.governor_status:
                status_msg = "SAFE ON OCEAN (IDLE)"
                status_col = "blue"
            elif "TAKEOFF" in self.governor_status:
                status_msg = "LAUNCH SEQUENCE (ASCENDING)"
                status_col = "blue"
            elif "PASSIVE" in self.governor_status:
                status_msg = "PASSIVE ASCENT (KITE MODE)"
                status_col = "teal"
            elif "TRANSITION" in self.governor_status:
                status_msg = "PREPARING HARVEST"
                status_col = "teal"
                
        elif stress_pct > 80: 
            status_msg = "WARNING: HIGH STRESS"
            status_col = "orange"
            
        if self.structural_failure: 
            status_msg = "*** CRITICAL FAILURE ***"
            status_col = "red"
            
        tether_state = t.get('tether_state', 'PARKED (SETUP)')
        actual_pitch_deg = np.degrees(-self.kin.angles[0])
        
        text_block_1 = (
            f"QUAD-MAGNUS: KINEMATICS & GRAVITY ENGINE\n"
            f"============================================\n"
            f"[SYSTEM STATUS]\n"
            f" MODE       : {ap_status}\n"
            f" STATE      : {status_msg}\n"
            f" STRESS(T)  : {drag_curr:0.0f} / {self.drag_limit:0.0f} N\n"
            f" LOAD       : {stress_bar_visual} {stress_pct:.1f}%\n"
            f"\n"
            f"[KINEMATICS (FLIGHT DATA - 6DoF)]\n"
            f" Altitude   : {self.kin.pos[1]:.1f} m (0 is max height)\n"
            f" Vert. Vel. : {self.kin.vel[1]:.2f} m/s\n"
            f" Tether     : {tether_state}\n"
            f" Actual Pitch: {actual_pitch_deg:0.1f} deg (Target Winch: {self.val_pitch:0.1f} deg)\n"
            f"\n"
            f"[FORCES (TRUE VECTOR SUM)]\n"
            f" F_Net Y (Lift - W): {t['diag_f_net_y']:.0f} N\n"
            f" Global Lift: {t['lift_total_kg']*9.81:.0f} N\n"
            f" Payload    : {self.val_target_payload:.0f} kg\n"
            f" Winch Load : 200.0 kg\n"
        )
        
        text_block_2 = (
            f"[LOCAL DRAG DIAGNOSTICS]\n"
            f" Global Z (Drag) : {t['diag_drag_total']:.0f} N\n"
            f" > Parachute Up  : {t['diag_parachute_drag']:.0f} N (Airbrake System)\n"
            f" > Drone Thrust  : {t['diag_thrust_vtol']:.0f} N (Active Propellers)\n"
            f" > Cones Z-Force : {t['diag_drag_cones']:.0f} N (Includes Magnus)\n"
            f" > Tubes Z-Force : {t['diag_drag_tubes']:.0f} N\n"
            f" > Frame Z-Force : {t['diag_drag_frame']:.0f} N\n"
            f"\n"
            f"[AERODYNAMICS & CONTROL]\n"
            f" Wind Speed : {self.val_wind:.1f} m/s\n"
            f" Tube RPM   : Top: {t.get('tube_rpm_top',0):.0f}\n"
            f"              Bot: {t.get('tube_rpm_bot',0):.0f}\n"
            f" Cone RPM   : Top: {t.get('cone_rpm_top',0):.0f}\n"
            f"              Bot: {t.get('cone_rpm_bot',0):.0f}\n"
            f" Rotor RPM  : Top: {self.rotor_rpm_top:.1f}\n"
            f"              Bot: {self.rotor_rpm_bot:.1f}\n"
            f"\n"
            f"[POWER ANALYSIS (META-BEM 3-STRIP LOAD)]\n"
            f" Betz Limit : {t['betz_limit_kw']:.2f} kW\n"
            f" Gen Load   : {self.val_gen_load:.1f} %\n"
            f" -----------------------------------\n"
            f" GENERATORS : +{(t['gen_top_kw'] + t['gen_bot_kw']):.2f} kW\n"
            f" MTR CONES  : -{(t['motor_cone_top_kw'] + t['motor_cone_bot_kw']):.2f} kW\n"
            f" MTR TUBES  : -{(t['motor_tube_top_kw'] + t['motor_tube_bot_kw']):.2f} kW\n"
            f" -----------------------------------\n"
            f" NET POWER  : {t['net_power_kw']:.3f} kW\n"
        )
        
        width, height = self.p.window_size
        self.p.add_text(text_block_1, position=(width - 400, height - 350), color=status_col if self.structural_failure else ap_color, font_size=7, font='courier', shadow=False, name='hud_block1')
        self.p.add_text(text_block_2, position=(width - 400, height - 760), color='black' if not self.structural_failure else 'white', font_size=7, font='courier', shadow=False, name='hud_block2')

    def set_wind(self, val): self.val_wind = val
    def set_pitch(self, val): 
        if not self.auto_pilot: self.val_pitch = val
    def set_spin_lift(self, val): 
        if not self.auto_pilot: self.val_spin_blue = val
    def set_spin_drive(self, val): 
        if not self.auto_pilot: self.val_spin_drive = val
    def set_gen_load(self, val): 
        if not self.auto_pilot: self.val_gen_load = val
    def set_target_payload(self, val): 
        self.val_target_payload = val
        self.kin.mass = val + 200.0 

    def calculate_geometry(self):
        f = self.fold_factor
        curr_W = self.Max_W * (1 - f) + self.Min_W * f
        curr_H = np.sqrt(self.Beam_Len**2 - curr_W**2)
        curr_L = self.Max_Tube_L * (1 - f) + self.Min_Tube_L * f
        
        pos_offset = self.kin.pos
        
        pitch_rad = self.kin.angles[0]
        c = np.cos(pitch_rad); s = np.sin(pitch_rad)
        Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        
        def trans(p): return Rx.dot(p) + pos_offset
        
        tube_y_top = curr_H + 2.5; tube_y_bot = -curr_H - 2.5
        
        anchor_ground = np.array([0.0, self.ground_level, -600.0])
        vec_to_anchor = anchor_ground - pos_offset
        dist = np.linalg.norm(vec_to_anchor)
        bridle_L = 30.0
        
        if dist > 0.001:
            dir_A = vec_to_anchor / dist
            knot_pos = pos_offset + dir_A * min(bridle_L, dist)
        else:
            knot_pos = pos_offset
            
        knot_pos[1] = max(self.ground_level, knot_pos[1])
        winch_global = knot_pos
        
        geo = {
            'W': curr_W, 'H': curr_H, 'L': curr_L, 'Rx': Rx, 'pos_offset': pos_offset,
            'TL': trans([-curr_W,  curr_H, self.Z_F]), 'BR': trans([ curr_W, -curr_H, self.Z_F]),
            'TR': trans([ curr_W,  curr_H, self.Z_B]), 'BL': trans([-curr_W, -curr_H, self.Z_B]),
            'p0_T': trans([-curr_L/2, tube_y_top, 0]), 'p1_T': trans([ curr_L/2, tube_y_top, 0]),
            'p0_B': trans([-curr_L/2, tube_y_bot, 0]), 'p1_B': trans([ curr_L/2, tube_y_bot, 0]),
            'hc_TL': trans([-curr_W/2,  curr_H/2, self.Z_F + 0.5]),
            'hc_BR': trans([ curr_W/2, -curr_H/2, self.Z_F + 0.5]),
            'hc_TR': trans([ curr_W/2,  curr_H/2, self.Z_B - 0.5]),
            'hc_BL': trans([-curr_W/2, -curr_H/2, self.Z_B - 0.5]),
            'Winch': winch_global, 'Anchor': anchor_ground
        }
        return geo

    def build_scene(self):
        geo = self.calculate_geometry()
        
        grid_sea = pv.Plane(center=(0, self.ground_level, -300), direction=(0, 1, 0), i_size=1500, j_size=1500, i_resolution=60, j_resolution=60)
        self.sea_part = ScenePart(self.p, grid_sea, 'dodgerblue', opacity=0.4, wireframe=True)
        
        self.pod_part = ScenePart(self.p, create_solid_template(0.6), 'darkorange')
        self.buoy_part = ScenePart(self.p, create_solid_template(2.0), 'gold')
        self.buoy_part.update_transform(geo['Anchor'] - np.array([3,0,0]), geo['Anchor'] + np.array([3,0,0]))
        
        self.frame_parts['Front'] = ScenePart(self.p, create_grid_mesh(geo['TL'], geo['BR'], 0.25, 0.25), 'silver')
        self.frame_parts['Back']  = ScenePart(self.p, create_grid_mesh(geo['TR'], geo['BL'], 0.25, 0.25), 'darkgrey')
        self.frame_parts['Pivot'] = ScenePart(self.p, create_solid_template(0.4), 'black')
        
        self.para_canopy = ScenePart(self.p, pv.Cone(center=(0,0,0), direction=(0,1,0), height=8.0, radius=12.0, resolution=32), 'lightblue', opacity=0.85)
        self.para_canopy.set_visibility(False)
        self.para_ropes = [ScenePart(self.p, create_grid_mesh([0,0,0], [1,0,0], 0.05, 0.05), 'black') for _ in range(4)]
        for r in self.para_ropes: r.set_visibility(False)

        self.winch_part = ScenePart(self.p, create_solid_template(0.6), 'darkred')
        self.ropes['TL'] = ScenePart(self.p, create_grid_mesh([0,0,0], [1,0,0], 0.04, 0.04), 'black')
        self.ropes['TR'] = ScenePart(self.p, create_grid_mesh([0,0,0], [1,0,0], 0.04, 0.04), 'black')
        self.ropes['BL'] = ScenePart(self.p, create_grid_mesh([0,0,0], [1,0,0], 0.04, 0.04), 'black')
        self.ropes['BR'] = ScenePart(self.p, create_grid_mesh([0,0,0], [1,0,0], 0.04, 0.04), 'black')
        self.ropes['Center'] = ScenePart(self.p, create_grid_mesh([0,0,0], [1,0,0], 0.04, 0.04), 'black')
        self.ropes['Main'] = ScenePart(self.p, create_grid_mesh([0,0,0], [1,0,0], 0.08, 0.08), 'white')
        self.ropes['SideL'] = ScenePart(self.p, create_grid_mesh([0,0,0], [1,0,0], 0.03, 0.03), 'black')
        self.ropes['SideR'] = ScenePart(self.p, create_grid_mesh([0,0,0], [1,0,0], 0.03, 0.03), 'black')

        corners_info = [('TL', 1), ('BR', 1), ('TR', -1), ('BL', -1)]
        spoke_idx = 0 
        for i, (tag, spin_dir) in enumerate(corners_info):
            joint = ScenePart(self.p, create_grid_mesh([0,0,0], [0,0,1], 0.5, 0.5), 'grey')
            strut = ScenePart(self.p, create_grid_mesh([0,0,0], [0,0,1], 0.15, 0.15), 'grey')
            m_body = ScenePart(self.p, create_solid_template(0.35), 'black')
            m_shaft = ScenePart(self.p, create_solid_template(0.12), 'darkred')
            m_wheel = ScenePart(self.p, create_solid_template(0.2), 'darkred')
            self.strut_parts.append({'tag': tag, 'joint': joint, 'strut': strut, 'body': m_body, 'shaft': m_shaft, 'wheel': m_wheel})
            gen = ScenePart(self.p, create_solid_template(0.25), 'darkgrey')
            clamp = ScenePart(self.p, create_solid_template(0.35), 'silver')
            local_parts = [ScenePart(self.p, create_solid_template(0.2), 'black'),
                           ScenePart(self.p, create_grid_mesh([0,0,-0.3], [0,0,0.3], 4.0, 4.0, res=40), 'red' if spin_dir==1 else 'darkred', 0.8)]
            self.rotors_matrix_parts.append({'tag': tag, 'parts': local_parts, 'spin_dir': spin_dir, 'gen': gen, 'clamp': clamp})
            for deg in [0, 120, 240]:
                spoke = ScenePart(self.p, create_grid_mesh([0,0,0], [1,0,0], 0.15, 0.37, res=40), 'orange')
                tape = ScenePart(self.p, create_strip_mesh_init([0,0,0], [1,0,0], 0.15, 0.37, 0.0), 'black')
                vectors = {}
                for key, col, scale in [('air_res_A','orange',1), ('air_res_B','orange',1), ('air_amb','cyan',1), 
                                        ('air_rot','yellow',1), ('force_res','purple',2), ('force_drive','red',2), ('force_load','blue',1)]:
                    vectors[key] = ScenePart(self.p, create_arrow_template(scale=scale), col)
                    vectors[key].set_visibility(False)
                self.cone_vector_parts[spoke_idx] = vectors
                self.dynamic_spokes.append({'tag': tag, 'base_angle': np.radians(deg), 'spin_dir': spin_dir, 'spoke': spoke, 'tape': tape, 'id': spoke_idx, 'strips': [{'a': 0.0, 'a_prime': 0.0} for _ in range(3)]})
                spoke_idx += 1

        for pos in ['Top', 'Bot']:
            bellows = ScenePart(self.p, create_bellows_mesh_init([0,0,0], [1,0,0], 1.1, 1.1, 24), 'blue')
            ep1 = ScenePart(self.p, create_solid_template(1.75), 'purple')
            ep2 = ScenePart(self.p, create_solid_template(1.75), 'purple')
            kn1 = ScenePart(self.p, create_solid_template(0.35), 'grey')
            kn2 = ScenePart(self.p, create_solid_template(0.35), 'grey')
            ribs = []
            for ang in np.linspace(0, 2*np.pi, 4, endpoint=False):
                rib = ScenePart(self.p, create_strip_mesh_init([0,0,0], [1,0,0], 1.1, 1.1, 0.0), 'black')
                ribs.append({'part': rib, 'base': ang})
            
            air_arrows = []
            for i in range(5):
                arr_top = ScenePart(self.p, create_arrow_template(), 'lime'); arr_top.set_visibility(False)
                arr_bot = ScenePart(self.p, create_arrow_template(), 'lime'); arr_bot.set_visibility(False)
                air_arrows.append({'top': arr_top, 'bot': arr_bot, 'idx': i})
            
            self.tube_air_parts.append({'pos': pos, 'arrows': air_arrows})
            lift_arrow = ScenePart(self.p, create_arrow_template(scale=3.0), 'red'); lift_arrow.set_visibility(False)
            self.tube_force_parts.append({'pos': pos, 'arrow': lift_arrow})
            self.blue_tubes.append({'pos': pos, 'bellows': bellows, 'ribs': ribs, 'ep1': ep1, 'ep2': ep2, 'kn1': kn1, 'kn2': kn2, 'strips': [{'a': 0.0, 'a_prime': 0.0} for _ in range(3)]})

        self.update_geometry()

    def update_sliders_ghost(self):
        try:
            if self.sl_pitch: self.sl_pitch.GetRepresentation().SetValue(self.val_pitch)
            if self.sl_spin_lift: self.sl_spin_lift.GetRepresentation().SetValue(self.val_spin_blue)
            if self.sl_spin_drive: self.sl_spin_drive.GetRepresentation().SetValue(self.val_spin_drive)
            if self.sl_gen_load: self.sl_gen_load.GetRepresentation().SetValue(self.val_gen_load)
        except: pass

    def run_auto_pilot_logic(self):
        if not self.auto_pilot: 
            self.governor_status = "IDLE"
            self.flight_phase = "HARVEST"
            if hasattr(self, 'vtol_base_rpm'): del self.vtol_base_rpm
            if hasattr(self, 'takeoff_base_rpm'): del self.takeoff_base_rpm
            return
            
        current_alt = self.kin.pos[1]
        actual_pitch_deg = np.degrees(-self.kin.angles[0])
        avg_rotor_rpm = (self.rotor_rpm_top + self.rotor_rpm_bot) / 2.0
        current_vz = self.kin.vel[1]
        
        height_above_ground = max(1.0, current_alt - self.ground_level)
        wind_shear_mult = max(0.1, np.log(height_above_ground / 0.1) / np.log(10.0 / 0.1))
        local_wind_mag = self.val_wind * wind_shear_mult

        if self.flight_phase == "LANDED":
            if self.val_wind >= 14.0:  
                self.flight_phase = "TAKEOFF"
        elif self.flight_phase == "TAKEOFF":
            if current_alt >= -320.0 and actual_pitch_deg <= 50.0: 
                self.flight_phase = "PASSIVE_ASCENT"
            elif self.val_wind < 8.0:
                self.flight_phase = "VTOL"
        elif self.flight_phase == "PASSIVE_ASCENT":
            if current_alt >= -150.0: 
                self.flight_phase = "TRANSITION_TO_HARVEST"  
            elif local_wind_mag < 10.0: 
                self.flight_phase = "TAKEOFF"
        elif self.flight_phase == "TRANSITION_TO_HARVEST":
            # Μπαίνει στο Harvest είτε αν κατεβάσει μύτη (νορμάλ αέρας), είτε αν ζορίζεται πολύ (τυφώνας)
            if actual_pitch_deg < 15.0 or (self.current_total_drag / self.drag_limit) > 0.70:
                self.flight_phase = "HARVEST"
            elif self.val_wind < 6.0:
                self.flight_phase = "GLIDE"
        elif self.flight_phase == "VTOL":
            if current_alt <= -398.0:
                self.flight_phase = "LANDED"
            elif self.val_wind >= 14.0:
                self.flight_phase = "TAKEOFF"
        elif self.flight_phase == "PARACHUTE":
            if current_alt <= -380.0:  
                self.flight_phase = "VTOL"
        elif self.flight_phase == "GLIDE":
            if current_alt <= -200.0:  
                self.flight_phase = "PARACHUTE"
            elif local_wind_mag >= 8.0:
                self.flight_phase = "PASSIVE_ASCENT" if current_alt < -150.0 else "TRANSITION_TO_HARVEST"
        elif self.flight_phase == "HARVEST":
            if self.val_wind < 6.0:
                self.flight_phase = "GLIDE"

        if self.flight_phase != "VTOL" and hasattr(self, 'vtol_base_rpm'): del self.vtol_base_rpm
        if self.flight_phase != "TAKEOFF" and hasattr(self, 'takeoff_base_rpm'): del self.takeoff_base_rpm

        if self.flight_phase == "LANDED":
            self.governor_status = "LANDED SAFE ON OCEAN"
            self.val_spin_drive = 0.0  
            self.val_spin_blue = 0.0   
            self.val_gen_load = 0.0    
            target_pitch = 90.0
            error = target_pitch - actual_pitch_deg
            self.val_pitch -= np.clip(error * 0.1, -0.5, 0.5)
            self.val_pitch = np.clip(self.val_pitch, -90.0, 90.0)
            self.update_sliders_ghost()

        elif self.flight_phase == "TAKEOFF":
            if current_alt < -320.0:
                self.governor_status = "VTOL BOOST (VERTICAL CLIMB)"
                target_pitch = 90.0 
            else:
                self.governor_status = "VTOL BOOST & PITCHING"
                target_pitch = 45.0 

            error = target_pitch - actual_pitch_deg
            self.val_pitch -= np.clip(error * 0.1, -0.5, 0.5) 
            
            target_vz = 6.5 
            vel_error = target_vz - current_vz 
            
            if not hasattr(self, 'takeoff_base_rpm'):
                self.takeoff_base_rpm = 2500.0 
            
            self.takeoff_base_rpm += vel_error * 5.0
            self.takeoff_base_rpm = np.clip(self.takeoff_base_rpm, 0.0, 4800.0)
            p_term = vel_error * 100.0
            
            self.val_spin_drive = np.clip(self.takeoff_base_rpm + p_term, 0.0, 5000.0)
            self.val_gen_load = -80.0  
            self.val_spin_blue = 0.0
            
            if hasattr(self, 'main_line_reeled_in') and self.main_line_reeled_in > 0.0:
                self.main_line_reeled_in = max(0.0, self.main_line_reeled_in - 5.0 * 0.04)

            self.val_pitch = np.clip(self.val_pitch, -90.0, 90.0)
            self.update_sliders_ghost()

        elif self.flight_phase == "PASSIVE_ASCENT":
            self.governor_status = "PASSIVE WIND ASCENT (KITE MODE)"
            
            if self.val_gen_load < 0.0:
                self.val_gen_load = 0.0
                
            stress_ratio = self.current_total_drag / self.drag_limit
            
            base_target_pitch = 45.0
            if stress_ratio > 0.70:
                depower_angle = (stress_ratio - 0.70) * 150.0 
                target_pitch = np.clip(base_target_pitch + depower_angle, 45.0, 85.0)
                self.governor_status = "PASSIVE WIND ASCENT (DEPOWERING PITCH)"
            else:
                target_pitch = base_target_pitch
                
            # --- FIX 1: ΕΠΙΣΤΡΟΦΗ ΣΤΟ ΣΤΑΘΕΡΟ ΑΡΝΗΤΙΚΟ ΠΡΟΣΗΜΟ (-=) ---
            error = target_pitch - actual_pitch_deg
            self.val_pitch -= np.clip(error * 0.05, -0.2, 0.2)
            self.val_pitch = np.clip(self.val_pitch, -90.0, 90.0)
            
            # Στιγμιαίο RPM Depower (Η δικλείδα ασφαλείας σου)
            if stress_ratio > 0.60:
                target_tubes = 0.0
                target_cones = 0.0
            elif stress_ratio > 0.45:
                reduce_factor = np.clip((stress_ratio - 0.65) * 5.0, 0.0, 1.0)
                target_tubes = 180.0 * (1.0 - reduce_factor)
                target_cones = 1400.0 * (1.0 - reduce_factor)
            else:
                target_tubes = 180.0
                target_cones = 1400.0
                
            if self.val_spin_blue < target_tubes:
                self.val_spin_blue = min(target_tubes, self.val_spin_blue + 2.0)
            elif self.val_spin_blue > target_tubes:
                self.val_spin_blue = max(target_tubes, self.val_spin_blue - 5.0)
                
            if self.val_spin_drive > target_cones + 20.0:
                self.val_spin_drive -= 50.0  
            elif self.val_spin_drive < target_cones - 20.0:
                self.val_spin_drive += 10.0
                
            # --- FIX 2: PROPORTIONAL GOVERNOR ΓΙΑ ΤΗ ΓΕΝΝΗΤΡΙΑ ---
            ideal_aero_rpm = 30.0 + (self.val_wind - 8.0) * 4.0 
            target_rpm_limit = min(ideal_aero_rpm, 238.0)
            
            rpm_error = avg_rotor_rpm - target_rpm_limit
            # Αλλάζει το φορτίο ανάλογα με το πόσο μακριά είναι από το στόχο
            self.val_gen_load = np.clip(self.val_gen_load + rpm_error * 0.01, 0.0, 15.0) 
                
            self.update_sliders_ghost()

        elif self.flight_phase == "TRANSITION_TO_HARVEST":
            self.governor_status = "PREPARING HARVEST (STABILIZING PITCH)"
            
            if self.val_gen_load < 0.0:
                self.val_gen_load = 0.0

            stress_ratio = self.current_total_drag / self.drag_limit
            
            # --- ΤΟ ΜΥΣΤΙΚΟ: Βάζουμε Depower ΚΑΙ στη μετάβαση! ---
            base_target_pitch = 5.0
            if stress_ratio > 0.70:
                depower_angle = (stress_ratio - 0.70) * 160.0 
                target_pitch = np.clip(base_target_pitch + depower_angle, 5.0, 80.0)
                self.governor_status = "PREPARING HARVEST (DEPOWERING)"
            else:
                target_pitch = base_target_pitch

            # Το σταθερό, σωστό PID (-) για να κατεβάσει μύτη ομαλά
            error = target_pitch - actual_pitch_deg
            self.val_pitch -= np.clip(error * 0.05, -0.2, 0.2)
            self.val_pitch = np.clip(self.val_pitch, -90.0, 90.0)
            
            if self.val_spin_blue < 180.0:
                self.val_spin_blue = min(180.0, self.val_spin_blue + 2.0)
                
            if hasattr(self, 'main_line_reeled_in') and self.main_line_reeled_in > 0.0:
                self.main_line_reeled_in = max(0.0, self.main_line_reeled_in - 5.0 * 0.04)

            # Στιγμιαίο RPM φρένο αν ζοριστεί απότομα στη μετάβαση
            if stress_ratio > 0.85:
                target_cones = 0.0
            elif stress_ratio > 0.70:
                reduce_factor = np.clip((stress_ratio - 0.70) * 5.0, 0.0, 1.0)
                target_cones = 1400.0 * (1.0 - reduce_factor)
            else:
                target_cones = 1400.0

            if self.val_spin_drive > target_cones + 20.0:
                self.val_spin_drive -= 50.0  
            elif self.val_spin_drive < target_cones - 20.0:
                self.val_spin_drive += 10.0
                
            self.update_sliders_ghost()

        elif self.flight_phase == "VTOL":
            self.governor_status = "VTOL FINAL DESCENT"
            target_pitch = 90.0
            error = target_pitch - actual_pitch_deg
            self.val_pitch -= np.clip(error * 0.1, -0.5, 0.5) 
            
            target_vz = -3.5 
            vel_error = target_vz - current_vz 
            
            if not hasattr(self, 'vtol_base_rpm'):
                self.vtol_base_rpm = 1000.0 
            
            self.vtol_base_rpm += vel_error * 5.0
            self.vtol_base_rpm = np.clip(self.vtol_base_rpm, 0.0, 4500.0)
            p_term = vel_error * 100.0
            
            self.val_spin_drive = np.clip(self.vtol_base_rpm + p_term, 0.0, 4500.0)
            self.val_gen_load = -80.0  
            self.val_spin_blue = 0.0

            self.main_line_reeled_in += 5.0 * 0.04
            
            self.val_pitch = np.clip(self.val_pitch, -90.0, 90.0)
            self.update_sliders_ghost()

        elif self.flight_phase == "PARACHUTE":
            self.governor_status = "PARACHUTE DEPLOYED"
            target_pitch = 90.0 
            error = target_pitch - actual_pitch_deg
            self.val_pitch -= np.clip(error * 0.1, -0.5, 0.5)
                
            self.val_spin_drive = 0.0 
            self.val_spin_blue = 0.0
            self.val_gen_load = 0.0 
            self.val_pitch = np.clip(self.val_pitch, -90.0, 90.0)
            self.update_sliders_ghost()

        elif self.flight_phase == "GLIDE":
            self.governor_status = "GLIDING (IDLE DESCENT / ENERGY SAVING)"
            
            # 1. Πλακέ πτώση στις 65 μοίρες (χρησιμοποιεί την επιφάνεια σαν φρένο)
            target_pitch = 65.0
            error = target_pitch - actual_pitch_deg
            self.val_pitch -= np.clip(error * 0.1, -0.2, 0.2) 
            self.val_pitch = np.clip(self.val_pitch, -90.0, 90.0)
            
            # 2. Μαζεύουμε σκοινί
            self.main_line_reeled_in += 8.0 * 0.04

            # 3. Σβήνουμε τους μπλε κυλίνδρους
            if self.val_spin_blue > 0.0:
                self.val_spin_blue = max(0.0, self.val_spin_blue - 2.0)

            # 4. Ρελαντί στους ρότορες (μόνο 300 RPM για αεροδυναμική ευστάθεια, όχι για ρεύμα)
            target_cones = 300.0
            if self.val_spin_drive > target_cones + 20.0:
                self.val_spin_drive -= 40.0 
            elif self.val_spin_drive < target_cones - 20.0:
                self.val_spin_drive += 10.0

            # 5. Αποσυνδέουμε πλήρως τις γεννήτριες (Freewheeling: Ούτε καίμε, ούτε παράγουμε)
            if self.val_gen_load > 0.0:
                self.val_gen_load = max(0.0, self.val_gen_load - 1.0)
            elif self.val_gen_load < 0.0:
                self.val_gen_load = min(0.0, self.val_gen_load + 1.0)
                
            self.update_sliders_ghost()

        elif self.flight_phase == "HARVEST":
            self.governor_status = "HARVESTING (GREEDY MODE)"
            
            if hasattr(self, 'main_line_reeled_in') and self.main_line_reeled_in > 0.0:
                self.main_line_reeled_in = max(0.0, self.main_line_reeled_in - 5.0 * 0.04)
            
            if self.val_spin_blue < 180.0:
                self.val_spin_blue = min(180.0, self.val_spin_blue + 2.0)
                
            base_target_pitch = 5.0
            stress_ratio = self.current_total_drag / self.drag_limit
            
            if stress_ratio > 0.75:
                depower_angle = (stress_ratio - 0.75) * 160.0 
                target_pitch = np.clip(base_target_pitch + depower_angle, 5.0, 60.0)
                self.governor_status = "HARVESTING (ACTIVE DEPOWER LIMITER)"
            else:
                target_pitch = base_target_pitch
                self.governor_status = "HARVESTING (GREEDY MODE)"

            error = target_pitch - actual_pitch_deg
            self.val_pitch -= np.clip(error * 0.05, -0.4, 0.4) 
            self.val_pitch = np.clip(self.val_pitch, -90.0, 90.0)



            target_cones = 1400.0
            if self.val_spin_drive > target_cones + 20.0:
                self.val_spin_drive -= 25.0  
            elif self.val_spin_drive < target_cones - 20.0:
                self.val_spin_drive += 10.0

            rotor_radius = 4.0 
            safe_tip_speed_mps = 25.0 
            max_safe_rpm = (safe_tip_speed_mps / rotor_radius) * 9.549 
            ideal_aero_rpm = 30.0 + (self.val_wind - 8.0) * 4.0 
            dynamic_target_rpm = min(ideal_aero_rpm, max_safe_rpm)
            
            # Hysteresis (Νεκρή ζώνη ±5 RPM) με μαλακό βήμα 0.1%
            if avg_rotor_rpm > (dynamic_target_rpm + 5.0):
                self.val_gen_load = min(98.0, self.val_gen_load + 0.1) 
            elif avg_rotor_rpm < (dynamic_target_rpm - 5.0):
                self.val_gen_load = max(0.0, self.val_gen_load - 0.1)  

	    # Στιγμιαίο RPM Depower (Η δικλείδα ασφαλείας σου)
            if stress_ratio > 0.60:
                target_tubes = 0.0
                target_cones = 0.0
            elif stress_ratio > 0.45:
                reduce_factor = np.clip((stress_ratio - 0.65) * 5.0, 0.0, 1.0)
                target_tubes = 180.0 * (1.0 - reduce_factor)
                target_cones = 1400.0 * (1.0 - reduce_factor)
            else:
                target_tubes = 180.0
                target_cones = 1400.0

            self.update_sliders_ghost()

    def trigger_reset(self, state):
        if not state: return 
        
        self.structural_failure = False
        self.spinning = False
        self.auto_pilot = False
        self.flight_phase = "HARVEST"
        
        if hasattr(self, 'base_tether_lengths'):
            del self.base_tether_lengths
        if hasattr(self, 'vtol_base_rpm'): del self.vtol_base_rpm
        if hasattr(self, 'takeoff_base_rpm'): del self.takeoff_base_rpm
        
        self.kin.pos = np.array([0.0, 0.0, 0.0])
        self.kin.vel = np.array([0.0, 0.0, 0.0])
        self.kin.angles = np.array([0.0, 0.0, 0.0])
        self.kin.omega = np.array([0.0, 0.0, 0.0])
        
        self.rotor_rpm_top = 0.0
        self.rotor_rpm_bot = 0.0
        self.current_total_drag = 0.0
        self.sim_time = 0.0
        self.para_inflation = 0.0 
        
        self.val_pitch = 0.0
        self.main_line_reeled_in = 0.0
        self.val_wind = 10.0
        self.val_gen_load = 18.0
        self.val_spin_drive = 3500.0
        self.val_spin_blue = 180.0
        
        self.p.set_background('white')
        self.set_labels_color('black')
        for part in self.frame_parts.values(): part.set_color(part.base_color)
        for bt in self.blue_tubes: bt['bellows'].set_color(bt['bellows'].base_color)
        for sp in self.dynamic_spokes: sp['spoke'].set_color(sp['spoke'].base_color); sp['tape'].set_color(sp['tape'].base_color)
        for k, rope in self.ropes.items(): rope.set_color('white' if k=='Main' else 'black')
        if self.winch_part: self.winch_part.set_color(self.winch_part.base_color)
        
        for btn in [self.btn_spin, self.btn_fold, self.btn_ap, self.btn_reset, 
                    self.btn_tube_air, self.btn_tube_force, self.btn_cone_air_res, 
                    self.btn_cone_force_res, self.btn_cone_air_comp, self.btn_cone_force_comp]:
            if btn: btn.GetRepresentation().SetState(0)
            
        self.show_tube_air = False
        self.show_tube_force = False
        self.show_cone_air_res = False
        self.show_cone_air_comp = False
        self.show_cone_force_res = False
        self.show_cone_force_comp = False
        
        self.update_sliders_ghost()
        try:
            if self.sl_wind: self.sl_wind.GetRepresentation().SetValue(self.val_wind)
        except: pass
        self.update_geometry()

    def update_geometry(self):
        if self.structural_failure: return
        
        dt = 0.04
        self.sim_time += dt
        
        self.run_auto_pilot_logic()

        geo = self.calculate_geometry()
        Rx = geo['Rx']
        pos_offset = geo['pos_offset']
        rho = self.env.rho
        
        self.p.camera.focal_point = pos_offset.tolist()
        
        is_vtol_or_para_or_takeoff = self.flight_phase in ["VTOL", "PARACHUTE", "TAKEOFF"]
        pitch_cmd = 0.0
        
        if self.auto_pilot and is_vtol_or_para_or_takeoff and self.spinning:
            target_pitch_rad = -1.5708 
            pitch_error = target_pitch_rad - self.kin.angles[0]
            
            pid_p = pitch_error * 2000.0
            pid_d = -self.kin.omega[0] * 1000.0
            pitch_cmd = pid_p + pid_d

        base_cone_rpm = self.val_spin_drive
        cone_rpm_top = np.clip(base_cone_rpm + pitch_cmd, 0.0, 5000.0)
        cone_rpm_bot = np.clip(base_cone_rpm - pitch_cmd, 0.0, 5000.0)

        tube_rpm_top = self.val_spin_blue
        tube_rpm_bot = self.val_spin_blue
        
        if self.spinning:
            self.telemetry['tube_rpm_top'] = tube_rpm_top
            self.telemetry['tube_rpm_bot'] = tube_rpm_bot
            self.telemetry['cone_rpm_top'] = cone_rpm_top
            self.telemetry['cone_rpm_bot'] = cone_rpm_bot
            self.tube_spin_phase_top -= tube_rpm_top * 0.1047 * 0.2
            self.tube_spin_phase_bot -= tube_rpm_bot * 0.1047 * 0.2
        else:
            self.telemetry.update({'tube_rpm_top': 0.0, 'tube_rpm_bot': 0.0, 'cone_rpm_top': 0.0, 'cone_rpm_bot': 0.0})

        height_above_ground = max(1.0, self.kin.pos[1] - self.ground_level)
        wind_shear_mult = max(0.1, np.log(height_above_ground / 0.1) / np.log(10.0 / 0.1))
        local_wind_mag = self.val_wind * wind_shear_mult
        
        wind_vec_global = np.array([-self.kin.vel[0], -self.kin.vel[1], local_wind_mag * 0.85 - self.kin.vel[2]])
        v_app_mag_global = np.linalg.norm(wind_vec_global)
        
        Sys_Net_Force = np.array([0.0, 0.0, 0.0])
        Sys_Net_Torque = np.array([0.0, 0.0, 0.0]) 
        
        gravity_force = np.array([0.0, -self.kin.mass * 9.81, 0.0])
        Sys_Net_Force += gravity_force

        tube_r = 1.1; tube_len = geo['L']
        tube_axis = Rx.dot(np.array([1.0, 0.0, 0.0]))
        
        p_mech_tube_top = 0.0
        p_mech_tube_bot = 0.0
        
        total_tube_z_force = 0.0
        F_tubes_global_top = np.zeros(3); F_tubes_global_bot = np.zeros(3)
        lift_dir_top = np.zeros(3); lift_dir_bot = np.zeros(3)
        
        num_strips = 3
        dr_strip = tube_len / num_strips
        area_proj_strip = dr_strip * (tube_r * 2.0)
        
        for pos_key, rpm_val in [('Top', tube_rpm_top), ('Bot', tube_rpm_bot)]:
            is_top = (pos_key == 'Top')
            center_y = geo['H'] + 2.5 if is_top else -geo['H'] - 2.5
            pos_tube_center = pos_offset + Rx.dot([0.0, center_y, 0.0])
            
            F_tube_total = np.zeros(3)
            
            for step in range(num_strips):
                offset_x = (step - 1) * dr_strip
                p_local = pos_tube_center + tube_axis * offset_x
                
                v_app_perp = wind_vec_global - np.dot(wind_vec_global, tube_axis) * tube_axis
                safe_v_app_perp = max(np.linalg.norm(v_app_perp), 0.01)
                v_app_perp_dir = v_app_perp / safe_v_app_perp
                
                dF, dP, _, _ = self.compute_strip_aero_forces(
                    v_inf=self.val_wind, a=0.0, a_prime=0.0, r_local_flat=0.0,
                    cyl_omega=rpm_val * 0.1047, rotor_omega=0.0, B_blades=1,
                    Hub_R=0.0, Cyl_L=tube_len, Cyl_r=tube_r, safe_v_app_perp=safe_v_app_perp,
                    v_app_perp_dir=v_app_perp_dir, cyl_vec_dir=tube_axis,
                    area_proj_strip=area_proj_strip, dr_strip=dr_strip, is_rotor=False
                )
                
                F_tube_total += dF
                total_tube_z_force += dF[2]
                Sys_Net_Torque += np.cross(p_local - pos_offset, dF)
                
                if is_top: 
                    p_mech_tube_top += dP
                    lift_dir_top = np.cross(v_app_perp_dir, tube_axis * np.sign(rpm_val) if rpm_val != 0 else tube_axis)
                else: 
                    p_mech_tube_bot += dP
                    lift_dir_bot = np.cross(v_app_perp_dir, tube_axis * np.sign(rpm_val) if rpm_val != 0 else tube_axis)

            Sys_Net_Force += F_tube_total
            if is_top: F_tubes_global_top = F_tube_total
            else: F_tubes_global_bot = F_tube_total
            
        if np.linalg.norm(lift_dir_top) > 0.001: lift_dir_top /= np.linalg.norm(lift_dir_top)
        if np.linalg.norm(lift_dir_bot) > 0.001: lift_dir_bot /= np.linalg.norm(lift_dir_bot)

        frame_drag_mag = 0.5 * rho * 15.0 * (v_app_mag_global**2) * 1.5
        wind_dir_global = wind_vec_global / max(v_app_mag_global, 0.001)
        F_frame = wind_dir_global * frame_drag_mag
        Sys_Net_Force += F_frame 
        
        local_omega = np.linalg.inv(Rx).dot(self.kin.omega)
        damping_torque_local = np.array([
            -10000.0 * local_omega[0] * abs(local_omega[0]),
            -50000.0 * local_omega[1] * abs(local_omega[1]), 
            -50000.0 * local_omega[2] * abs(local_omega[2])  
        ])
        Sys_Net_Torque += Rx.dot(damping_torque_local)

        parachute_drag_y = 0.0
        if self.flight_phase == "PARACHUTE":
            self.para_inflation = min(1.0, self.para_inflation + dt / 3.0) 
        else:
            self.para_inflation = max(0.0, self.para_inflation - dt / 2.0)

        if self.para_inflation > 0.01:
            v_kin_mag = np.linalg.norm(self.kin.vel)
            if v_kin_mag > 0.01:
                current_para_area = 450.0 * self.para_inflation
                para_drag_mag = 0.5 * rho * current_para_area * 1.5 * (v_kin_mag**2)
                para_drag_vec = -(self.kin.vel / v_kin_mag) * para_drag_mag
                Sys_Net_Force += para_drag_vec
                
                para_pos_physics = Rx.dot([0.0, 0.0, 3.0]) 
                Sys_Net_Torque += np.cross(para_pos_physics, para_drag_vec)
                parachute_drag_y = para_drag_vec[1]

        frame_n = Rx.dot(np.array([0.0, 0.0, 1.0])) 
        thrust_top = 0.0
        thrust_bot = 0.0
        
        if self.val_gen_load < 0 and self.spinning:
            thrust_top = 0.0005 * (cone_rpm_top**2) * (abs(self.val_gen_load) / 100.0) 
            thrust_bot = 0.0005 * (cone_rpm_bot**2) * (abs(self.val_gen_load) / 100.0) 
            
        F_prop_top = frame_n * thrust_top
        F_prop_bot = frame_n * thrust_bot
        
        Sys_Net_Force += F_prop_top + F_prop_bot
        
        pos_top_rotors = pos_offset + Rx.dot([0.0, geo['H']/2, 0.0])
        pos_bot_rotors = pos_offset + Rx.dot([0.0, -geo['H']/2, 0.0])
        Sys_Net_Torque += np.cross(pos_top_rotors - pos_offset, F_prop_top)
        Sys_Net_Torque += np.cross(pos_bot_rotors - pos_offset, F_prop_bot)
        
        self.telemetry['diag_thrust_vtol'] = (F_prop_top + F_prop_bot)[1]

        torque_top = 0.0; torque_bot = 0.0
        total_cone_z_force = F_prop_top[2] + F_prop_bot[2]
        p_mech_cone_top = 0.0; p_mech_cone_bot = 0.0
        
        cone_radius = 0.15; cone_length = 4.0; rotor_R_full = 4.0
        avg_cone_radius = 0.26 
        hc_map = {'TL': geo['hc_TL'] - pos_offset, 'BR': geo['hc_BR'] - pos_offset, 'TR': geo['hc_TR'] - pos_offset, 'BL': geo['hc_BL'] - pos_offset}

        dr_strip_cone = cone_length / num_strips
        area_proj_strip_cone = dr_strip_cone * (avg_cone_radius * 2.0)

        for item in self.dynamic_spokes:
            tag = item['tag']
            is_top = (tag in ['TL', 'TR'])
            hc = hc_map[tag] 
            
            drive_rpm = cone_rpm_top if is_top else cone_rpm_bot
            rotor_omega = (self.rotor_rpm_top if is_top else self.rotor_rpm_bot) * 0.1047
            
            orbit_a = item['base_angle'] + ((self.rotor_angle_top if is_top else self.rotor_angle_bot) * item['spin_dir'])
            rad_vec = Rx.dot(np.array([rotor_R_full*np.cos(orbit_a), rotor_R_full*np.sin(orbit_a), 0])) 
            
            p_s = hc + pos_offset; p_e = hc + rad_vec + pos_offset
            cone_spin = (self.spoke_spin_phase_top if is_top else self.spoke_spin_phase_bot) * item['spin_dir']
            item['spoke'].mesh.points, _ = math_pts_cyl(p_s, p_e, 0.15, 0.37, 40, 0.0)
            item['tape'].mesh.points, _ = math_pts_strip(p_s, p_e, 0.15, 0.37, 0.0, cone_spin)
            vecs = self.cone_vector_parts[item['id']]; mid_cone = (p_s + p_e) / 2.0
            
            rad_norm = rad_vec / np.linalg.norm(rad_vec)
            rot_axis = Rx.dot(np.array([0,0,1])) 
            tan_vec = np.cross(rot_axis, rad_norm)
            if item['spin_dir'] == -1: tan_vec *= -1
            
            total_force_spoke = np.zeros(3)
            visual_lift_dir = np.zeros(3)
            visual_app_vec = wind_vec_global.copy()
            
            if self.spinning:
                for step in range(num_strips):
                    r_local_flat = 0.0 + (step + 0.5) * dr_strip_cone
                    p_local = hc + pos_offset + rad_norm * r_local_flat
                    
                    v_tan_mag = rotor_omega * r_local_flat
                    headwind_vec = -tan_vec * v_tan_mag
                    apparent_vec_local = wind_vec_global + headwind_vec
                    
                    v_app_perp = apparent_vec_local - np.dot(apparent_vec_local, rad_norm) * rad_norm
                    safe_v_app_perp = max(np.linalg.norm(v_app_perp), 0.01)
                    v_app_perp_dir = v_app_perp / safe_v_app_perp
                    
                    a = item['strips'][step]['a']
                    a_prime = item['strips'][step]['a_prime']
                    
                    dF, dP, a_new, a_prime_new = self.compute_strip_aero_forces(
                        v_inf=self.val_wind, a=a, a_prime=a_prime, r_local_flat=r_local_flat,
                        cyl_omega=drive_rpm * 0.1047 * item['spin_dir'], rotor_omega=rotor_omega,
                        B_blades=3.0, Hub_R=0.0, Cyl_L=cone_length, Cyl_r=avg_cone_radius,
                        safe_v_app_perp=safe_v_app_perp, v_app_perp_dir=v_app_perp_dir,
                        cyl_vec_dir=rad_norm, area_proj_strip=area_proj_strip_cone, dr_strip=dr_strip_cone,
                        is_rotor=True
                    )
                    
                    item['strips'][step]['a'] = a_new
                    item['strips'][step]['a_prime'] = a_prime_new
                    
                    total_force_spoke += dF
                    Sys_Net_Torque += np.cross(p_local - pos_offset, dF)
                    
                    f_drive_mag = np.dot(dF, tan_vec)
                    if is_top: 
                        torque_top += f_drive_mag * r_local_flat
                        p_mech_cone_top += dP
                    else:      
                        torque_bot += f_drive_mag * r_local_flat
                        p_mech_cone_bot += dP
                        
                    if step == 1: 
                        visual_lift_dir = np.cross(apparent_vec_local, rad_norm * item['spin_dir'])
                        if np.linalg.norm(visual_lift_dir) > 0.001: visual_lift_dir /= np.linalg.norm(visual_lift_dir)
                        visual_app_vec = apparent_vec_local
                
                Sys_Net_Force += total_force_spoke 
                total_cone_z_force += total_force_spoke[2]
                
                force_drive_vec = tan_vec * np.dot(total_force_spoke, tan_vec)
                force_load_vec = total_force_spoke - force_drive_vec
                scale_A = 1.0 + (drive_rpm * 0.005)
                scale_B = max(0.1, 1.0 - (drive_rpm * 0.005))
            else: 
                headwind_vec = np.array([0,0,0], dtype=float); visual_app_vec = wind_vec_global
                visual_lift_dir = np.array([0,0,0], dtype=float); total_force_spoke = np.array([0,0,0], dtype=float)
                force_drive_vec = np.array([0,0,0], dtype=float); force_load_vec = np.array([0,0,0], dtype=float)
                scale_A = 1.0; scale_B = 1.0
                
                safe_v_wind = max(v_app_mag_global, 0.001)
                static_area = cone_length * (avg_cone_radius * 2.0)
                drag_static_mag = 0.5 * rho * static_area * 1.15 * (safe_v_wind**2) 
                
                drag_static = (wind_vec_global / safe_v_wind) * drag_static_mag
                Sys_Net_Force += drag_static
                Sys_Net_Torque += np.cross(hc, drag_static)
                total_cone_z_force += drag_static[2]

            offset_dist = 0.5 
            if np.linalg.norm(visual_lift_dir) > 0.001: lift_dir_norm = visual_lift_dir
            else: lift_dir_norm = np.cross(wind_vec_global, rad_norm); lift_dir_norm /= (np.linalg.norm(lift_dir_norm)+0.001)

            pos_A = mid_cone + lift_dir_norm * offset_dist - visual_app_vec * 0.5
            pos_B = mid_cone - lift_dir_norm * offset_dist - visual_app_vec * 0.5
            vecs['air_res_A'].update_transform(pos_A, pos_A + visual_app_vec * scale_A)
            vecs['air_res_A'].set_color('red' if self.spinning else 'orange'); vecs['air_res_A'].set_visibility(self.show_cone_air_res)
            vecs['air_res_B'].update_transform(pos_B, pos_B + visual_app_vec * scale_B)
            vecs['air_res_B'].set_color('blue' if self.spinning else 'orange'); vecs['air_res_B'].set_visibility(self.show_cone_air_res)
            origin = mid_cone - visual_app_vec * 0.8
            vecs['air_amb'].update_transform(origin, origin + wind_vec_global); vecs['air_amb'].set_visibility(self.show_cone_air_comp)
            vecs['air_rot'].update_transform(origin + wind_vec_global, origin + visual_app_vec); vecs['air_rot'].set_visibility(self.show_cone_air_comp and self.spinning)
            v_scale = 0.02; origin_f = mid_cone
            vecs['force_res'].update_transform(origin_f, origin_f + total_force_spoke * v_scale); vecs['force_res'].set_visibility(self.show_cone_force_res and self.spinning)
            vecs['force_drive'].update_transform(origin_f, origin_f + force_drive_vec * v_scale); vecs['force_drive'].set_visibility(self.show_cone_force_comp and self.spinning)
            vecs['force_load'].update_transform(origin_f + force_drive_vec * v_scale, origin_f + (force_drive_vec + force_load_vec) * v_scale); vecs['force_load'].set_visibility(self.show_cone_force_comp and self.spinning)

        node_map = {'TL': geo['TL'], 'TR': geo['TR'], 'BL': geo['BL'], 'BR': geo['BR']}
        
        if not hasattr(self, 'base_tether_lengths'):
            self.base_tether_lengths = {}
            for tag, c_pos in node_map.items():
                self.base_tether_lengths[tag] = np.linalg.norm(geo['Anchor'] - c_pos)
                
        k_spring = 12000.0  
        c_damper = 3500.0   
        winch_cmd = self.val_pitch * 0.5 
        
        max_stretch = 0.0
        for tag, corner_pos in node_map.items():
            dist = np.linalg.norm(geo['Anchor'] - corner_pos)
            t_len = self.base_tether_lengths[tag] - self.main_line_reeled_in
            t_len += winch_cmd if 'T' in tag else -winch_cmd
            if (dist - t_len) > max_stretch: 
                max_stretch = (dist - t_len)
                
        if max_stretch > 16.0:  
            self.main_line_reeled_in -= (max_stretch - 16.0)
            
        tether_tension_N = 0.0
        
        for tag, corner_pos in node_map.items():
            vec_to_anchor = geo['Anchor'] - corner_pos
            dist = np.linalg.norm(vec_to_anchor)
            
            target_len = self.base_tether_lengths[tag] - self.main_line_reeled_in
            
            if 'T' in tag: 
                target_len += winch_cmd  
            else:          
                target_len -= winch_cmd  
                
            stretch = dist - target_len
            
            if stretch > 0:
                dir_vec = vec_to_anchor / max(dist, 0.001)
                r_vec = corner_pos - self.kin.pos
                v_corner = self.kin.vel + np.cross(self.kin.omega, r_vec)
                
                vel_rel = np.dot(v_corner, -dir_vec) 
                
                F_mag = (stretch * k_spring) + (vel_rel * c_damper)
                F_mag = max(F_mag, 0.0) 
                
                F_vec = dir_vec * F_mag
                Sys_Net_Force += F_vec
                Sys_Net_Torque += np.cross(r_vec, F_vec)
                
                tether_tension_N += F_mag

        self.telemetry['diag_f_net_y'] = Sys_Net_Force[1]
        self.telemetry['diag_drag_total'] = Sys_Net_Force[2]
        self.telemetry['diag_drag_tubes'] = total_tube_z_force
        self.telemetry['diag_drag_cones'] = total_cone_z_force
        self.telemetry['diag_drag_frame'] = F_frame[2] 
        self.telemetry['diag_parachute_drag'] = parachute_drag_y

        if self.spinning:
            self.kin.step(dt, Sys_Net_Force, Sys_Net_Torque, Rx)
            
            if self.kin.pos[1] >= 0.0:
                self.telemetry['tether_state'] = 'TAUT'
            elif self.kin.pos[1] <= self.ground_level:
                if self.kin.vel[1] < -15.0: 
                    self.trigger_failure_mode()
                else: 
                    self.telemetry['tether_state'] = 'LANDED SAFE'
            else:
                self.telemetry['tether_state'] = 'SLACK (FALLING/HOVERING)'
        else:
            self.telemetry['tether_state'] = 'PARKED (IDLE)'

        self.current_total_drag = tether_tension_N
        if self.current_total_drag > self.drag_limit: 
            self.trigger_failure_mode()

        MAX_GEN_W = 150000.0   
        MAX_MOT_W = 15000.0    
        INERTIA_ROTOR = 8000.0 
        MAX_BRAKE_TORQUE = 80000.0 

        if self.spinning:
            eff_mot_tube_top = max(np.interp(min(p_mech_tube_top / MAX_MOT_W, 1.2), self.lut_load_pct, self.lut_eff_mot), 0.05)
            eff_mot_tube_bot = max(np.interp(min(p_mech_tube_bot / MAX_MOT_W, 1.2), self.lut_load_pct, self.lut_eff_mot), 0.05)
            eff_mot_cone_top = max(np.interp(min(p_mech_cone_top / MAX_MOT_W, 1.2), self.lut_load_pct, self.lut_eff_mot), 0.05)
            eff_mot_cone_bot = max(np.interp(min(p_mech_cone_bot / MAX_MOT_W, 1.2), self.lut_load_pct, self.lut_eff_mot), 0.05)

            p_motor_tube_top_kw = (p_mech_tube_top / eff_mot_tube_top) / 1000.0
            p_motor_tube_bot_kw = (p_mech_tube_bot / eff_mot_tube_bot) / 1000.0
            p_motor_cone_top_kw = (p_mech_cone_top / eff_mot_cone_top) / 1000.0
            p_motor_cone_bot_kw = (p_mech_cone_bot / eff_mot_cone_bot) / 1000.0

            gen_val = self.val_gen_load / 100.0
            omega_top = self.rotor_rpm_top * 0.1047
            omega_bot = self.rotor_rpm_bot * 0.1047

            if gen_val >= 0:
                brake_top = MAX_BRAKE_TORQUE * gen_val * np.sign(omega_top) if abs(omega_top) > 0.05 else 0.0
                brake_bot = MAX_BRAKE_TORQUE * gen_val * np.sign(omega_bot) if abs(omega_bot) > 0.05 else 0.0
                motor_assist_top, motor_assist_bot = 0.0, 0.0
                
                P_gen_mech_top = abs(brake_top * omega_top)
                P_gen_mech_bot = abs(brake_bot * omega_bot)
                
                eff_gen_top = np.interp(min(P_gen_mech_top / MAX_GEN_W, 1.2), self.lut_load_pct, self.lut_eff_gen)
                eff_gen_bot = np.interp(min(P_gen_mech_bot / MAX_GEN_W, 1.2), self.lut_load_pct, self.lut_eff_gen)
                
                gen_power_top_kw = (P_gen_mech_top * eff_gen_top) / 1000.0
                gen_power_bot_kw = (P_gen_mech_bot * eff_gen_bot) / 1000.0
            else:
                brake_top, brake_bot = 0.0, 0.0
                motor_assist_top = MAX_BRAKE_TORQUE * abs(gen_val)
                motor_assist_bot = MAX_BRAKE_TORQUE * abs(gen_val)
                
                P_mot_mech_top = motor_assist_top * max(abs(omega_top), 2.0)
                P_mot_mech_bot = motor_assist_bot * max(abs(omega_bot), 2.0)
                
                eff_gen_mot_top = max(np.interp(min(P_mot_mech_top / MAX_GEN_W, 1.2), self.lut_load_pct, self.lut_eff_mot), 0.05)
                eff_gen_mot_bot = max(np.interp(min(P_mot_mech_bot / MAX_GEN_W, 1.2), self.lut_load_pct, self.lut_eff_mot), 0.05)
                
                gen_power_top_kw = -(P_mot_mech_top / eff_gen_mot_top) / 1000.0
                gen_power_bot_kw = -(P_mot_mech_bot / eff_gen_mot_bot) / 1000.0

            friction_top = 50.0 * omega_top * abs(omega_top) 
            friction_bot = 50.0 * omega_bot * abs(omega_bot)
            
            net_torque_top = torque_top + motor_assist_top - brake_top - friction_top
            net_torque_bot = torque_bot + motor_assist_bot - brake_bot - friction_bot
            
            omega_top += np.clip(np.nan_to_num(net_torque_top / INERTIA_ROTOR), -50.0, 50.0) * dt
            omega_bot += np.clip(np.nan_to_num(net_torque_bot / INERTIA_ROTOR), -50.0, 50.0) * dt
            
            if abs(omega_top) < 0.02 and abs(net_torque_top) < (MAX_BRAKE_TORQUE * gen_val + 10): omega_top = 0.0
            if abs(omega_bot) < 0.02 and abs(net_torque_bot) < (MAX_BRAKE_TORQUE * gen_val + 10): omega_bot = 0.0

            self.rotor_rpm_top = max(0, omega_top / 0.1047)
            self.rotor_rpm_bot = max(0, omega_bot / 0.1047)

            effective_wind_speed = max(v_app_mag_global, 0.1) 
            power_wind_kinetic = 0.5 * rho * 254.0 * (effective_wind_speed**3)
            betz_limit_kw = (power_wind_kinetic * 0.593) / 1000.0
            
            total_gen = gen_power_top_kw + gen_power_bot_kw
            if total_gen > betz_limit_kw:
                ratio = betz_limit_kw / max(0.001, total_gen)
                gen_power_top_kw *= ratio
                gen_power_bot_kw *= ratio
            
            p_mech_vtol_top = thrust_top * 12.0 if gen_val < 0 else 0.0
            p_mech_vtol_bot = thrust_bot * 12.0 if gen_val < 0 else 0.0
            
            motor_vtol_top_kw = (p_mech_vtol_top / eff_mot_cone_top) / 1000.0
            motor_vtol_bot_kw = (p_mech_vtol_bot / eff_mot_cone_bot) / 1000.0
            
            total_motor_load = p_motor_cone_top_kw + p_motor_cone_bot_kw + p_motor_tube_top_kw + p_motor_tube_bot_kw + motor_vtol_top_kw + motor_vtol_bot_kw
            
            net_power = (gen_power_top_kw + gen_power_bot_kw) - total_motor_load
            
            pure_lift_force = Sys_Net_Force[1] + (self.kin.mass * 9.81)
            
            self.telemetry.update({
                'gen_top_kw': gen_power_top_kw, 'gen_bot_kw': gen_power_bot_kw,
                'motor_cone_top_kw': p_motor_cone_top_kw + motor_vtol_top_kw, 'motor_cone_bot_kw': p_motor_cone_bot_kw + motor_vtol_bot_kw,
                'motor_tube_top_kw': p_motor_tube_top_kw, 'motor_tube_bot_kw': p_motor_tube_bot_kw,
                'net_power_kw': net_power, 'lift_total_kg': pure_lift_force / 9.81, 
                'betz_limit_kw': betz_limit_kw
            })
            self.rotor_angle_top += self.rotor_rpm_top * 0.1047 * 0.2
            self.rotor_angle_bot += self.rotor_rpm_bot * 0.1047 * 0.2
            self.spoke_spin_phase_top += cone_rpm_top * 0.1047 * 0.2
            self.spoke_spin_phase_bot += cone_rpm_bot * 0.1047 * 0.2

        else:
            for k in ['gen_top_kw', 'gen_bot_kw', 'motor_cone_top_kw', 'motor_cone_bot_kw', 'motor_tube_top_kw', 'motor_tube_bot_kw', 'net_power_kw', 'lift_total_kg', 'betz_limit_kw']:
                self.telemetry[k] = 0.0

        self.update_hud()
        
        p_TL, p_BR = math_pts_cyl(geo['TL'], geo['BR'], 0.25, 0.25); self.frame_parts['Front'].mesh.points = p_TL
        p_TR, p_BL = math_pts_cyl(geo['TR'], geo['BL'], 0.25, 0.25); self.frame_parts['Back'].mesh.points = p_TR
        pivot_off = Rx.dot([0,0,0.3])
        self.frame_parts['Pivot'].update_transform(pos_offset - pivot_off, pos_offset + pivot_off)
        
        pod_off = Rx.dot([0,0,0.5])
        self.pod_part.update_transform(pos_offset - pod_off, pos_offset + pod_off)
        
        if self.para_inflation > 0.01:
            self.para_canopy.set_visibility(True)
            for r in self.para_ropes: r.set_visibility(True)
            
            para_center = pos_offset + np.array([0.0, 40.0, 0.0]) 
            
            m_para = np.eye(4)
            scale = max(0.1, self.para_inflation)
            m_para[0,0] = scale; m_para[1,1] = scale; m_para[2,2] = scale
            m_para[0:3, 3] = para_center
            self.para_canopy.set_matrix(m_para)
            
            R = 11.0 * scale; H = -4.0 * scale
            p_bases = [np.array([R, H, 0]), np.array([-R, H, 0]), np.array([0, H, R]), np.array([0, H, -R])]
            for i, rope in enumerate(self.para_ropes):
                rope.mesh.points, _ = math_pts_cyl(pos_offset, para_center + p_bases[i], 0.05, 0.05)
        else:
            self.para_canopy.set_visibility(False)
            for r in self.para_ropes: r.set_visibility(False)
        
        tube_coords = {'Top': (geo['p0_T'], geo['p1_T']), 'Bot': (geo['p0_B'], geo['p1_B'])}
        for idx, bt in enumerate(self.blue_tubes):
            pos_key = bt['pos']; p0, p1 = tube_coords[pos_key]
            pts_b, _ = math_pts_bellows(p0, p1, 1.1, 1.1, self.fold_factor, 24, 0.0); bt['bellows'].mesh.points = pts_b
            
            phase = self.tube_spin_phase_top if pos_key == 'Top' else self.tube_spin_phase_bot
            for rib in bt['ribs']:
                pts_s, _ = math_pts_strip(p0, p1, 1.1, 1.1, rib['base'] + phase, 0.0); rib['part'].mesh.points = pts_s
                
            v_ep = Rx.dot([0.05, 0, 0])
            bt['ep1'].update_transform(p0, p0+v_ep); bt['ep2'].update_transform(p1, p1+v_ep)
            bt['kn1'].update_transform(p0-v_ep*2, p0+v_ep*2); bt['kn2'].update_transform(p1-v_ep*2, p1+v_ep*2)
            
            tube_len = geo['L']; center = (p0 + p1) / 2.0; af_data = self.tube_air_parts[idx]
            tube_y_local = geo['H'] + 2.5 if pos_key == 'Top' else -geo['H'] - 2.5
            
            local_lift_dir = lift_dir_top if pos_key == 'Top' else lift_dir_bot
            
            for i, arr_set in enumerate(af_data['arrows']):
                offset_x = (i - 2) * (tube_len / 6.0)
                base_pt = pos_offset + Rx.dot([offset_x, tube_y_local, 0.0])
                
                v_app_perp_vis = wind_vec_global - np.dot(wind_vec_global, tube_axis) * tube_axis
                v_app_dir_vis = v_app_perp_vis / max(np.linalg.norm(v_app_perp_vis), 0.001)
                wind_path = v_app_dir_vis * 4.0
                
                pt_fast = base_pt + local_lift_dir * 1.5
                pt_slow = base_pt - local_lift_dir * 1.5
                
                v_tube_app_mag_vis = np.linalg.norm(v_app_perp_vis)
                scale_fast = v_tube_app_mag_vis * 0.2; scale_slow = v_tube_app_mag_vis * 0.2
                col_fast = 'lime'; col_slow = 'lime'
                
                if self.spinning:
                    rpm_val = tube_rpm_top if pos_key == 'Top' else tube_rpm_bot
                    eff = abs(rpm_val) * 0.01
                    scale_fast += eff; scale_slow = max(0.1, scale_slow - eff)
                    col_fast = 'red'; col_slow = 'blue'
                        
                arr_set['top'].update_transform(pt_fast - wind_path, pt_fast + wind_path, scale_z=scale_fast)
                arr_set['top'].set_color(col_fast); arr_set['top'].set_visibility(self.show_tube_air)
                
                arr_set['bot'].update_transform(pt_slow - wind_path, pt_slow + wind_path, scale_z=scale_slow)
                arr_set['bot'].set_color(col_slow); arr_set['bot'].set_visibility(self.show_tube_air)
                
            force_data = self.tube_force_parts[idx]
            vis_len = 0.0 
            if self.spinning: 
                tube_lift_force = F_tubes_global_top if pos_key == 'Top' else F_tubes_global_bot
                vis_len = np.linalg.norm(tube_lift_force) / (9.81 * 1000.0)
                
            lift_vec_global_vis = local_lift_dir * vis_len
            force_data['arrow'].update_transform(center, center + lift_vec_global_vis)
            force_data['arrow'].set_visibility(self.show_tube_force and self.spinning)
            
        node_map = {'TL': geo['TL'], 'TR': geo['TR'], 'BL': geo['BL'], 'BR': geo['BR']}
        tube_pos_map = {'TL': geo['p0_T'], 'TR': geo['p1_T'], 'BR': geo['p1_B'], 'BL': geo['p0_B']}
        
        for item in self.strut_parts:
            tag = item['tag']; F_Pos = node_map[tag]; T_Pos = tube_pos_map[tag]
            j_off = Rx.dot([0,0,0.2])
            item['joint'].mesh.points, _ = math_pts_cyl(F_Pos-j_off, F_Pos+j_off, 0.5, 0.5)
            item['strut'].mesh.points, _ = math_pts_cyl(F_Pos, T_Pos, 0.15, 0.15)
            
            P_motor = F_Pos + 0.8 * (T_Pos - F_Pos)
            mb_off = Rx.dot([0.3,0,0])
            item['body'].update_transform(P_motor-mb_off, P_motor+mb_off)
            item['shaft'].update_transform(P_motor, T_Pos)
            mw_off = Rx.dot([0.1,0,0])
            item['wheel'].update_transform(T_Pos-mw_off, T_Pos+mw_off)
            
        for item in self.rotors_matrix_parts:
            tag = item['tag']; hc = hc_map[tag]
            is_top = (tag in ['TL', 'TR'])
            angle = (self.rotor_angle_top if is_top else self.rotor_angle_bot) * item['spin_dir']
            c_ang = np.cos(angle); s_ang = np.sin(angle)
            Rz = np.array([[c_ang, -s_ang, 0], [s_ang, c_ang, 0], [0, 0, 1]])
            R_comb = Rx.dot(Rz)
            m = np.eye(4); m[0:3, 0:3] = R_comb; m[0:3, 3] = hc + pos_offset
            for part in item['parts']: part.set_matrix(m)
            
            cx = -geo['W']/2 if 'L' in tag else geo['W']/2
            cy = geo['H']/2 if 'T' in tag else -geo['H']/2
            vx, vy = -cx, -cy
            mag = np.sqrt(vx**2 + vy**2); vx/=mag; vy/=mag
            local_g = np.array([cx + vx*4.2, cy + vy*4.2, 0])
            base_z = self.Z_F if tag in ['TL', 'BR'] else self.Z_B
            item['gen'].update_transform(Rx.dot([local_g[0], local_g[1], base_z+0.5]) + pos_offset, Rx.dot([local_g[0], local_g[1], base_z-0.5]) + pos_offset)
            item['clamp'].update_transform(Rx.dot([local_g[0], local_g[1], base_z+0.2]) + pos_offset, Rx.dot([local_g[0], local_g[1], base_z-0.2]) + pos_offset)
            
        w_pos = geo['Winch']
        w_off = np.array([0.5, 0, 0])
        self.winch_part.update_transform(w_pos - w_off, w_pos + w_off)
        
        self.ropes['TL'].mesh.points, _ = math_pts_cyl(geo['TL'], w_pos, 0.04, 0.04)
        self.ropes['TR'].mesh.points, _ = math_pts_cyl(geo['TR'], w_pos, 0.04, 0.04)
        self.ropes['BL'].mesh.points, _ = math_pts_cyl(geo['BL'], w_pos, 0.04, 0.04)
        self.ropes['BR'].mesh.points, _ = math_pts_cyl(geo['BR'], w_pos, 0.04, 0.04)
        self.ropes['Center'].mesh.points, _ = math_pts_cyl(pos_offset, w_pos, 0.04, 0.04) 
        self.ropes['Main'].mesh.points, _ = math_pts_cyl(w_pos, geo['Anchor'], 0.08, 0.08)
        self.ropes['SideL'].mesh.points, _ = math_pts_cyl(geo['p0_T'], geo['p0_B'], 0.03, 0.03)
        self.ropes['SideR'].mesh.points, _ = math_pts_cyl(geo['p1_T'], geo['p1_B'], 0.03, 0.03)

    def trigger_failure_mode(self):
        if self.structural_failure: return 
        self.structural_failure = True
        self.spinning = False 
        self.p.set_background('mistyrose') 
        self.set_labels_color('white') 
        fail_color = 'red'
        for part in self.frame_parts.values(): part.set_color(fail_color)
        for bt in self.blue_tubes: bt['bellows'].set_color(fail_color)
        for sp in self.dynamic_spokes: sp['spoke'].set_color(fail_color); sp['tape'].set_color(fail_color)
        for rope in self.ropes.values(): rope.set_color(fail_color)
        if self.winch_part: self.winch_part.set_color(fail_color)
        self.update_hud()

    def toggle_spin(self, state): 
        if self.structural_failure: return
        self.spinning = state
    def toggle_auto_pilot(self, state): self.auto_pilot = state
    def toggle_fold(self, state): self.folding = state
    def toggle_tube_air(self, state): self.show_tube_air = state; self.update_geometry()
    def toggle_tube_force(self, state): self.show_tube_force = state; self.update_geometry()
    def toggle_cone_air_res(self, state): self.show_cone_air_res = state; self.update_geometry()
    def toggle_cone_air_comp(self, state): self.show_cone_air_comp = state; self.update_geometry()
    def toggle_cone_force_res(self, state): self.show_cone_force_res = state; self.update_geometry()
    def toggle_cone_force_comp(self, state): self.show_cone_force_comp = state; self.update_geometry()

    def run(self):
        self.p.show(interactive_update=True, auto_close=False, full_screen=True)
        while True:
            try:
                if not hasattr(self.p, 'render_window') or self.p.render_window is None: break
                needs_update = False
                if self.folding:
                    self.fold_factor += 0.01 * self.fold_direction
                    if self.fold_factor >= 0.95: self.fold_factor = 0.95; self.fold_direction = -1
                    elif self.fold_factor <= 0.0: self.fold_factor = 0.0; self.fold_direction = 1
                    needs_update = True
                if self.spinning or self.was_spinning: needs_update = True
                self.was_spinning = self.spinning
                if needs_update or True: self.update_geometry()
                self.p.update(); time.sleep(0.04) 
            except Exception as e:
                print(f"\n[CRASH DETECTED] Το σύστημα κατέρρευσε: {e}\n")
                break
        try: self.p.close()
        except: pass

if __name__ == "__main__":
    app = QuadMagnusApp()
    app.run()