import pyvista as pv
import numpy as np
import time
import sys

# --- 0. CONFIG ---
pv.global_theme.allow_empty_mesh = True
pv.global_theme.font.color = 'black'

# --- 1. MATH ENGINE ---
def get_align_matrix(p0, p1, scale_x=1.0, scale_y=1.0, scale_z=1.0):
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
        self.p = pv.Plotter(title="Quad-Magnus: FULL FLIGHT SIMULATOR", window_size=(1600, 1000))
        self.p.set_background('white')
        
        # --- SAFEGUARD ΓΙΑ GHOST LOOPS (VTK-correct kill) ---
        def _on_close(*args):
            try:
                self.p.iren.TerminateApp()
            except:
                pass
            sys.exit(0)
            
        if hasattr(self.p, 'iren') and self.p.iren is not None:
            self.p.iren.add_observer("ExitEvent", _on_close)
            self.p.iren.add_observer("WindowCloseEvent", _on_close)
            
        self.spinning = False
        self.was_spinning = False 
        self.folding = False
        self.structural_failure = False 
        self.auto_pilot = False 
        self.governor_status = "IDLE"
        
        self.ground_level = -400.0
        self.pos_y = 0.0  
        self.vel_y = 0.0
        
        self.val_wind = 10.0         
        self.val_pitch = 0.0         
        self.val_spin_drive = 3500.0 
        self.val_spin_blue = 180.0   
        self.val_gen_load = 18.0     
        self.val_target_payload = 1500.0 
        
        self.rotor_rpm_top = 0.0
        self.rotor_rpm_bot = 0.0
        self.current_total_drag = 0.0 
        self.drag_limit = 150000.0 
        
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
        
        self.sl_pitch = None
        self.sl_spin_lift = None
        self.sl_spin_drive = None
        self.sl_gen_load = None
        self.sl_wind = None
        
        self.btn_spin = None
        self.btn_fold = None
        self.btn_ap = None
        self.btn_reset = None
        self.btn_tube_air = None
        self.btn_tube_force = None
        self.btn_cone_air_res = None
        self.btn_cone_force_res = None
        self.btn_cone_air_comp = None
        self.btn_cone_force_comp = None
        
        self.setup_ui()     
        self.setup_hud()    
        self.build_scene()  
        
        self.p.camera.position = (250, -200, -500)
        self.p.camera.focal_point = (0, -200, -200)
        self.p.camera.up = (0, 1, 0)
        self.p.camera.zoom(1.1)

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
        self.sl_pitch = self.p.add_slider_widget(self.set_pitch, [-90, 90], title="Pitch Angle (deg)", value=0.0, pointa=(0.03, 0.82), pointb=(0.20, 0.82), style='modern')
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
                status_msg = "ACTIVE DEPOWER & BRAKING"
                status_col = "orange"
            elif "GLIDE" in self.governor_status:
                status_msg = "AERODYNAMIC BRAKE (GLIDING)"
                status_col = "purple"
            elif "PARACHUTE" in self.governor_status:
                status_msg = "EMERGENCY FLARE (AIRBRAKE DEPLOYED)"
                status_col = "purple"
            elif "VTOL" in self.governor_status:
                status_msg = "DRONE MODE (FINAL APPROACH)"
                status_col = "teal"
            elif "LANDED" in self.governor_status:
                status_msg = "SAFE ON OCEAN (IDLE)"
                status_col = "blue"
            elif "RECOVERING" in self.governor_status or "HOMEOSTASIS" in self.governor_status:
                status_msg = "ALTITUDE RECOVERY"
                status_col = "blue"
                
        elif stress_pct > 80: 
            status_msg = "WARNING: HIGH STRESS"
            status_col = "orange"
            
        if self.structural_failure: 
            status_msg = "*** CRITICAL FAILURE ***"
            status_col = "red"
            
        tether_state = t.get('tether_state', 'PARKED (SETUP)')
        
        text_block_1 = (
            f"QUAD-MAGNUS: KINEMATICS & GRAVITY ENGINE\n"
            f"============================================\n"
            f"[SYSTEM STATUS]\n"
            f" MODE       : {ap_status}\n"
            f" STATE      : {status_msg}\n"
            f" STRESS(T)  : {drag_curr:0.0f} / {self.drag_limit:0.0f} N\n"
            f" LOAD       : {stress_bar_visual} {stress_pct:.1f}%\n"
            f"\n"
            f"[KINEMATICS (FLIGHT DATA)]\n"
            f" Altitude   : {self.pos_y:.1f} m (0 is max height)\n"
            f" Vert. Vel. : {self.vel_y:.2f} m/s\n"
            f" Tether     : {tether_state}\n"
            f" Pitch Angle: {self.val_pitch:0.1f} deg\n"
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
            f"[POWER ANALYSIS]\n"
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
    def set_target_payload(self, val): self.val_target_payload = val

    def calculate_geometry(self):
        f = self.fold_factor
        curr_W = self.Max_W * (1 - f) + self.Min_W * f
        curr_H = np.sqrt(self.Beam_Len**2 - curr_W**2)
        curr_L = self.Max_Tube_L * (1 - f) + self.Min_Tube_L * f
        
        pos_z = self.pos_y * 1.25 
        pos_offset = np.array([0.0, self.pos_y, pos_z])
        
        pitch_rad = np.radians(-self.val_pitch)
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
                self.dynamic_spokes.append({'tag': tag, 'base_angle': np.radians(deg), 'spin_dir': spin_dir, 'spoke': spoke, 'tape': tape, 'id': spoke_idx})
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
            self.blue_tubes.append({'pos': pos, 'bellows': bellows, 'ribs': ribs, 'ep1': ep1, 'ep2': ep2, 'kn1': kn1, 'kn2': kn2})
            air_arrows = []
            for i in range(5):
                arr_top = ScenePart(self.p, create_arrow_template(), 'lime'); arr_top.set_visibility(False)
                arr_bot = ScenePart(self.p, create_arrow_template(), 'lime'); arr_bot.set_visibility(False)
                air_arrows.append({'top': arr_top, 'bot': arr_bot, 'idx': i})
            self.tube_air_parts.append({'pos': pos, 'arrows': air_arrows})
            lift_arrow = ScenePart(self.p, create_arrow_template(scale=3.0), 'red'); lift_arrow.set_visibility(False)
            self.tube_force_parts.append({'pos': pos, 'arrow': lift_arrow})

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
            return
            
        winch_mass = 200.0
        base_weight = self.val_target_payload + winch_mass
        SAFE_LIMIT = 120000.0        
        CRITICAL_LIMIT = 145000.0    
        predictive_stress = max(self.current_total_drag, self.telemetry['diag_drag_total'])
        
        # --- SPACEX STYLE STATE MACHINE (400m) ---
        if self.pos_y <= (self.ground_level + 0.5) and abs(self.vel_y) < 1.0:
            if self.val_wind < 4.0:
                self.governor_status = "LANDED SAFE (WAITING FOR WIND)"
                self.val_pitch = 90.0
                self.val_spin_drive = 0.0
                self.val_spin_blue = 0.0
                self.val_gen_load = 0.0
            else:
                self.governor_status = "VTOL TAKEOFF (LIFTING)"
                self.val_pitch = 90.0
                self.val_gen_load = -80.0 
                self.val_spin_drive = 4500.0 
                self.val_spin_blue = 100.0
            self.update_sliders_ghost()
            return
            
        if self.val_wind < 4.0 and self.pos_y > (self.ground_level + 0.5):
            if self.pos_y > -150.0:
                self.governor_status = "GLIDE FREEFALL (AERODYNAMIC BRAKE)"
                if self.val_pitch < 85.0: self.val_pitch += 2.0
                self.val_gen_load = 0.0 
                self.val_spin_drive = 0.0 
            elif self.pos_y > -320.0:
                self.governor_status = "PARACHUTE DEPLOYED (NO MOTORS YET)"
                if self.val_pitch < 85.0: self.val_pitch += 2.0
                self.val_gen_load = 0.0 
                self.val_spin_drive = 0.0 
                if self.val_spin_blue < 300.0: self.val_spin_blue += 5.0 
            elif self.pos_y > -370.0:
                self.governor_status = "PARACHUTE + VTOL SPOOL-UP PREP"
                self.val_gen_load = -60.0 
                self.val_spin_drive = 4500.0 
            else:
                self.governor_status = "VTOL FINAL DESCENT (PROPELLERS)"
                if self.val_pitch < 90.0: self.val_pitch += 2.0
                self.val_spin_drive = 4500.0 
                
                if self.vel_y < -2.0: 
                    self.val_gen_load -= 10.0 
                elif self.vel_y > -0.5:
                    self.val_gen_load += 3.0 
                    
                self.val_gen_load = np.clip(self.val_gen_load, -100.0, -5.0)
                
            self.update_sliders_ghost()
            return
            
        if predictive_stress > CRITICAL_LIMIT:
            self.val_spin_drive = 0.0
            self.val_pitch = 30.0 
            self.val_gen_load = 100.0 
            self.governor_status = "EMERGENCY DEFENSE (PREDICTIVE)"
            
        elif predictive_stress > SAFE_LIMIT:
            excess_stress = predictive_stress - SAFE_LIMIT
            self.val_spin_drive -= excess_stress * 0.005 
            if self.val_spin_drive < 0: self.val_spin_drive = 0
            
            self.val_pitch += 0.5 
            if self.val_pitch > 30.0: self.val_pitch = 30.0
            
            self.val_gen_load += 2.0
            if self.val_gen_load > 100.0: self.val_gen_load = 100.0
            
            self.governor_status = "ACTIVE DEFENSE (PITCH+BRAKES)"
            
        else:
            if self.val_spin_drive < 3500.0: self.val_spin_drive += 10.0 
            if self.val_pitch > 0.0:
                self.val_pitch -= 0.5
                if self.val_pitch < 0.0: self.val_pitch = 0.0
                
            if self.val_gen_load > 18.0:
                self.val_gen_load -= 0.5
            elif self.val_gen_load < 18.0:
                self.val_gen_load += 0.5
            
            if self.pos_y < -0.5:
                target_lift = base_weight + (-self.pos_y * 10.0) - (self.vel_y * 250.0)
                self.governor_status = "ALTITUDE HOMEOSTASIS"
            else:
                target_lift = base_weight
                self.governor_status = "OPTIMIZING"
                
            lift_error = target_lift - self.telemetry['lift_total_kg']
            self.val_spin_blue += lift_error * 0.015 
            self.val_spin_blue = np.clip(self.val_spin_blue, 0.0, 300.0)
            
        self.update_sliders_ghost()

    def trigger_reset(self, state):
        if not state: return 
        
        self.structural_failure = False
        self.spinning = False
        self.auto_pilot = False
        self.pos_y = 0.0 
        self.vel_y = 0.0
        self.rotor_rpm_top = 0.0
        self.rotor_rpm_bot = 0.0
        self.current_total_drag = 0.0
        
        self.val_pitch = 0.0
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
        self.run_auto_pilot_logic()

        geo = self.calculate_geometry()
        Rx = geo['Rx']
        pos_offset = geo['pos_offset']
        rho = 1.225
        
        self.p.camera.focal_point = pos_offset.tolist()
        
        if self.val_gen_load < 0:
            tube_rpm_top = self.val_spin_blue
            tube_rpm_bot = self.val_spin_blue
            cone_rpm_top = self.val_spin_drive
            cone_rpm_bot = self.val_spin_drive
        else:
            if self.val_pitch >= 0:
                tube_rpm_bot = self.val_spin_blue
                tube_rpm_top = max(0, self.val_spin_blue - self.val_pitch * 5.0)
                cone_rpm_bot = self.val_spin_drive
                cone_rpm_top = max(0, self.val_spin_drive - self.val_pitch * 80.0)
            else:
                tube_rpm_top = self.val_spin_blue
                tube_rpm_bot = max(0, self.val_spin_blue + self.val_pitch * 5.0) 
                cone_rpm_top = self.val_spin_drive
                cone_rpm_bot = max(0, self.val_spin_drive + self.val_pitch * 80.0)
        
        if self.spinning:
            self.telemetry['tube_rpm_top'] = tube_rpm_top
            self.telemetry['tube_rpm_bot'] = tube_rpm_bot
            self.telemetry['cone_rpm_top'] = cone_rpm_top
            self.telemetry['cone_rpm_bot'] = cone_rpm_bot
            self.tube_spin_phase_top -= tube_rpm_top * 0.1047 * 0.2
            self.tube_spin_phase_bot -= tube_rpm_bot * 0.1047 * 0.2
        else:
            self.telemetry.update({'tube_rpm_top': 0.0, 'tube_rpm_bot': 0.0, 'cone_rpm_top': 0.0, 'cone_rpm_bot': 0.0})

        pitch_rad = np.radians(-self.val_pitch)
        
        vel_z = self.vel_y * 1.25 
        v_wind_eff_z = self.val_wind * 0.85 - vel_z 
        updraft_v_y = -self.vel_y if self.vel_y < 0 else 0.0
        
        wind_vec_global = np.array([0.0, updraft_v_y, v_wind_eff_z])
        v_app_mag_global = np.linalg.norm(wind_vec_global)
        low_wind_factor = min(1.0, (self.val_wind / 5.0)**2)
        
        F_total = np.array([0.0, 0.0, 0.0])

        tube_r = 1.1; tube_len = geo['L']
        v_surf_top = tube_rpm_top * 0.1047 * tube_r
        v_surf_bot = tube_rpm_bot * 0.1047 * tube_r
        tube_area = tube_len * (tube_r * 2.0)
        v_wind_local_tubes = v_wind_eff_z * np.cos(pitch_rad)
        
        tube_drag_local = 0.5 * rho * tube_area * (v_wind_local_tubes**2) * 1.2 * 2.0 
        tube_lift_top_local = (rho * tube_len * v_wind_local_tubes * v_surf_top * 2.0) * low_wind_factor if self.spinning else 0.0
        tube_lift_bot_local = (rho * tube_len * v_wind_local_tubes * v_surf_bot * 2.0) * low_wind_factor if self.spinning else 0.0
        
        F_tubes_local = np.array([0.0, tube_lift_top_local + tube_lift_bot_local, tube_drag_local])
        F_tubes_global = Rx.dot(F_tubes_local)
        F_total += F_tubes_global
        
        p_motor_tube_top_kw = (tube_rpm_top / 100.0)**2 * 0.5 if self.spinning else 0.0
        p_motor_tube_bot_kw = (tube_rpm_bot / 100.0)**2 * 0.5 if self.spinning else 0.0

        frame_n = Rx.dot(np.array([0.0, 0.0, 1.0])) 
        if frame_n[2] < 0 and self.val_wind > 0: frame_n *= -1 
        v_n_z = v_wind_eff_z * frame_n[2]
        drag_force_frame_z = 0.5 * rho * 15.0 * (v_n_z**2) * 1.5 * np.sign(v_n_z)
        F_frame = frame_n * drag_force_frame_z 
        F_total += F_frame 

        parachute_drag_y = 0.0
        is_para_deployed = "PARACHUTE DEPLOYED" in self.governor_status or "SPOOL-UP" in self.governor_status
        if is_para_deployed:
            para_area = 450.0 
            parachute_drag_y = 0.5 * rho * para_area * 1.5 * (updraft_v_y**2)
            F_total += np.array([0.0, parachute_drag_y, 0.0])

        thrust_mag = 0.0
        if self.val_gen_load < 0 and self.spinning and not is_para_deployed:
            rpm_sq = cone_rpm_top**2 + cone_rpm_bot**2
            thrust_mag = 0.0005 * rpm_sq * (abs(self.val_gen_load) / 100.0) 
            
        F_prop_vtol = frame_n * thrust_mag
        F_total += F_prop_vtol
        self.telemetry['diag_thrust_vtol'] = F_prop_vtol[1]

        torque_top = 0.0; torque_bot = 0.0
        total_cone_z_force = F_prop_vtol[2] 
        cone_radius = 0.15; cone_length = 4.0; rotor_R_full = 4.0; rotor_R_eff = rotor_R_full * 0.7 
        hc_map = {'TL': geo['hc_TL'] - pos_offset, 'BR': geo['hc_BR'] - pos_offset, 'TR': geo['hc_TR'] - pos_offset, 'BL': geo['hc_BL'] - pos_offset}

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
            
            if self.spinning:
                v_tan_mag = rotor_omega * rotor_R_eff
                headwind_vec = -tan_vec * v_tan_mag
                apparent_vec = wind_vec_global + headwind_vec
                v_app_mag = np.linalg.norm(apparent_vec)
                spin_axis_vec = rad_norm * (1.0 if item['spin_dir'] == 1 else -1.0)
                v_surface = (drive_rpm * 0.1047) * cone_radius
                
                safe_v_app = max(v_app_mag, 0.1)
                spin_ratio = min(v_surface / safe_v_app, 6.0) 
                
                eff_factor = spin_ratio**3 if spin_ratio < 1.0 else (1.0 + (spin_ratio - 4.0)*0.05 if spin_ratio > 4.0 else 1.0)
                lift_mag_rotor = (rho * cone_length * v_app_mag * v_surface * 2.0) * eff_factor * low_wind_factor
                
                lift_vec_visual = np.cross(apparent_vec, spin_axis_vec)
                if np.linalg.norm(lift_vec_visual) > 0.001: lift_vec_visual /= np.linalg.norm(lift_vec_visual)
                lift_force = lift_vec_visual * lift_mag_rotor
                
                cd_total = 0.8 + (spin_ratio ** 2) * 0.08
                drag_mag = 0.5 * rho * (cone_length*0.3) * (v_app_mag**2) * cd_total
                
                drag_dir = apparent_vec / max(v_app_mag, 0.001)
                drag_vec = drag_dir * drag_mag * low_wind_factor

                total_force = lift_force + drag_vec
                F_total += total_force 
                total_cone_z_force += total_force[2]
                
                f_drive_mag = np.dot(total_force, tan_vec)
                if is_top: torque_top += f_drive_mag * rotor_R_eff 
                else:      torque_bot += f_drive_mag * rotor_R_eff 
                
                force_drive_vec = tan_vec * f_drive_mag
                force_load_vec = total_force - force_drive_vec
                scale_A = 1.0 + (drive_rpm * 0.005)
                scale_B = max(0.1, 1.0 - (drive_rpm * 0.005))
            else: 
                headwind_vec = np.array([0,0,0], dtype=float); apparent_vec = wind_vec_global
                lift_vec_visual = np.array([0,0,0], dtype=float); total_force = np.array([0,0,0], dtype=float)
                force_drive_vec = np.array([0,0,0], dtype=float); force_load_vec = np.array([0,0,0], dtype=float)
                scale_A = 1.0; scale_B = 1.0
                
                safe_v_wind = max(v_wind_eff_z, 0.001)
                drag_static = (np.array([0,0,1])) * (0.5 * rho * (cone_length*0.3) * (v_wind_eff_z**2))
                F_total += drag_static
                total_cone_z_force += drag_static[2]

            offset_dist = 0.5 
            if np.linalg.norm(lift_vec_visual) > 0.001: lift_dir_norm = lift_vec_visual
            else: lift_dir_norm = np.cross(wind_vec_global, rad_norm); lift_dir_norm /= (np.linalg.norm(lift_dir_norm)+0.001)

            pos_A = mid_cone + lift_dir_norm * offset_dist - apparent_vec * 0.5
            pos_B = mid_cone - lift_dir_norm * offset_dist - apparent_vec * 0.5
            vecs['air_res_A'].update_transform(pos_A, pos_A + apparent_vec * scale_A)
            vecs['air_res_A'].set_color('red' if self.spinning else 'orange'); vecs['air_res_A'].set_visibility(self.show_cone_air_res)
            vecs['air_res_B'].update_transform(pos_B, pos_B + apparent_vec * scale_B)
            vecs['air_res_B'].set_color('blue' if self.spinning else 'orange'); vecs['air_res_B'].set_visibility(self.show_cone_air_res)
            origin = mid_cone - apparent_vec * 0.8
            vecs['air_amb'].update_transform(origin, origin + wind_vec_global); vecs['air_amb'].set_visibility(self.show_cone_air_comp)
            vecs['air_rot'].update_transform(origin + wind_vec_global, origin + wind_vec_global + headwind_vec); vecs['air_rot'].set_visibility(self.show_cone_air_comp and self.spinning)
            v_scale = 0.02; origin_f = mid_cone
            vecs['force_res'].update_transform(origin_f, origin_f + total_force * v_scale); vecs['force_res'].set_visibility(self.show_cone_force_res and self.spinning)
            vecs['force_drive'].update_transform(origin_f, origin_f + force_drive_vec * v_scale); vecs['force_drive'].set_visibility(self.show_cone_force_comp and self.spinning)
            vecs['force_load'].update_transform(origin_f + force_drive_vec * v_scale, origin_f + (force_drive_vec + force_load_vec) * v_scale); vecs['force_load'].set_visibility(self.show_cone_force_comp and self.spinning)

        winch_mass = 200.0
        total_mass = self.val_target_payload + winch_mass
        total_weight_N = total_mass * 9.81
        F_y_net = F_total[1] - total_weight_N
        
        self.telemetry['diag_f_net_y'] = F_y_net
        self.telemetry['diag_drag_total'] = F_total[2]
        self.telemetry['diag_drag_tubes'] = F_tubes_global[2]
        self.telemetry['diag_drag_cones'] = total_cone_z_force
        self.telemetry['diag_drag_frame'] = F_frame[2] 
        self.telemetry['diag_parachute_drag'] = parachute_drag_y

        if self.spinning:
            dt = 0.04 
            accel_y = F_y_net / total_mass
            self.vel_y += accel_y * dt
            self.pos_y += self.vel_y * dt
            
            if self.pos_y >= 0.0:
                self.pos_y = 0.0
                self.vel_y = 0.0
                self.telemetry['tether_state'] = 'TAUT'
                tether_tension_N = np.sqrt(max(0, F_y_net)**2 + F_total[2]**2)
                
            elif self.pos_y <= self.ground_level:
                self.pos_y = self.ground_level
                if self.vel_y < -15.0: 
                    self.vel_y = 0.0
                    self.trigger_failure_mode()
                    tether_tension_N = 0.0
                else: 
                    self.vel_y = 0.0
                    self.telemetry['tether_state'] = 'LANDED SAFE'
                    tether_tension_N = F_total[2] * 0.1 
            else:
                self.telemetry['tether_state'] = 'SLACK (FALLING/HOVERING)'
                tether_tension_N = F_total[2] * 0.2
        else:
            self.vel_y = 0.0
            self.telemetry['tether_state'] = 'PARKED (IDLE)'
            tether_tension_N = F_total[2]

        self.current_total_drag = tether_tension_N
        if self.current_total_drag > self.drag_limit: 
            self.trigger_failure_mode()

        p_motor_cone_top_kw = 6 * (cone_rpm_top / 3000.0)**2 * 0.8 if self.spinning else 0.0
        p_motor_cone_bot_kw = 6 * (cone_rpm_bot / 3000.0)**2 * 0.8 if self.spinning else 0.0
        
        if self.spinning:
            gen_val = self.val_gen_load / 100.0
            inertia_factor = 1000.0 if gen_val < 0 else 5000.0
            
            if gen_val >= 0: 
                gen_resist_top = 2.0 * gen_val * (self.rotor_rpm_top * 1500.0) 
                motor_torque_top = 0.0
                gross_gen_watts_top = gen_resist_top * (self.rotor_rpm_top * 0.1047) * 0.90 
                gen_power_top_kw = gross_gen_watts_top / 1000.0
            else: 
                gen_resist_top = 0.0
                motor_torque_top = 2.0 * abs(gen_val) * 100000.0 
                gen_power_top_kw = 0.0
                
            drag_rotor_top = 2.0 * (self.rotor_rpm_top ** 2) * 5.0 
            net_torque_top = (torque_top + motor_torque_top - gen_resist_top - drag_rotor_top) * 0.90
            self.rotor_rpm_top = max(0, self.rotor_rpm_top + (net_torque_top / inertia_factor) * 0.1)

            if gen_val >= 0:
                gen_resist_bot = 2.0 * gen_val * (self.rotor_rpm_bot * 1500.0) 
                motor_torque_bot = 0.0
                gross_gen_watts_bot = gen_resist_bot * (self.rotor_rpm_bot * 0.1047) * 0.90 
                gen_power_bot_kw = gross_gen_watts_bot / 1000.0
            else:
                gen_resist_bot = 0.0
                motor_torque_bot = 2.0 * abs(gen_val) * 100000.0 
                gen_power_bot_kw = 0.0

            drag_rotor_bot = 2.0 * (self.rotor_rpm_bot ** 2) * 5.0 
            net_torque_bot = (torque_bot + motor_torque_bot - gen_resist_bot - drag_rotor_bot) * 0.90
            self.rotor_rpm_bot = max(0, self.rotor_rpm_bot + (net_torque_bot / inertia_factor) * 0.1)
            
            power_wind_kinetic = 0.5 * rho * 254.0 * (self.val_wind**3)
            betz_limit_kw = (power_wind_kinetic * 0.593) / 1000.0
            if (gen_power_top_kw + gen_power_bot_kw) > betz_limit_kw:
                ratio = betz_limit_kw / max(0.001, (gen_power_top_kw + gen_power_bot_kw))
                gen_power_top_kw *= ratio
                gen_power_bot_kw *= ratio
            
            motor_vtol_top_kw = p_motor_cone_top_kw if gen_val < 0 else 0.0
            motor_vtol_bot_kw = p_motor_cone_bot_kw if gen_val < 0 else 0.0
            
            total_motor_load = p_motor_cone_top_kw + p_motor_cone_bot_kw + p_motor_tube_top_kw + p_motor_tube_bot_kw + motor_vtol_top_kw + motor_vtol_bot_kw
            net_power = (gen_power_top_kw + gen_power_bot_kw) - total_motor_load
            
            self.telemetry.update({
                'gen_top_kw': gen_power_top_kw, 'gen_bot_kw': gen_power_bot_kw,
                'motor_cone_top_kw': p_motor_cone_top_kw + motor_vtol_top_kw, 'motor_cone_bot_kw': p_motor_cone_bot_kw + motor_vtol_bot_kw,
                'motor_tube_top_kw': p_motor_tube_top_kw, 'motor_tube_bot_kw': p_motor_tube_bot_kw,
                'net_power_kw': net_power, 'lift_total_kg': F_total[1] / 9.81, 
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
        
        if is_para_deployed:
            self.para_canopy.set_visibility(True)
            for r in self.para_ropes: r.set_visibility(True)
            para_center = pos_offset + np.array([0.0, 40.0, 0.0]) 
            m_para = np.eye(4); m_para[0:3, 3] = para_center
            self.para_canopy.set_matrix(m_para)
            
            R = 11.0; H = -4.0
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
            
            for i, arr_set in enumerate(af_data['arrows']):
                offset_x = (i - 2) * (tube_len / 6.0)
                pt_top_s = Rx.dot([offset_x, tube_y_local + 1.8, -8.0]) + pos_offset
                pt_top_e = Rx.dot([offset_x, tube_y_local + 1.8,  8.0]) + pos_offset
                pt_bot_s = Rx.dot([offset_x, tube_y_local - 1.8, -8.0]) + pos_offset
                pt_bot_e = Rx.dot([offset_x, tube_y_local - 1.8,  8.0]) + pos_offset
                
                scale_top = v_wind_eff_z * 0.2; scale_bot = v_wind_eff_z * 0.2; col_top = 'lime'; col_bot = 'lime'
                if self.spinning:
                    eff = (tube_rpm_top if pos_key == 'Top' else tube_rpm_bot) * 0.01
                    scale_top += eff; scale_bot = max(0.1, scale_bot - eff); col_top = 'red'; col_bot = 'blue'
                arr_set['top'].update_transform(pt_top_s, pt_top_e, scale_z=scale_top)
                arr_set['top'].set_color(col_top); arr_set['top'].set_visibility(self.show_tube_air)
                arr_set['bot'].update_transform(pt_bot_s, pt_bot_e, scale_z=scale_bot)
                arr_set['bot'].set_color(col_bot); arr_set['bot'].set_visibility(self.show_tube_air)
                
            force_data = self.tube_force_parts[idx]; vis_len = 0.0 
            if self.spinning: 
                tube_lift_kg = (tube_lift_top_local if pos_key == 'Top' else tube_lift_bot_local) / 9.81
                vis_len = tube_lift_kg / 1000.0
            lift_vec_global = Rx.dot(np.array([0, vis_len, 0]))
            force_data['arrow'].update_transform(center, center + lift_vec_global)
            force_data['arrow'].set_visibility(self.show_tube_force and self.spinning)
            
        node_map = {'TL': geo['TL'], 'BR': geo['BR'], 'TR': geo['TR'], 'BL': geo['BL']}
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
            except Exception: break
        try: self.p.close()
        except: pass

if __name__ == "__main__":
    app = QuadMagnusApp()
    app.run()