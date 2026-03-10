# Tethered-Quad-Magnus
Airborne Wind Turbine for steady flight (Tethered Fly-Gen AWE) with Quad-rotor layout and vertical takeoff/landing capability (VTOL).


[![License: CERN-OHL](https://img.shields.io/badge/License-CERN%20OHL-blue.svg)](https://ohwr.org/cernohl)
[![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)](https://www.python.org/)


## The Engineering Philosophy: Active Lift over Passive Glide

Traditional AWE systems (tethered kites or gliders) rely on passive airfoils that are highly susceptible to stall during turbulent gusts or sudden wind drops. QuadMagnus explores active aerodynamic control.

By utilizing the Magnus effect, the drone can theoretically generate lift coefficients ($C_L$) even at near-zero relative wind speeds.

### 1. General Architecture & Goals (System Overview)

* **System Type:** Airborne Wind Turbine with Fixed Flight (Tethered Fly-Gen AWE) capable of vertical takeoff/landing (VTOL).
* **Layout:** Quad-rotor (quadcopter layout).
* **Ecology Goal:** 100% bird protection (Bird-Safe Design) due to large visual presence, absence of sharp blades, and lack of “sweeping” figure-eight motions in the sky.
* **Rated Power:** Reference baseline **20+ kW**, with design tolerance for much higher production (scalable) depending on the generator and high-altitude winds.
* **Total Weight (Dry Mass):** < 280 kg (reduced thanks to the 3-cylinder rotor configuration).

---

### 2. The Rotor System (Rotor Assembly)

Each rotor eliminates classic blades by using rotating cylinders.

* **Rotor Diameter:** 8 meters.
* **Magnus Cylinders (Spokes):** **3 per rotor** (12 total).

*Rationale:* Just like in conventional horizontal-axis wind turbines (HAWT), a 3-blade configuration is ideal for balancing vibrations. It prevents asymmetric gyroscopic torques (1P/3P vibrations) on the shaft when the craft changes direction (yaw) or is hit by gusts, protecting the bearings from fatigue—something 2- or 4-blade configurations cannot handle as smoothly.

* **Coning Angle:** 10–15 degrees backward (downwind swept).

*Rationale:* The wind pushes the rotor. The coning angle converts pressure into radial tension, putting the outer rim in pure tensile load and keeping it perfectly circular without heavy reinforcement.

* **Rotor Speed:** Constantly low (e.g., 10–20 RPM).

---

### 3. Control & Stability (The Quad Paradox)

This is the “heart” of the invention, explaining why it appears here and not in conventional aviation.

* **Mechanism:** 12 independent BLDC motors control the spin (up to 3,000 RPM) of each cylinder individually. The autopilot stabilizes the craft by changing the speed of the lightweight cylinders, not the large rotors.

* **The Advantage (Why the Quad Works Here):**
  In conventional drones, quadcopter architecture collapses if the rotors become huge (e.g., 8 meters) due to the Law of Inertia ($R^5$). A heavy rotor changes speed slowly, making the craft unstable. The “Quad-Magnus” bypasses this law: it keeps the inertia of the large rotor constant and changes *instantly* the spin of the small, ultra-light cylinders.
  You get the agility of a small drone in an industrial-scale craft, without complex helicopter swashplates.

* **The Disadvantage (Why It Doesn’t Exist in Other Aircraft/Energy Systems):**
  The Magnus cylinder has a terrible Lift-to-Drag ratio (L/D) and low Figure of Merit in hover compared to carbon-fiber blades. In a free-flying airplane or eVTOL, using cylinders would be “aerodynamic and energy suicide” (huge battery consumption, massive drag).
  Here, however, because the craft is *permanently tethered* and generates electricity, absolute aerodynamic efficiency (L/D) is sacrificed in favor of mechanical simplicity, cheap flight control, and bird protection.

---

### 4. Power Extraction (Power Take-Off – PTO)

* **Rim Drive (Mechanical):** External toothed belt (or gear teeth) around the 8-meter rim, transferring motion to small gears/wheels.
* **Generators:** 4 off-the-shelf (COTS) industrial generators mounted on the perimeter, taking advantage of the rotational speed multiplication of the large ring.

*Rationale:* This eliminates the weight and vulnerability of a central gearbox while avoiding the astronomical cost (hundreds of thousands of euros) and collision risk of a custom 8-meter air-gap electromagnetic generator.

---

### 5. Frame & Aerodynamics (Tensile Rigid Frame)

The craft has a conventional rigid frame (not just cables) to maintain its geometry, but it is designed to operate mainly in tension.

* **Rotor Tilt (Dihedral / Outward Tilt):** The frame gives the 4 rotors a slight outward tilt of **3–4 degrees**.
* **Frame Structure:** Tubular frame made from lightweight alloy (e.g., aluminum). It is placed below the rotors (underslung) on the tether side.

*Rationale:* Because of the 3–4 degree tilt, as the rotors generate lift they also try to “escape” outward. This pulls on the frame tubes, placing them under pure *tension*. Since materials do not buckle when pulled, the frame can be made from extremely thin-section, lightweight aluminum (“small aluminum tubes”), saving hundreds of kilograms of weight. At the same time, being located under the rotors allows the frame to avoid aerodynamic hammering (wake blockage) from the cylinders.



## Experimental Scope & Limitations

This project is released as a **Proof-of-Concept and Academic Sandbox**. The physics engine is not yet accurate enough. Energy efficiency is probably overestimated.


## Installation & Usage

To run the simulation locally, you need Python 3.8+ installed.

1. Clone the repository:
   
   git clone [https://github.com/yourusername/QuadMagnus.git](https://github.com/yourusername/QuadMagnus.git)
   cd QuadMagnus

2. Install dependencies:

    pip install numpy pyvista

3. Run the physics engine:

   python quad12brushless.py


### Controls (In-Simulation):

* Use the UI sliders to adjust Wind Speed, Cylinder RPMs, Pitch Angle, and Generator Load.
* Toggle the UI checkboxes to visualize aerodynamic force components and autopilot behavior.

