# Tethered-Quad-Magnus
Airborne Wind Turbine for steady flight (Tethered Fly-Gen AWE) with Quad-rotor layout and vertical takeoff/landing capability (VTOL).


[![License: CERN-OHL](https://img.shields.io/badge/License-CERN%20OHL-blue.svg)](https://ohwr.org/cernohl)
[![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)](https://www.python.org/)


## The Engineering Philosophy: Active Lift over Passive Glide

Traditional AWE systems (tethered kites or gliders) rely on passive airfoils that are highly susceptible to stall during turbulent gusts or sudden wind drops. QuadMagnus explores active aerodynamic control.

By utilizing the Magnus effect, the drone can theoretically generate lift coefficients ($C_L$) even at near-zero relative wind speeds.

---

## 1. General Architecture & Dimensions (System Overview)

* **System Type:** Airborne Wind Turbine for steady flight (Tethered Fly-Gen AWE) with **vertical takeoff/landing capability (VTOL)**.
* **Configuration:** Quad-rotor layout.
* **Total Weight (Dry Mass):** ~300 kg.
* **Rated Power:** 20 kW (at wind speed 8–10 m/s).
* **Tether Tension:** Estimated **10–12 tons** under full load.

---

## 2. Rotor Assembly

Each of the four rotors does **not** have conventional blades. Instead, it consists of a central **hub**, four cylindrical **spokes**, and an outer **rim**.

* **Rotor Diameter:** 8 meters.
* **Coning Angle:** 10–15° backward (downwind swept).

**Rationale:**
The wind strikes the rotor and pushes it backward. The coning angle forces this force to convert into **radial tension**, putting the outer rim in **100% pure tension (stretch)**. The rim does not buckle and remains a perfectly rigid **8-meter circle** while weighing very little.

### Magnus Cylinders (Spokes)

* **Quantity:** 4 per rotor (16 total).
* **Dimensions:** Length 4 m, radius 0.15 m.
* **Material:** Internal aluminum/carbon ribs covered with lightweight durable **ripstop fabric** (~6 kg each).

### Rotor Rotation Speed

* **Rotor RPM:** Slow rotation, constant **10–15 RPM** during generation.

**Rationale:**
Absolute **bird safety** and minimization of **parasitic drag** from the huge rotor disk.

---

## 3. Control & Stability (Flight Authority)

Heavy variable-pitch systems (**swashplates**) or large-rotor RPM changes are **completely absent**.

* **Mechanism:** 16 independent **COTS (Commercial Off-The-Shelf) BLDC drone motors**, mounted at the base of each cylinder (at the hub).
* **Cylinder Spin:** Up to **3,000 RPM**.
* **Bearings:** **Tapered roller bearings** (e.g., SKF 30205) at each cylinder end, capable of simultaneously handling **>2,000 kg dynamic load** (lift + cone compression) at 3,000 RPM.

**Rationale:**
The autopilot stabilizes the craft (**pitch/roll**) by instantly adjusting the **spin of the small lightweight cylinders**. The **Magnus lift** changes differentially across each rotor quadrant, creating immediate control torque **without the nightmare of inertia**.

---

## 4. Power Take-Off (PTO)

Electricity generation is not from the center but from the **rotor perimeter (rim-driven)**, using mechanical transmission instead of extremely expensive electromagnets.

* **Transmission Mechanism:** External **timing belt** around the 8-meter rim driving a small gear/pulley.
* **Gear Ratio:** Approximately **40:1**.
* **Generators:** **4 × 5 kW sealed industrial BLDC generators** (total 20 kW).

**Rationale:**
The **10 RPM** of the massive rotor becomes **400+ RPM** at the generator. This saves **dozens of kilograms** by eliminating a traditional **central gearbox**, dramatically reducing cost while keeping the generator **air gap completely safe from aeroelastic deformation**.

---

## 5. Frame & Aerodynamics (Underslung Tensegrity Frame)

The frame is **not** a heavy rigid cross. It is a **tension-based cable structure**.

* **Rotor Tilt (Outward Tilt / Dihedral):** The four rotors are tilted **3–4° outward**.
* **Frame Structure:** The tether attachment point is located **low**, forming an **inverted pyramid**. The center (tether) pulls downward while the four rotors pull **upward and outward**.

**Rationale:**
The 4° tilt generates a **centrifugal horizontal force (~170 kg per rotor)** that tensions the frame. Instead of heavy beams that would buckle, the system is held together with **light aluminum tubes and Dyneema cables** (tensile structure).

By placing the frame **below the rotors (underslung)**, the system completely avoids **aerodynamic shadowing (wake blockage)** that would otherwise destroy the performance of the Magnus cylinders.

---

## 6. Operational Profiles (Flight Modes)

### Kite Mode (Power Generation)

* At wind speeds **>6 m/s**, the craft flies statically at an **elevation angle ~30°–40°**.
* Rotors spin **passively at ~10 RPM**.
* The small cylinder motors consume **~1–2 kW** for spin.
* PTO produces **20+ kW**.
* **Net ground power:** ~18–19 kW.

### VTOL Mode (Takeoff / No Wind)

* In zero wind, the system **draws ~30 kW from the ground**.
* The four generators act as **powerful motors**, violently spinning the rotor rims at **30–40 RPM** via the belts.
* The spinning cylinders encounter **artificial airflow**, producing the required **vertical lift (>3,400 N)**.
* The system ascends **vertically like a drone** to reach higher-altitude wind resources.


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

