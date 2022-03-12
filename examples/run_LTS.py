import openmdao.api as om
from lts.lts import LTS_Outer_Rotor_Opt
import os
import pandas as pd

def cleanup_femm_files(clean_dir):
    files = os.listdir(clean_dir)
    for file in files:
        if file.endswith(".ans") or file.endswith(".fem") or file.endswith(".csv"):
            os.remove(os.path.join(clean_dir, file))

if __name__ == "__main__":

    mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
    output_dir = os.path.join(os.path.dirname(os.path.dirname(mydir)), 'outputs', 'test1')
    os.makedirs(output_dir, exist_ok=True)

    modeling_options = {}
    modeling_options['output_dir'] = output_dir

    cleanup_flag = True
    # Clean run directory before the run
    if cleanup_flag:
        cleanup_femm_files(mydir)

    prob = om.Problem()
    prob.model = LTS_Outer_Rotor_Opt(modeling_options = modeling_options)

    prob.driver = om.ScipyOptimizeDriver()  # pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'SLSQP' #'COBYLA' #'
    prob.driver.options["maxiter"] = 10
    # prob.driver.opt_settings['IPRINT'] = 4
    # prob.driver.opt_settings['ITRM'] = 3
    # prob.driver.opt_settings['ITMAX'] = 10
    # prob.driver.opt_settings['DELFUN'] = 1e-3
    # prob.driver.opt_settings['DABFUN'] = 1e-3
    # prob.driver.opt_settings['IFILE'] = 'CONMIN_LST.out'
    # prob.root.deriv_options['type']='fd'

    recorder = om.SqliteRecorder(os.path.join(output_dir,"log.sql"))
    prob.driver.add_recorder(recorder)
    prob.add_recorder(recorder)
    prob.driver.recording_options["excludes"] = ["*_df"]
    prob.driver.recording_options["record_constraints"] = True
    prob.driver.recording_options["record_desvars"] = True
    prob.driver.recording_options["record_objectives"] = True

    prob.model.add_design_var("D_a", lower=7, upper=8, ref=7.5)
    prob.model.add_design_var("delta_em", lower=0.060, upper=0.10, ref=0.08)
    prob.model.add_design_var("h_sc", lower=0.03, upper=0.25, ref=0.01)
    prob.model.add_design_var("h_s", lower=0.1, upper=0.4, ref=0.1)
    prob.model.add_design_var("p", lower=10, upper=30, ref=20)
    prob.model.add_design_var("h_yr", lower=0.01, upper=0.4, ref=0.1)
    prob.model.add_design_var("l_s", lower=1, upper=2.5, ref=1.625)
    prob.model.add_design_var("alpha", lower=0.5, upper=20, ref=10)
    prob.model.add_design_var("dalpha", lower=1, upper=10, ref=10)
    # prob.model.add_design_var('beta', lower=0.75, upper=20,ref=10)
    prob.model.add_design_var("I_sc", lower=200, upper=700, ref=450)
    prob.model.add_design_var("N_sc", lower=1500, upper=2500, ref=1500)
    prob.model.add_design_var("N_c", lower=2, upper=30, ref=16)
    prob.model.add_design_var("I_s", lower=500, upper=3000, ref=1750)
    #prob.model.add_design_var("J_s", lower=1.5, upper=6, ref=3.75)
    prob.model.add_objective("l_sc", ref=1e3)

    # prob.model.add_constraint('K_rad',    lower=0.15,upper=0.3)						#10
    # prob.model.add_constraint("Slot_aspect_ratio", lower=4.0, upper=10.0)  # 11
    prob.model.add_constraint("con_angle", lower=0.001)
    prob.model.add_constraint("con_angle2", lower=0.001)
    #prob.model.add_constraint("E_p", lower=3300, upper=3350)
    #prob.model.add_constraint("con_N_sc", lower=-5, upper=5)
    prob.model.add_constraint("B_rymax", upper=2.1)

    prob.model.add_constraint("Torque_actual", lower=23.07e6, ref=1e7)
    prob.model.add_constraint("Critical_current_ratio",upper=1.) 
    prob.model.add_constraint("Coil_max_ratio",upper=1.) 
    prob.model.approx_totals(method="fd")

    prob.setup()
    # --- Design Variables ---

    # UNUSED
    # Assign values to universal constants
    #prob["B_r"] = 1.279  # Tesla remnant flux density
    #prob["E"] = 2e11  # N/m^2 young's modulus
    #prob["ratio"] = 0.8  # ratio of magnet width to pole pitch(bm/self.tau_p)
    #prob["mu_0"] = np.pi * 4e-7  # permeability of free space

    #prob["mu_r"] = 1.06  # relative permeability
    #prob["cofi"] = 0.85  # power factor

    # Assign values to design constants
    #prob["h_0"] = 0.005  # Slot opening height
    #prob["h_1"] = 0.004  # Slot wedge height
    #prob["k_sfil"] = 0.65  # Slot fill factor
    #prob["P_Fe0h"] = 4  # specific hysteresis losses W/kg @ 1.5 T
    #prob["P_Fe0e"] = 1  # specific hysteresis losses W/kg @ 1.5 T
    #prob["k_fes"] = 0.8  # Iron fill factor

    # Assign values to universal constants
    #prob["gravity"] = 9.8106  # m/s**2 acceleration due to gravity
    #prob["E"] = 2e11  # Young's modulus
    #prob["phi"] = 90 * 2 * np.pi / 360  # tilt angle (rotor tilt -90 degrees during transportation)
    #prob["v"] = 0.3  # Poisson's ratio
    #prob["G"] = 79.3e9
    prob["m"] = 6  # phases
    prob["q"] = 2  # slots per pole
    prob["b_s_tau_s"] = 0.45
    prob["conductor_area"] = 1.8 * 1.2e-6
    prob["K_h"] = 2.0
    prob["K_e"] = 0.5

    # Initial design variables for a PMSG designed for a 15MW turbine
    prob["P_rated"] = 17e6
    prob["T_rated"] = 23.07e6  # rev 1 9.94718e6
    prob["N_nom"] = 7.7  # 7.5598598  #8.68                # rpm 9.6
    prob["l_s"] = 1.00390095  # 8.68                # rpm 9.6
    prob["D_a"] = 7.74736313  # rev 1 6.8
    prob["delta_em"] = 0.02  # rev 2.1
    prob["h_s"] = 0.1803019703  # rev 1 0.3
    prob["p"] = 21 # 100.0    # rev 1 160
    prob["h_sc"] = 0.0503409354 # rev 1 0.034
    prob["h_yr"] = 0.1254730934  # rev 1 0.045
    prob["alpha"] = 0.53805442  # rev 1 0.045
    prob["dalpha"] = 1.0  # rev 1 0.045
    # prob['beta']        =   1.75 # rev 1 0.045
    prob["I_sc"] = 400.4427393  # rev 1 0.045
    prob["N_sc"] = 1502  # rev 1 0.045
    prob["N_c"] = 2
    prob["I_s"] = 2995.06090335
    prob["J_s"] = 3.0

    # Material properties
    prob["rho_Fe"] = 7700.0  # Steel density
    prob["rho_Copper"] = 8900.0  # Kg/m3 copper density
    prob["rho_NbTi"] = 8442.37093661195  # magnet density
    prob["rho_Cu"] = 1.8e-8 * 1.4  # Copper resisitivty
    prob["U_b"] = 2     # brush contact voltage
    prob["Y"] = 10                 #Short pitch

    #prob.model.approx_totals(method="fd")

    #prob.run_model()
    prob.run_driver()

    # Clean run directory after the run
    if cleanup_flag:
        cleanup_femm_files(mydir)

    prob.model.list_outputs(values = True, hierarchical=True)

    raw_data = {
        "Parameters": [
            "Rating",
            "Armature diameter",
            "Field coil diameter",
            "Stator length",
            "l_eff_rotor",
            "l_eff_stator",
            "l/d ratio",
            "Alpha",
            "Beta",
            "Slot_aspect_ratio",
            "Pole pitch",
            "Slot pitch",
            "Stator slot height",
            "Stator slotwidth",
            "Stator tooth width",
            "Rotor yoke height",
            "Field coil height",
            "Field coil width",
            "Outer width",
            "Coil separation distance",
            "Pole pairs",
            "Generator Terminal voltage",
            "Stator current",
            "Armature slots",
            "Armature turns/phase/pole",
            "Armature current density",
            "Resistance per phase",
            "Shear stress",
            "Normal stress",
            "Torque",
            "Field coil turns",
            "Field coil current",
            "Layer count",
            "Turns per layer",
            "length per racetrack",
            "Mass per coil",
            "Total mass of SC coils",
            "B_rymax",
            "B_g",
            "B_coil_max",
            "Copper mass",
            "Iron mass",
            "Total active mass",
            "Efficiency",
        ],
        "Values": [
            prob.get_val("P_rated", units="MW"),
            prob["D_a"],
            prob["D_sc"],
            prob["l_s"],
            prob["l_eff_rotor"],
            prob["l_eff_stator"],
            prob["K_rad"],
            prob["alpha"],
            prob["beta"],
            prob["Slot_aspect_ratio"],
            prob.get_val("tau_p", units="mm"),
            prob.get_val("tau_s", units="mm"),
            prob.get_val("h_s", units="mm"),
            prob.get_val("b_s", units="mm"),
            prob.get_val("b_t", units="mm"),
            prob.get_val("h_yr", units="mm"),
            prob.get_val("h_sc", units="mm"),
            prob.get_val("W_sc", units="mm"),
            prob.get_val("Outer_width", units="mm"),
            prob.get_val("a_m", units="mm"),
            prob["p1"],
            prob["E_p"],
            prob["I_s"],
            prob["S"],
            int(prob["N_c"]),
            prob["J_s"],
            prob["R_s"],
            prob.get_val("Sigma_shear", units="kPa"),
            prob.get_val("Sigma_normal", units="kPa"),
            prob.get_val("Torque_actual", units="MN*m"),
            int(prob["N_sc"]),
            prob["I_sc"],
            prob["N_l"],
            prob["N_sc_layer"],
            prob.get_val("l_sc", units="km"),
            prob["mass_SC"],
            prob.get_val("Total_mass_SC",units="t"),
            prob["B_rymax"],
            prob["B_g"],
            prob["B_coil_max"],
            prob.get_val("Copper", units="t"),
            prob.get_val("Iron", units="t"),
            prob.get_val("Mass", units="t"),
            prob["gen_eff"] * 100,
        ],
        "Limit": [
            "",
            "",
            "",
            "",
            "",
            "",
            "(0.15-0.3)",
            "",
            "",
            "(4-10)",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "<2.1 Tesla",
            "",
            "",
            "",
            "",
            "",
            "",
        ],
        "Units": [
            "MW",
            "m",
            "m",
            "m",
            "m",
            "m",
            "",
            "deg",
            "deg",
            "",
            "mm",
            "mm",
            "mm",
            "mm",
            "mm",
            "mm",
            "mm",
            "mm",
            "mm",
            "mm",
            "",
            "Volts",
            "A",
            "slots",
            "turns",
            "A/mm^2",
            "ohm",
            "kPa",
            "kPa",
            "MNm",
            "turns",
            "A",
            "layers",
            "turns",
            "km",
            "kg",
            "Tons",
            "Tesla",
            "Tesla",
            "Tesla",
            "Tons",
            "Tons",
            "Tons",
            "%",
        ],
    }
    #print(raw_data)
    df = pd.DataFrame(raw_data, columns=["Parameters", "Values", "Limit", "Units"])

    #print(df)

    df.to_excel(os.path.join(output_dir,"Optimized_LTSG_" + str(prob["P_rated"][0] / 1e6) + "_MW.xlsx"))
    print("Final solution:")
    print("Slot", prob["Slot_aspect_ratio"])
    print("con_angle", prob["con_angle"])
    #print("con_I_sc", prob["con_I_sc"])
    #print("con_N_sc", prob["con_N_sc"])
    print("B_g", prob["B_g"])
    print("l_s", prob["l_s"])
    print("Torque_actual", prob["Torque_actual"])
