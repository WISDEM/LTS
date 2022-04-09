import openmdao.api as om
from lts.lts import LTS_Outer_Rotor_Opt
from lts.structural import LTS_Outer_Rotor_Structural
import os
import pandas as pd
import numpy as np

ratings_known = [15, 17, 20, 25]
rotor_diameter = {}
rotor_diameter[15] = 242.24
rotor_diameter[17] = 257.88
rotor_diameter[20] = 279.71
rotor_diameter[25] = 312.73
rated_speed = {}
rated_speed[15] = 7.49
rated_speed[17] = 7.04
rated_speed[20] = 6.49
rated_speed[25] = 5.80
target_eff = 0.97

mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file

def cleanup_femm_files(clean_dir):
    files = os.listdir(clean_dir)
    for file in files:
        if file.endswith(".ans") or file.endswith(".fem") or file.endswith(".csv"):
            os.remove(os.path.join(clean_dir, file))

def copy_data(prob_in, prob_out):

    # Get all OpenMDAO inputs and outputs into a dictionary
    def create_dict(prob):
        dict_omdao = prob.model.list_inputs(val=True, hierarchical=False, prom_name=True, units=False, desc=False, out_stream=None)
        temp = prob.model.list_outputs(val=True, hierarchical=False, prom_name=True, units=False, desc=False, out_stream=None)
        dict_omdao.extend(temp)
        my_dict = {}
        for k in range(len(dict_omdao)):
            my_dict[ dict_omdao[k][1]["prom_name"] ] = dict_omdao[k][1]["val"]

        return my_dict

    in_dict = create_dict(prob_in)
    out_dict = create_dict(prob_out)

    for k in in_dict:
        if k in out_dict:
            prob_out[k] = in_dict[k]

    return prob_out

def save_data(fname, prob):
    # Remove file extension
    froot = os.path.splitext(fname)[0]

    # Get all OpenMDAO inputs and outputs into a dictionary
    var_dict = prob.model.list_inputs(val=True, hierarchical=True, prom_name=True, units=True, desc=True, out_stream=None)
    out_dict = prob.model.list_outputs(val=True, hierarchical=True, prom_name=True, units=True, desc=True, out_stream=None)
    var_dict.extend(out_dict)

    data = {}
    data["variables"] = []
    data["units"] = []
    data["values"] = []
    data["description"] = []
    for k in range(len(var_dict)):
        unit_str = var_dict[k][1]["units"]
        if unit_str is None:
            unit_str = ""

        iname = var_dict[k][1]["prom_name"]
        if iname in data["variables"]:
            continue

        data["variables"].append(iname)
        data["units"].append(unit_str)
        data["values"].append(var_dict[k][1]["val"])
        data["description"].append(var_dict[k][1]["desc"])
    df = pd.DataFrame(data)
    df.to_excel(froot + ".xlsx", index=False)
    df.to_csv(froot + ".csv", index=False)

def optimize_magnetics_design(prob_in=None, output_dir=None, cleanup_flag=True, opt_flag=False, obj_str="cost", ratingMW=17):
    if output_dir is None:
        output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    ratingMW = int(ratingMW)
    target_torque = 1e6*ratingMW/(np.pi*rated_speed[ratingMW]/30.0)/target_eff

    modeling_options = {}
    modeling_options["output_dir"] = output_dir

    # Clean run directory before the run
    if cleanup_flag:
        cleanup_femm_files(mydir)

    prob = om.Problem()
    prob.model = LTS_Outer_Rotor_Opt(modeling_options=modeling_options)

    #prob.driver = om.ScipyOptimizeDriver()
    #prob.driver.options['optimizer'] = 'COBYLA' #'SLSQP' #
    #prob.driver.options["maxiter"] = 500 #50
    prob.driver = om.DifferentialEvolutionDriver()
    prob.driver.options["max_gen"] = 15
    prob.driver.options["pop_size"] = 60
    prob.driver.options["penalty_exponent"] = 2

    recorder = om.SqliteRecorder(os.path.join(output_dir, "log.sql"))
    prob.driver.add_recorder(recorder)
    prob.add_recorder(recorder)
    prob.driver.recording_options["excludes"] = ["*_df"]
    prob.driver.recording_options["record_constraints"] = True
    prob.driver.recording_options["record_desvars"] = True
    prob.driver.recording_options["record_objectives"] = True

    prob.model.add_design_var("D_a", lower=5, upper=9, ref=8)
    prob.model.add_design_var("delta_em", lower=0.060, upper=0.15, ref=0.08)
    prob.model.add_design_var("h_sc", lower=0.03, upper=0.15, ref=0.06)
    prob.model.add_design_var("h_s", lower=0.1, upper=0.4, ref=0.1)
    prob.model.add_design_var("p", lower=20, upper=30, ref=25)
    prob.model.add_design_var("h_yr", lower=0.01, upper=0.45, ref=0.1)
    prob.model.add_design_var("l_s", lower=1, upper=1.75)
    prob.model.add_design_var("alpha", lower=0.1, upper=1)
    prob.model.add_design_var("dalpha", lower=1, upper=5)
    prob.model.add_design_var("I_sc", lower=150, upper=700, ref=450)
    prob.model.add_design_var("N_sc", lower=1500, upper=3500, ref=1500)
    prob.model.add_design_var("N_c", lower=1, upper=15, ref=8)
    prob.model.add_design_var("I_s", lower=500, upper=3000, ref=1750)
    # prob.model.add_design_var("J_s", lower=1.5, upper=6, ref=3.75)

    # prob.model.add_constraint("Slot_aspect_ratio", lower=4.0, upper=10.0)  # 11
    prob.model.add_constraint("con_angle", lower=0.001)
    # Differential evolution driver cannot do double-sided constraints, so have to hack it
    prob.model.add_constraint("E_p", lower=0.8 * 3300, ref=3000)
    prob.model.add_constraint("E_p_ratio", upper=1.20)
    prob.model.add_constraint("B_coil_max", lower=6.0)
    prob.model.add_constraint("Coil_max_ratio", upper=1.2)  # Consider user-defined limit instead of load line
    prob.model.add_constraint("B_rymax", upper=2.1)
    prob.model.add_constraint("torque_ratio", lower=1.0)
    prob.model.add_constraint("Torque_actual", upper=1.2*target_torque, ref=20e6)
    prob.model.add_constraint("Critical_current_ratio",upper=1.2)
    if not obj_str.lower() in ['eff','efficiency']:
        prob.model.add_constraint("gen_eff", lower=0.97)

    if obj_str.lower() == 'cost':
        prob.model.add_objective("cost_total", ref=1e6)
    elif obj_str.lower() == 'mass':
        prob.model.add_objective("mass_total", ref=1e6)
    elif obj_str.lower() in ['eff','efficiency']:
        prob.model.add_objective("gen_eff", scaler=-1.0)

    prob.model.approx_totals(method="fd")

    prob.setup()
    # --- Design Variables ---

    if prob_in is None:
        prob["m"] = 6  # phases
        prob["q"] = 2  # slots per pole per phase
        prob["b_s_tau_s"] = 0.45
        prob["conductor_area"] = 1.8 * 1.2e-6
        prob["K_h"] = 2  # specific hysteresis losses W/kg @ 1.5 T
        prob["K_e"] = 0.5  # specific hysteresis losses W/kg @ 1.5 T

        # ## Initial design variables for a PMSG designed for a 15MW turbine
        print(ratingMW, rated_speed[ratingMW], target_torque)
        prob["system_coverage"] = 0.63
        prob["P_rated"] = ratingMW * 1e6
        prob["T_rated"] = target_torque
        prob["E_p_target"] = 3300.0
        prob["N_nom"] = rated_speed[ratingMW]
        prob["D_a"] = 9.0
        prob["delta_em"] = 0.1248916 #0.07114809
        prob["h_s"] = 0.1 #0.26448025
        prob["p"] = 30.0
        prob["h_sc"] = 0.06053662 #0.0798385
        prob["h_yr"] = 0.16824667 #0.4
        prob["alpha"] = 0.1
        prob["dalpha"] = 1.0
        prob["I_sc"] = 284.90199962 #675.23529314
        prob["N_sc"] = 2811.37208924 #1500.0
        prob["N_c"] = 2.2532984 #1.0
        prob["I_s"] = 1945.9858772 #2933.37488172
        prob["J_s"] = 3.0
        prob["l_s"] = 1.11086996 #1.75

        # Specific costs
        prob["C_Cu"] = 10.3  #  https://markets.businessinsider.com/commodities/copper-price
        prob["C_Fe"] = 0.556
        prob["C_Fes"] = 0.50139
        prob["C_NbTi"] = 30.0

        # Material properties
        prob["rho_steel"] = 7850
        prob["rho_Fe"] = 7700.0  # Steel density
        prob["rho_Copper"] = 8900.0  # Kg/m3 copper density
        prob["rho_NbTi"] = 8442.37093661195  # magnet density
        prob["resisitivty_Cu"] = 1.724e-8  # 1.8e-8 * 1.4  # Copper resisitivty
        prob["U_b"] = 1  # brush voltage drop
        prob["Y"] = 10  # Short pitch

        prob["Tilt_angle"] = 90.0
        prob["R_shaft_outer"] = 1.25
        prob["R_nose_outer"] = 0.95
        prob["u_allow_pcent"] = 30
        prob["y_allow_pcent"] = 20
        prob["h_yr_s"] = 0.025 #0.18866371
        prob["h_ys"] = 0.025
        prob["t_rdisc"] = 0.02507061
        prob["t_sdisc"] = 0.025
        prob["y_bd"] = 0.00
        prob["theta_bd"] = 0.00
        prob["y_sh"] = 0.00
        prob["theta_sh"] = 0.00

    else:
        prob = copy_data(prob_in, prob)

    prob.model.approx_totals(method="fd")

    if opt_flag:
        prob.run_driver()
    else:
        prob.run_model()

    # Clean run directory after the run
    if cleanup_flag:
        cleanup_femm_files(mydir)

    prob.model.list_outputs(val=True, hierarchical=True)
    print("Final solution:")
    print("E_p_ratio", prob["E_p_ratio"])
    # print("con_angle", prob["con_angle"])
    print("gen_eff", prob["gen_eff"])
    print("N_c", prob["N_c"])
    print("N_sc", prob["N_sc"])
    print("B_coil_max", prob["B_coil_max"])
    print("l_s", prob["l_s"])
    print("Torque_actual", prob["Torque_actual"])

    return prob

def optimize_structural_design(prob_in=None, output_dir=None, opt_flag=False):
    if output_dir is None:
        output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    prob_struct = om.Problem()
    prob_struct.model = LTS_Outer_Rotor_Structural()

    prob_struct.driver = om.ScipyOptimizeDriver()
    prob_struct.driver.options["optimizer"] = "SLSQP"
    prob_struct.driver.options["maxiter"] = 100

    #recorder = om.SqliteRecorder("log.sql")
    #prob_struct.driver.add_recorder(recorder)
    #prob_struct.add_recorder(recorder)
    #prob_struct.driver.recording_options["excludes"] = ["*_df"]
    #prob_struct.driver.recording_options["record_constraints"] = True
    #prob_struct.driver.recording_options["record_desvars"] = True
    #prob_struct.driver.recording_options["record_objectives"] = True

    prob_struct.model.add_design_var("h_yr_s", lower=0.0250, upper=0.5, ref=0.1)
    prob_struct.model.add_design_var("h_ys", lower=0.025, upper=0.6, ref=0.1)
    prob_struct.model.add_design_var("t_rdisc", lower=0.025, upper=0.5, ref=0.1)
    prob_struct.model.add_design_var("t_sdisc", lower=0.025, upper=0.5, ref=0.1)
    prob_struct.model.add_objective("mass_structural", ref=1e6)

    prob_struct.model.add_constraint("U_rotor_radial_constraint", upper=1.0)
    prob_struct.model.add_constraint("U_rotor_axial_constraint", upper=1.0)
    prob_struct.model.add_constraint("U_stator_radial_constraint", upper=1.0)
    prob_struct.model.add_constraint("U_stator_axial_constraint", upper=1.0)

    prob_struct.model.approx_totals(method="fd")

    prob_struct.setup()
    # --- Design Variables ---

    if prob_in is None:
        # Initial design variables for a PMSG designed for a 15MW turbine
        #prob_struct["Sigma_shear"] = 74.99692029e3
        prob_struct["Sigma_normal"] = 378.45123826e3
        prob_struct["T_e"] = 9e06
        prob_struct["l_eff_stator"] = 1.44142189  # rev 1 9.94718e6
        prob_struct["l_eff_rotor"] = 1.2827137
        prob_struct["D_a"] = 7.74736313
        prob_struct["delta_em"] = 0.0199961
        prob_struct["h_s"] = 0.1803019703
        prob_struct["D_sc"] = 7.78735533
        prob_struct["rho_steel"] = 7850
        prob_struct["rho_Fe"] = 7700
        prob_struct["Tilt_angle"] = 90.0
        prob_struct["R_shaft_outer"] = 1.25
        prob_struct["R_nose_outer"] = 0.95
        prob_struct["u_allow_pcent"] = 50
        prob_struct["y_allow_pcent"] = 20
        prob_struct["h_yr"] = 0.1254730934
        prob_struct["h_yr_s"] = 0.025
        prob_struct["h_ys"] = 0.050
        prob_struct["t_rdisc"] = 0.05
        prob_struct["t_sdisc"] = 0.1
        prob_struct["y_bd"] = 0.00
        prob_struct["theta_bd"] = 0.00
        prob_struct["y_sh"] = 0.00
        prob_struct["theta_sh"] = 0.00

        #prob_struct["mass_copper"] = 60e3
        prob_struct["mass_SC"] = 4000
    else:
        prob_struct = copy_data(prob_in, prob_struct)

    prob_struct.model.approx_totals(method="fd")

    if opt_flag:
        prob_struct.run_driver()
    else:
        prob_struct.run_model()

    #prob_struct.model.list_outputs(val=True, hierarchical=True)

    raw_data = {
        "Parameters": [
            "Rotor disc thickness",
            "Rotor yoke thickness",
            "Stator disc thickness",
            "Stator yoke thickness",
            "Rotor radial deflection",
            "Rotor axial deflection",
            "Stator radial deflection",
            "Stator axial deflection",
            "Rotor structural mass",
            "Stator structural mass",
            "Total structural mass",
        ],
        "Values": [
            prob_struct.get_val("t_rdisc", units="mm"),
            prob_struct.get_val("h_yr_s", units="mm"),
            prob_struct.get_val("t_sdisc", units="mm"),
            prob_struct.get_val("h_ys", units="mm"),
            prob_struct.get_val("u_ar", units="mm"),
            prob_struct.get_val("y_ar", units="mm"),
            prob_struct.get_val("u_as", units="mm"),
            prob_struct.get_val("y_as", units="mm"),
            prob_struct.get_val("Structural_mass_rotor", units="t"),
            prob_struct.get_val("Structural_mass_stator", units="t"),
            prob_struct.get_val("mass_structural", units="t"),
        ],
        "Limit": [
            "",
            "",
            "",
            "",
            prob_struct.get_val("u_allowable_r", units="mm"),
            prob_struct.get_val("y_allowable_r", units="mm"),
            prob_struct.get_val("u_allowable_s", units="mm"),
            prob_struct.get_val("y_allowable_s", units="mm"),
            "",
            "",
            "",
        ],
        "Units": [
            "mm",
            "mm",
            "mm",
            "mm",
            "mm",
            "mm",
            "mm",
            "mm",
            "tons",
            "tons",
            "tons",
        ],
    }
    #df = pd.DataFrame(raw_data, columns=["Parameters", "Values", "Limit", "Units"])
    #df.to_excel(os.path.join(output_dir, "Optimized_structure_LTSG_MW.xlsx"))

    return prob_struct

def write_all_data(prob, output_dir=None):
    if output_dir is None:
        output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    save_data(os.path.join(output_dir, "LTS_output"), prob)

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
            "Rotor disc thickness",
            "Rotor yoke thickness",
            "Stator disc thickness",
            "Stator yoke thickness",
            "Rotor radial deflection",
            "Rotor axial deflection",
            "Rotor torsional twist",
            "Stator radial deflection",
            "Stator axial deflection",
            "Stator torsional twist",
            "Rotor structural mass",
            "Stator structural mass",
            "Total structural mass",
            "Total generator mass",
            "Total generator cost",
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
            prob["Slots"],
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
            prob["mass_SC_racetrack"],
            prob.get_val("mass_SC", units="t"),
            prob["B_rymax"],
            prob["B_g"],
            prob["B_coil_max"],
            prob.get_val("mass_copper", units="t"),
            prob.get_val("mass_iron", units="t"),
            prob.get_val("mass_active", units="t"),
            prob["gen_eff"] * 100,
            prob.get_val("t_rdisc", units="mm"),
            prob.get_val("h_yr_s", units="mm"),
            prob.get_val("t_sdisc", units="mm"),
            prob.get_val("h_ys", units="mm"),
            prob.get_val("u_ar", units="mm"),
            prob.get_val("y_ar", units="mm"),
            prob.get_val("twist_r"),
            prob.get_val("u_as", units="mm"),
            prob.get_val("y_as", units="mm"),
            prob.get_val("twist_s"),
            prob.get_val("Structural_mass_rotor", units="t"),
            prob.get_val("Structural_mass_stator", units="t"),
            prob.get_val("mass_structural", units="t"),
            prob.get_val("mass_total", units="t"),
            prob.get_val("cost_total", units="USD"),
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
            "",
            "",
            "",
            "",
            prob.get_val("u_allowable_r", units="mm"),
            prob.get_val("y_allowable_r", units="mm"),
            "",
            prob.get_val("u_allowable_s", units="mm"),
            prob.get_val("y_allowable_s", units="mm"),
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
            "mm",
            "mm",
            "mm",
            "mm",
            "mm",
            "mm",
            "deg",
            "mm",
            "mm",
            "deg",
            "tons",
            "tons",
            "tons",
            "tons",
            "$",
        ],
    }

    df = pd.DataFrame(raw_data, columns=["Parameters", "Values", "Limit", "Units"])
    df.to_excel(os.path.join(output_dir, "Optimized_LTSG_" + str(prob["P_rated"][0] / 1e6) + "_MW.xlsx"))

def run_all(output_str, opt_flag, obj_str, ratingMW):
    output_dir = os.path.join(mydir, output_str)

    # Optimize just magnetrics with GA and then structural with SLSQP
    prob = optimize_magnetics_design(output_dir=output_dir, opt_flag=opt_flag, obj_str=obj_str, ratingMW=int(ratingMW))
    prob_struct = optimize_structural_design(prob_in=prob, output_dir=output_dir, opt_flag=opt_flag)

    # Bring all data together
    prob = copy_data(prob_struct, prob)
    prob.run_model()

    # Write to xlsx and csv files
    write_all_data(prob, output_dir=output_dir)
    prob.model.list_outputs(val=True, hierarchical=True)
    cleanup_femm_files(mydir)

if __name__ == "__main__":
    opt_flag = True
    for k in ratings_known:
        for obj in ["cost", "mass", "eff"]:
            run_all(f"outputs{k}-{obj}", opt_flag, obj, k)
