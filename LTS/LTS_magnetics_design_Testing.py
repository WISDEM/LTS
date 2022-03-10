"""PMSG_Outer_rotor.py
Created by Latha Sethuraman
Copyright (c) NREL. All rights reserved.
Electromagnetic design based on conventional magnetic circuit laws
Structural design based on {Structural mass in direct-drive permanent magnet electrical generators by
McDonald,A.S. et al. IET Renewable Power Generation(2008),2(1):3 http://dx.doi.org/10.1049/iet-rpg:20070071 """

import pandas as pd
import numpy as np
import openmdao.api as om
from Geometry_fea_torque5 import FEMM_Geometry

class LTS_active(om.ExplicitComponent):

    """Estimates overall electromagnetic dimensions and Efficiency of PMSG -arms generator."""

    def setup(self):

        self.add_discrete_input("m", 6, desc="number of phases")
        self.add_discrete_input("q", 2, desc="slots per pole")
        self.add_input("b_s_tau_s", 0.45, desc="??")
        self.add_input("conductor_area", 1.8 * 1.2e-6, desc="??")

        self.add_input("D_a", 0.0, units="m", desc="armature diameter ")
        self.add_input("l_s", 0.0, units="m", desc="Stack length ")
        self.add_input("h_s", 0.0, units="m", desc="Slot height ")

        self.add_output("D_sc", 0.0, units="m", desc="field coil diameter ")

        # field coil parameters
        self.add_input("h_sc", 0.0, units="m", desc="SC coil height")
        self.add_input("alpha_p", 0.0, desc="pole arc coefficient")
        self.add_input("alpha", 0.0, units="deg", desc="Start angle of field coil")
        self.add_input("dalpha", 0.0, units="deg", desc="Angle subtended by field coil")
        self.add_input("h_yr", 0.0, units="m", desc="rotor yoke height")
        #self.add_input("I_sc", 0.0, units="A", desc="SC current ")
        self.add_input("N_sc", 0.0, desc="Number of turns of SC field coil")
        self.add_input("W_sc", 0.0, units="m", desc="SC coil width")
        self.add_input("N_c", 0.0, desc="Number of turns per coil")
        self.add_input("p", 0.0, desc="Pole pairs ")
        self.add_output("p1", 0.0, desc="Pole pairs ")
        self.add_input("delta_em", 0.0, units="m", desc="airgap length ")
        self.add_input("Y", 0.0, desc="coil pitch")
        self.add_input("I_s", 0.0, units="A", desc="Generator output phase current")

        self.add_output("N_s", 0.0, desc="Number of turns per phase in series")
        self.add_output("N_l", 0.0, desc="Number of layers of the SC field coil")
        #self.add_output("Dia_sc", 0.0, units="m", desc="field coil diameter")
        self.add_input("Outer_width", 0.0, units="m", desc="Coil outer width")
        self.add_input("a_m", 0.0, units="m", desc="Coil separation distance")
        self.add_output("beta", 0.0, units="deg", desc="End angle of field coil")
        self.add_output("con_angle", 0.0, units="deg", desc="End angle of field coil")
        self.add_output("con_angle2", 0.0, units="deg", desc="End angle of field coil")
        self.add_output("theta_p", 0.0, units="deg", desc="Pole pitch angle in degrees")

        # Material properties
        self.add_input("rho_Fe", 0.0, units="kg/(m**3)", desc="Electrical Steel density ")
        self.add_input("rho_Copper", 0.0, units="kg/(m**3)", desc="Copper density")
        self.add_input("rho_NbTi", 0.0, units="kg/(m**3)", desc="SC conductor mass density ")
        self.add_input("rho_Cu", 0.0, units="ohm*m", desc="Copper resistivity ")
        self.add_input("U_b", 0.0, units="V", desc="brush voltage ")

        self.add_input("P_rated", units="W", desc="Machine rating")
        self.add_input("N_nom", 0.0, units="rpm", desc="rated speed")
        self.add_input("T_rated", 0.0, units="N*m", desc="Rated torque ")
        #self.add_input("r_strand", 0.0, units="mm", desc="radius of the SC wire strand")
        #self.add_input("k_pf_sc", 0.0, units="mm", desc="packing factor for SC wires")
        self.add_input("J_s", 0.0, units="A/(mm*mm)", desc="Stator winding current density")
        #self.add_input("J_c", 0.0, units="A/(mm*mm)", desc="SC critical current density")

        # Magnetic loading
        self.add_output("tau_p", 0.0, units="m", desc="Pole pitch")
        self.add_output("b_p", 0.0, units="m", desc="distance between positive and negative side of field coil")
        self.add_output("alpha_u", 0.0, units="rad", desc="slot angle")
        self.add_output("tau_v", 0.0, units="m", desc="Phase zone span")
        self.add_output("zones", 0.0, desc="Phase zones")
        self.add_output("y_Q", 0.0, desc="Slots per pole also pole pitch")
        self.add_output("delta", 0.0, units="rad", desc="short-pitch angle")
        self.add_output("k_p1", 0.0, desc="Pitch factor-fundamental harmonic")
        self.add_output("k_d1", 0.0, desc="Distribution factor-fundamental harmonic")
        self.add_output("k_w1", 0.0, desc="Winding factor- fundamental harmonic")
        #self.add_output("Iscn", 0.0, units="A", desc="SC current")
        #self.add_output("Iscp", 0.0, units="A", desc="SC current")
        #self.add_output("g", 0.0, units="m", desc="Air gap length")
        
        self.add_output("l_sc", 0.0, units="m", desc="SC coil length")
        self.add_output("l_eff_rotor", 0.0, units="m", desc="effective rotor length with end windings")
        self.add_output("l_eff_stator", 0.0, units="m", desc="effective stator length with end field windings")
        self.add_output("l_Cus", 0.0, units="m", desc="copper winding length")
        self.add_output("R_sc", 0.0, units="m", desc="Radius of the SC coils")

        # Stator design
        #self.add_output("h41", 0.0, units="m", desc="Bottom coil height")
        #self.add_output("h42", 0.0, units="m", desc="Top coil height")
        self.add_output("b_s", 0.0, units="m", desc="slot width")
        self.add_output("b_t", 0.0, units="m", desc="tooth width")
        #self.add_output("h_t", 0.0, units="m", desc="tooth height")
        self.add_output("A_Cuscalc", 0.0, units="mm**2", desc="Conductor cross-section")
        self.add_output("tau_s", 0.0, units="m", desc="Slot pitch ")

        # Electrical performance
        self.add_output("f", 0.0, units="Hz", desc="Generator output frequency")
        self.add_output("R_s", 0.0, units="ohm", desc="Stator resistance")
        self.add_output("A_1", 0.0, units="A/m", desc="Electrical loading")
        #self.add_output("J_actual", 0.0, units="A/m**2", desc="Current density")
        #self.add_output("T_e", 0.0, units="N*m", desc="Electromagnetic torque")
        #self.add_output("Torque_constraint", 0.0, units="N/(m*m)", desc="Shear stress contraint")

        # Objective functions
        self.add_output("Mass", 0.0, units="kg", desc="Actual mass")
        self.add_output("K_rad", desc="Aspect ratio")
        self.add_output("Cu_losses", units="W", desc="Copper losses")
        self.add_output("P_add", units="W", desc="Additional losses")
        self.add_output("P_brushes", units="W", desc="brush losses")

        # Other parameters
        #self.add_output("R_out", 0.0, units="m", desc="Outer radius")
        self.add_output("S", 0.0, desc="Stator slots")
        self.add_output("Slot_aspect_ratio", 0.0, desc="Slot aspect ratio")

        # Mass Outputs
        self.add_output("mass_SC", 0.0, units="kg", desc="SC conductor mass per racetrack")
        self.add_output("Total_mass_SC", 0.0, units="kg", desc=" Total SC conductor mass")
        self.add_output("Copper", 0.0, units="kg", desc="Copper Mass")
        self.add_output("Iron", 0.0, units="kg", desc="Electrical Steel Mass")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):

        # Unpack inputs
        b_s_tau_s = inputs["b_s_tau_s"]
        conductor_area = inputs["conductor_area"]

        ###################################################### Electromagnetic design#############################################

        outputs["K_rad"] = inputs["l_s"][0] / (inputs["D_a"][0])  # Aspect ratio
        outputs["D_sc"] = inputs["D_a"] + 2 * inputs["delta_em"]
        outputs["R_sc"] = outputs["D_sc"] * 0.5
        outputs['p1']        =np.round(inputs['p'])

        # Calculating pole pitch
        #r_s = 0.5 * inputs["D_a"]  # Stator outer radius # UNUSED
        outputs["tau_p"] = (np.pi * (inputs["D_a"] + 2 * inputs["delta_em"] + 2 * inputs["h_sc"])) / (
            2 * outputs["p1"]
        )  # pole pitch
        outputs["b_p"] = inputs["alpha_p"] * outputs["tau_p"]

        # Calculating winding factor
        m = discrete_inputs["m"]
        q = discrete_inputs["q"]
        outputs["S"] = q * 2 * outputs["p1"] * m
        outputs["tau_s"] = np.pi * (inputs["D_a"]) / outputs["S"]  # Slot pitch
        outputs["alpha_u"] = outputs["p1"] * 2 * np.pi / outputs["S"]  # slot angle
        outputs["tau_v"] = outputs["tau_p"] / m
        outputs["zones"] = 2 * outputs["p1"] * m
        outputs["y_Q"] = outputs["S"] / (2 * outputs["p1"])  # coil span
        outputs["delta"] = inputs["Y"] * np.pi * 0.5 / outputs["y_Q"]  # short pitch angle     #coil span
        outputs["k_p1"] = np.sin(np.pi * 0.5 * inputs["Y"] / outputs["y_Q"])
        outputs["k_d1"] = np.sin(q * outputs["alpha_u"] * 0.5) / (q * np.sin(outputs["alpha_u"] * 0.5))
        outputs["k_w1"] = outputs["k_p1"] * outputs["k_d1"]

        # magnet width
        #alpha_p = np.pi / 2 * ratio # COMMENTING OUT BECAUSE ALPHA_P IS AN INPUT
        outputs["b_s"] = b_s_tau_s * outputs["tau_s"]
        outputs["b_t"] = outputs["tau_s"] - outputs["b_s"]  # slot width
        # UNUSED
        #gamma =  2 / np.pi * (
        #    np.atan(outputs["b_s"] * 0.5 / inputs["delta_em"])
        #    - (2 * inputs["delta_em"] / outputs["b_s"])
        #    * log(((1 + (outputs["b_s"] * 0.5 / inputs["delta_em"]) ** 2)) ** 0.5)
        #)

        #k_C = outputs["tau_s"] / (outputs["tau_s"] - gamma * (outputs["b_s"]))  # carter coefficient UNUSED
        #g_eff = k_C * inputs["delta_em"] # UNUSED

        # angular frequency in radians
        om_m = 2 * np.pi * inputs["N_nom"] / 60
        om_e = outputs["p1"] * om_m
        outputs["f"] = om_e / 2 / np.pi  # outout frequency
        # outputs['N_s']          =   outputs['p1']*inputs['Slot_pole']*N_c*3                  #2*m*p*q
        cos_theta_end = 1 - (outputs["b_s"] / (outputs["b_t"] + outputs["b_s"]) ** 2) ** 0.5
        #l_end = 4 * outputs["tau_s"] * 0.5 / cos_theta_end
        l_end = 10 * outputs["tau_s"] / 2 * np.tan(30 * np.pi / 180)
        
        outputs["l_eff_rotor"]   =inputs["l_s"]+2*l_end
        # l_end                  =     np.sqrt(3)/6*(10*outputs['tau_s']+)

        # Stator winding length ,cross-section and resistance
        outputs["l_Cus"] = 8 * l_end + 2 * inputs["l_s"]  # length of a turn
        z = outputs["S"]  # Number of coils
        #A_slot = inputs["h_s"] * outputs["b_s"] # UNUSED
        #d_cu = 2 * np.sqrt(outputs["A_Cuscalc"] / pi) # UNUSED
        outputs["A_Cuscalc"] = inputs["I_s"] * 1e-06 / (inputs["J_s"])
        #k_fill = A_slot / (2 * inputs["N_c"] * outputs["A_Cuscalc"]) # UNUSED
        outputs["N_s"] = int(inputs["N_c"])* z / (m)  # turns per phase
        
        outputs["R_s"] = (
            inputs["rho_Cu"]
            * (1 + 20 * 0.00393)
            * outputs["l_Cus"]
            * outputs["N_s"]
            * inputs["J_s"]*1000000
            / (2 * (inputs["I_s"]))
        )
        #print ("Resitance per phase:" ,outputs["R_s"])
        # r_strand                =0.425e-3
        # inputs['N_sc']         =  inputs['k_pf_sc']*outputs['W_sc']*inputs['h_sc']/pi*r_strand**2
        outputs["theta_p"] = np.rad2deg(outputs["tau_p"] / (outputs["R_sc"] + inputs["h_sc"]))
        outputs['beta']   = inputs["alpha"] + inputs["dalpha"]
        outputs["con_angle"] = 0.5*outputs["theta_p"] - outputs["beta"]
        outputs["con_angle2"] = outputs["beta"] - inputs["alpha"]
        # outputs['beta']   = (outputs['theta_p'] - 2*inputs['alpha'])*0.5-2
        #random_degree = np.random.uniform(1.0, outputs["theta_p"] * 0.5 - inputs["alpha"] - 0.25)
        #random_degree = np.mean([1.0, float(outputs["theta_p"]) * 0.5 - float(inputs["alpha"]) - 0.25])
        # outputs["beta"] = outputs["theta_p"] * 0.5 - random_degree
        
        outputs["l_sc"] = inputs["N_sc"] * (2 * inputs["l_s"] + np.pi * (inputs["a_m"] + inputs["W_sc"]))
        
        outputs["l_eff_stator"]=inputs["l_s"] +(inputs["a_m"] + inputs["W_sc"])
        
        # outputs['A_Cuscalc']	= I_n/J_s
        # A_slot                  = 2*N_c*outputs['A_Cuscalc']*(10**-6)/k_sfil
        outputs["Slot_aspect_ratio"] = inputs["h_s"] / outputs["b_s"]

        # Calculating stator current and electrical loading
        # +I_s              = sqrt(Z**2+(((outputs['E_p']-G**0.5)/(om_e*outputs['L_s'])**2)**2))
        # Calculating volumes and masses
        # V_Cus 	                    =   m*L_Cus*(outputs['A_Cuscalc']*(10**-6))     # copper volume
        # outputs['h_t']              =   (inputs['h_s']+h_1+h_0)
        V_Fery = (
            inputs["l_s"]
            * np.pi
            * 0.25
            * ((inputs["D_a"] - 2 * inputs["h_s"]) ** 2 - (inputs["D_a"] - 2 * inputs["h_s"] - 2 * inputs["h_yr"]) ** 2)
        )  # volume of iron in stator tooth
        # outputs['Copper']		    =   V_Cus*inputs['rho_Copper']
        outputs["Iron"] = V_Fery * inputs["rho_Fe"]  # Mass of stator yoke
        #k_pf = 0.8 # UNUSED
        # inputs['N_sc']  =int(inputs['N_sc'])
        # outputs['Dia_sc'] =2*((k_pf*outputs['W_sc']*inputs['h_sc'])/inputs['N_sc']/pi)**0.5
        #k_pf = inputs["N_sc"] * (1.8e-3 * 1.2e-3) / (outputs["W_sc"] * inputs["h_sc"]) # UNUSED
        # outputs['SC mass']          =

        # Calculating Losses
        ##1. Copper Losses
        # outputs['N_s']       =int(outputs['N_s'])
        # outputs['N_s']  =q*inputs['p']
        # print (inputs['alpha'],outputs['beta'])

        outputs["N_l"] = int(inputs["h_sc"] / (1.2e-3)) # round later!

        # 0.01147612156295224312590448625181
        outputs["mass_SC"] = outputs["l_sc"] * conductor_area * inputs["rho_NbTi"]
        outputs["Total_mass_SC"]=outputs["p1"]*outputs["mass_SC"]
        V_Cus = m * outputs["l_Cus"] * outputs["N_s"] * (outputs["A_Cuscalc"])
        outputs["Copper"] = V_Cus * inputs["rho_Copper"]
        
        outputs["Mass"] = outputs["Total_mass_SC"] + outputs["Iron"] + outputs["Copper"]
        outputs["A_1"] = (2 * inputs["I_s"] * outputs["N_s"] * m) / (np.pi * (inputs["D_a"]))
        outputs["Cu_losses"] = m * (inputs["I_s"]*0.707) ** 2 * outputs["R_s"]
        outputs["P_add"] = 0.01 * inputs["P_rated"]
        outputs["P_brushes"] = 2 * inputs["U_b"] * inputs["I_s"]
        
        #print (inputs["N_sc"],inputs["I_s"], outputs["p1"], inputs["D_a"],inputs["delta_em"], inputs["N_c"],outputs["S"])

        #print(outputs["mass_SC"])


class Results(om.ExplicitComponent):
    def setup(self):
        self.add_input("K_h", 2.0, desc="??")
        self.add_input("K_e", 0.5, desc="??")

        self.add_input("I_sc", 0.0, units="A", desc="SC current ")
        self.add_input("N_sc", 0.0, desc="Number of turns of SC field coil")
        self.add_input("N_l", 0.0, desc="Number of layers of the SC field coil")
        self.add_input("D_sc", 0.0, units="m", desc="field coil diameter ")
        self.add_input("k_w1", 0.0, desc="Winding factor- fundamental harmonic")
        self.add_input("B_rymax", 0.0, desc="Peak Rotor yoke flux density")
        self.add_input("B_g", 0.0, desc="Peak air gap flux density ")
        self.add_input("N_s", 0.0, desc="Number of turns per phase in series")
        self.add_input("N_nom", 0.0, units="rpm", desc="rated speed")
        self.add_input("p1", 0.0, desc="Pole pairs ")
        self.add_input("Iron", 0.0, units="kg", desc="Electrical Steel Mass")
        self.add_input("P_rated", units="W", desc="Machine rating")
        self.add_input("Cu_losses", units="W", desc="Copper losses")
        self.add_input("P_add", units="W", desc="Additional losses")
        self.add_input("P_brushes", units="W", desc="brush losses")
        self.add_input("l_s", 0.0, units="m", desc="Stator core length")
        #self.add_output("con_I_sc", 0.0, units="A/(mm*mm)", desc="SC current ")
        #self.add_output("con_N_sc", 0.0, desc="Number of turns of SC field coil")
        self.add_output("E_p", 0.0, units="V", desc="terminal voltage")
        self.add_output("N_sc_layer", 0.0, desc="Number of turns per layer")
        self.add_output("P_Fe", units="W", desc="Iron losses")
        self.add_output("P_Losses", units="W", desc="Total power losses")
        self.add_output("gen_eff", desc="Generator efficiency")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        # Unpack inputs
        K_h = inputs["K_h"]
        K_e = inputs["K_e"]

        #outputs["con_N_sc"] = inputs["N_sc"] - inputs["N_sc_out"]
        #outputs["con_I_sc"] = inputs["I_sc"] - inputs["I_sc_out"]
        outputs["N_sc_layer"] = int(inputs["N_sc"] / inputs["N_l"])

        # Calculating  voltage per phase
        om_m = 2 * np.pi * inputs["N_nom"] / 60
        outputs["E_p"] = inputs["l_s"]*(inputs["D_sc"] * 0.5 * inputs["k_w1"] * inputs["B_g"] * om_m * inputs["N_s"])
        
        
        #print ("Voltage and lengths are:",outputs["E_p"],inputs["l_s"] )
        

        om_e = inputs["p1"] * om_m
        outputs["P_Fe"] = (
            2 * (inputs["B_rymax"] / 1.5) ** 2 * (K_h * (om_e * 0.5 / np.pi) + K_e * (om_e * 0.5 / 50) ** 2) * inputs["Iron"]
        )
        outputs["P_Losses"] = inputs["Cu_losses"] + outputs["P_Fe"] + inputs["P_add"] + inputs["P_brushes"]
        outputs["gen_eff"] = 1 - outputs["P_Losses"] / inputs["P_rated"]


class LTS_Outer_rotor_Opt(om.Group):
    def setup(self):
        #self.linear_solver = lbgs = om.LinearBlockJac() #om.LinearBlockGS()
        #self.nonlinear_solver = nlbgs = om.NonlinearBlockGS()
        #nlbgs.options["maxiter"] = 3
        #nlbgs.options["atol"] = 1e-2
        #nlbgs.options["rtol"] = 1e-8
        #nlbgs.options["iprint"] = 2

        ivcs = om.IndepVarComp()
        ivcs.add_output("P_rated", 0.0, units="W", desc="Rated Power")
        ivcs.add_output("T_rated", 0.0, units="N*m", desc="Torque")
        ivcs.add_output("N_nom", 0.0, units="rpm", desc="rated speed")
        ivcs.add_output("D_a", 0.0, units="m", desc="Armature outer diameter")
        ivcs.add_output("delta_em", 0.0, units="m", desc="Field coil diameter")
        ivcs.add_output("h_s", 0.0, units="m", desc="Slot height")
        ivcs.add_output("p", 0, desc="Pole pairs")
        ivcs.add_output("h_yr", 0.0, units="m", desc="Rotor yoke height")
        ivcs.add_output("h_sc", 0.0, units="m", desc="SC coil height ")

        #ivcs.add_output("alpha_p", 0.0, desc="pole arc coefficient")
        ivcs.add_output("alpha", 0.0, units="deg", desc="Start angle of field coil")
        ivcs.add_output("dalpha", 0.0, units="deg", desc="Start angle of field coil")
        ivcs.add_output("I_sc", 0.0, units="A", desc="SC current")
        ivcs.add_output("N_sc", 0.0, desc="Number of turns of SC field coil")
        ivcs.add_output("N_c", 0.0, desc="Number of turns of armature winding")
        ivcs.add_output("I_s", 0.0, units="A", desc="Armature current")
        ivcs.add_output("J_s", 0.0, units="A/mm/mm", desc="Armature current density")
        ivcs.add_output("l_s", 0.0, units="m", desc="Stator core length")

        ivcs.add_discrete_output("m", 6, desc="number of phases")
        ivcs.add_discrete_output("q", 2, desc="slots per pole")
        ivcs.add_output("b_s_tau_s", 0.45, desc="??")
        ivcs.add_output("conductor_area", 1.8 * 1.2e-6, desc="??")
        ivcs.add_output("Y", 0.0, desc="coil pitch")
        ivcs.add_output("K_h", 2.0, desc="??")
        ivcs.add_output("K_e", 0.5, desc="??")

        ivcs.add_output("rho_Fe", 0.0, units="kg/(m**3)", desc="Electrical Steel density ")
        ivcs.add_output("rho_Copper", 0.0, units="kg/(m**3)", desc="Copper density")
        ivcs.add_output("rho_NbTi", 0.0, units="kg/(m**3)", desc="SC conductor mass density ")
        ivcs.add_output("rho_Cu", 0.0, units="ohm*m", desc="Copper resistivity ")
        ivcs.add_output("U_b", 0.0, units="V", desc="brush voltage ")
        #ivcs.add_output("r_strand", 0.0, units="mm", desc="radius of the SC wire strand")
        #ivcs.add_output("k_pf_sc", 0.0, units="mm", desc="packing factor for SC wires")
        #ivcs.add_output("J_c", 0.0, units="A/(mm*mm)", desc="SC critical current density")

        self.add_subsystem("ivcs", ivcs, promotes=["*"])
        self.add_subsystem("sys", LTS_active(), promotes=["*"])
        self.add_subsystem("geom", FEMM_Geometry(), promotes=["*"])
        self.add_subsystem("results", Results(), promotes=["*"])


if __name__ == "__main__":

    prob = om.Problem()
    prob.model = LTS_Outer_rotor_Opt()

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

    recorder = om.SqliteRecorder("log.sql")
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
    prob.model.add_constraint("Slot_aspect_ratio", lower=4.0, upper=10.0)  # 11
    prob.model.add_constraint("con_angle", lower=0.0)
    prob.model.add_constraint("con_angle2", lower=0.0)
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

    print(df)

    df.to_excel("Optimized_LTSG_" + str(prob["P_rated"][0] / 1e6) + "_MW.xlsx")
    print("Slot", prob["Slot_aspect_ratio"])
    print("con_angle", prob["con_angle"])
    #print("con_I_sc", prob["con_I_sc"])
    #print("con_N_sc", prob["con_N_sc"])
    print("B_g", prob["B_g"])
    print("l_s", prob["l_s"])
    print("Torque_actual", prob["Torque_actual"])
