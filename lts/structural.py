"""LTS_outer_stator.py
Created by Latha Sethuraman
Copyright (c) NREL. All rights reserved.
Electromagnetic design based on conventional magnetic circuit laws
Structural design based on {Structural mass in direct-drive permanent magnet electrical generators by
McDonald,A.S. et al. IET Renewable Power Generation(2008),2(1):3 http://dx.doi.org/10.1049/iet-rpg:20070071 """

import pandas as pd
import numpy as np
import openmdao.api as om


def shell_constant(R, t, l, x, v, E):

    Lambda = (3 * (1 - v ** 2) / (R ** 2 * t ** 2)) ** 0.25
    D = E * t ** 3 / (12 * (1 - v ** 2))
    C_14 = (np.sinh(Lambda * l)) ** 2 + (np.sin(Lambda * l)) ** 2
    C_11 = (np.sinh(Lambda * l)) ** 2 - (np.sin(Lambda * l)) ** 2
    F_2 = np.cosh(Lambda * x) * np.sin(Lambda * x) + np.sinh(Lambda * x) * np.cos(Lambda * x)
    C_13 = np.cosh(Lambda * l) * np.sinh(Lambda * l) - np.cos(Lambda * l) * np.sin(Lambda * l)
    F_1 = np.cosh(Lambda * x) * np.cos(Lambda * x)
    F_4 = np.cosh(Lambda * x) * np.sin(Lambda * x) - np.sinh(Lambda * x) * np.cos(Lambda * x)

    return D, Lambda, C_14, C_11, F_2, C_13, F_1, F_4


def plate_constant(a, b, v, r_o, t, E):

    D = E * t ** 3 / (12 * (1 - v ** 2))
    C_2 = 0.25 * (1 - (b / a) ** 2 * (1 + 2 * np.log10(a / b)))
    C_3 = 0.25 * (b / a) * (((b / a) ** 2 + 1) * np.log10(a / b) + (b / a) ** 2 - 1)
    C_5 = 0.5 * (1 - (b / a) ** 2)
    C_6 = 0.25 * (b / a) * ((b / a) ** 2 - 1 + 2 * np.log10(a / b))
    C_8 = 0.5 * (1 + v + (1 - v) * (b / a) ** 2)
    C_9 = (b / a) * (0.5 * (1 + v) * np.log10(a / b) + 0.25 * (1 - v) * (1 - (b / a) ** 2))
    L_11 = (1 / 64) * (
        1 + 4 * (r_o / a) ** 2 - 5 * (r_o / a) ** 4 - 4 * (r_o / a) ** 2 * (2 + (r_o / a) ** 2) * np.log10(a / r_o)
    )
    L_17 = 0.25 * (1 - 0.25 * (1 - v) * ((1 - (r_o / a) ** 4) - (r_o / a) ** 2 * (1 + (1 + v) * np.log10(a / r_o))))

    return D, C_2, C_3, C_5, C_6, C_8, C_9, L_11, L_17


class LTS_inactive_rotor(om.ExplicitComponent):

    """Estimates overall electromagnetic dimensions and Efficiency of PMSG -arms generator."""

    def setup(self):

        self.add_input("y_sh", units="W", desc="Deflection at the shaft")
        self.add_input("theta_sh", 0.0, units="rad", desc="slope of shaft deflection")
        self.add_input("D_a", 0.0, units="m", desc="armature outer diameter ")
        self.add_input("h_s", 0.0, units="m", desc="Slot height ")
        self.add_input("delta_em", 0.0, units="m", desc="air gap length")

        self.add_input("h_yr", 0.0, units="m", desc="rotor yoke height")
        self.add_input("h_yr_s", 0.0, units="m", desc="rotor yoke disc thickness")
        self.add_input("l_eff_rotor", 0.0, units="m", desc="rotor effective length including end winding")

        self.add_input("t_rdisc", 0.0, units="m", desc="rotor disc thickness")

        self.add_input("E", 2e11, units="Pa", desc="Young's modulus of elasticity")
        self.add_input("v", 0.3, desc="Poisson's ratio")
        self.add_input("g", 9.8106, units="m/s/s", desc="Acceleration due to gravity")
        self.add_input("rho_steel", 0.0, units="kg/m**3", desc="Structural steel mass density")
        self.add_input("rho_Fe", 0.0, units="kg/m**3", desc="Electrical steel mass desnity")
        self.add_input("T_e", 0.0, units="N*m", desc="Electromagnetic torque")

        self.add_output("R_ry", 0.0, units="m", desc="mean radius of the rotor yoke")
        self.add_output("r_o", 0.0, units="m", desc="Outer radius of rotor yoke")
        self.add_output("r_i", 0.0, units="m", desc="inner radius of rotor yoke")

        self.add_input("R_shaft_outer", 0.0, units="m", desc=" Main shaft outer radius")
        # self.add_input("R_nose_outer", 0.0, units="m", desc=" Bedplate nose outer radius")
        self.add_output("W_ry", 0.0, desc=" line load of rotor yoke thickness")
        # self.add_input("Copper", 0.0, units="kg", desc=" Copper mass")
        # self.add_input("mass_SC", 0.0, units="kg", desc=" SC mass")
        self.add_input("Tilt_angle", 0.0, units="deg", desc=" Main shaft tilt angle")

        # Material properties
        self.add_input("Sigma_normal", 0.0, units="Pa", desc="Normal stress ")
        # self.add_input("Sigma_shear", 0.0, units="Pa", desc="Normal stress ")

        # Deflection
        self.add_output("u_ar", 0.0, units="m", desc="rotor radial deflection")
        self.add_output("y_ar", 0.0, units="m", desc="rotor axial deflection")
        self.add_output("U_rotor_radial_constraint", 0.0, units="m", desc="Stator radial deflection contraint")
        self.add_output("U_rotor_axial_constraint", 0.0, units="m", desc="Rotor axial deflection contraint")

        # Mass Outputs

        self.add_input("u_allow_pcent", 0.0, desc="Radial deflection as a percentage of air gap diameter")
        self.add_input("y_allow_pcent", 0.0, desc="Radial deflection as a percentage of air gap diameter")
        # self.add_input("z_allow_deg", 0.0, units="deg", desc="Allowable torsional twist")

        # structural design variables

        # self.add_input("K_rad", desc="Aspect ratio")

        # Material properties

        self.add_output("Structural_mass_rotor", 0.0, units="kg", desc="Rotor mass (kg)")

        self.add_output("twist_r", 0.0, units="deg", desc="torsional twist")

        self.add_output("u_allowable_r", 0.0, units="m", desc="Allowable Radial deflection")
        self.add_output("y_allowable_r", 0.0, units="m", desc="Allowable Radial deflection")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):

        # Unpack inputs
        y_sh = inputs["y_sh"]
        theta_sh = inputs["theta_sh"]
        D_a = inputs["D_a"]
        h_s = inputs["h_s"]
        delta_em = inputs["delta_em"]
        h_yr = inputs["h_yr"]
        h_yr_s = inputs["h_yr_s"]
        l_eff_rotor = inputs["l_eff_rotor"]
        t_rdisc = inputs["t_rdisc"]
        E = inputs["E"]
        v = inputs["v"]
        g = inputs["g"]
        rho_steel = inputs["rho_steel"]
        rho_Fe = inputs["rho_Fe"]
        T_e = inputs["T_e"]
        R_shaft_outer = inputs["R_shaft_outer"]
        # R_nose_outer = inputs["R_nose_outer"]
        # Copper = inputs["Copper"]
        # mass_SC = inputs["mass_SC"]
        Tilt_angle = inputs["Tilt_angle"]
        Sigma_normal = inputs["Sigma_normal"]
        # Sigma_shear = inputs["Sigma_shear"]
        u_allow_pcent = inputs["u_allow_pcent"]
        y_allow_pcent = inputs["y_allow_pcent"]
        # z_allow_deg = inputs["z_allow_deg"]
        # K_rad = inputs["K_rad"]

        # Radial deformation of rotor

        ###################################################### Rotor disc radial deflection#############################################
        outputs["r_o"] = r_o = D_a * 0.5 - h_s

        h_total = h_yr + h_yr_s
        outputs["r_i"] = r_i = r_o - h_total
        outputs["R_ry"] = (r_o + r_i) * 0.5

        L_r = l_eff_rotor + t_rdisc
        D_0, Lambda_0, C_14_0, C_11_0, F_2_0, C_13_0, F_1_0, F_4_0 = shell_constant(r_i, t_rdisc, L_r, 0, v, E)
        D_L, Lambda_L, C_14_L, C_11_L, F_2_L, C_13_L, F_1_L, F_4_L = shell_constant(r_i, t_rdisc, L_r, L_r, v, E)

        f_d_denom1 = (
            r_i / (E * (r_i ** 2 - (R_shaft_outer) ** 2)) * ((1 - v) * r_i ** 2 + (1 + v) * (R_shaft_outer) ** 2)
        )

        f_d_denom2 = (
            t_rdisc
            / (2 * D_0 * (Lambda_0) ** 3)
            * (C_14_0 / (2 * C_11_0) * F_2_0 - C_13_0 / C_11_0 * F_1_0 - 0.5 * F_4_0)
        )

        f = Sigma_normal * r_i ** 2 * t_rdisc / (E * (h_total) * (f_d_denom1 + f_d_denom2))

        u_d = (
            f / (D_L * (Lambda_L) ** 3) * ((C_14_L / (2 * C_11_L) * F_2_L - C_13_L / C_11_L * F_1_L - 0.5 * F_4_L))
            + y_sh
        )

        outputs["u_ar"] = (Sigma_normal * r_i ** 2) / (E * (h_total)) - u_d

        outputs["Structural_mass_rotor"] = (
            rho_steel
            * np.pi
            * (((r_i) ** 2 - (R_shaft_outer) ** 2) * t_rdisc + ((r_i + h_yr_s) ** 2 - (r_i ** 2)) * l_eff_rotor)
        )

        # active_mass = rho_Fe * np.pi * ((r_o) ** 2 - (r_i + h_yr_s) ** 2) * L_r
        # Total_mass_rotor = active_mass + outputs["Structural_mass_rotor"]
        # M_r = Total_mass_rotor * g * l_eff_rotor / 2
        # delta_u = 6 * M_r * (1 - v ** 2) * (r_o) / (E * (1 + v) * t_rdisc ** 3)

        outputs["u_ar"] = abs(outputs["u_ar"] + y_sh)

        outputs["u_allowable_r"] = delta_em * u_allow_pcent / 100

        outputs["U_rotor_radial_constraint"] = outputs["u_allowable_r"] - outputs["u_ar"]

        ###################################################### Electromagnetic design#############################################
        # return D,C_2,C_3,C_5,C_6,C_8,C_9,L_11,L_17
        # axial deformation of rotor
        W_back_iron = plate_constant(
            r_i + h_yr_s + h_yr * 0.5,
            R_shaft_outer,
            v,
            +h_yr_s + h_yr * 0.5,
            t_rdisc,
            E,
        )
        W_ssteel = plate_constant(
            r_i + h_yr_s * 0.5,
            R_shaft_outer,
            v,
            h_yr + r_i + h_yr_s * 0.5,
            t_rdisc,
            E,
        )
        # W_cu = plate_constant(
        #    D_a * 0.5 - h_s * 0.5,
        #    R_shaft_outer,
        #    v,
        #    D_a * 0.5 - h_s * 0.5,
        #    t_rdisc,
        #    E,
        # )

        outputs["W_ry"] = rho_Fe * g * np.sin(np.deg2rad(Tilt_angle)) * (L_r - t_rdisc) * h_total

        wr_disc = rho_steel * g * np.sin(np.deg2rad(Tilt_angle)) * t_rdisc

        y_ai1r = (
            -0.5
            * outputs["W_ry"]
            * (r_i + h_yr_s + h_yr * 0.5) ** 4
            / (R_shaft_outer * W_back_iron[0])
            * (W_back_iron[1] * W_back_iron[4] / W_back_iron[3] - W_back_iron[2])
        )
        W_sr = rho_steel * g * np.sin(np.deg2rad(Tilt_angle)) * (L_r - t_rdisc) * h_yr_s
        y_ai2r = (
            -0.5
            * W_sr
            * (r_i + h_yr_s * 0.5) ** 4
            / (R_shaft_outer * W_ssteel[0])
            * (W_ssteel[1] * W_ssteel[4] / W_ssteel[3] - W_ssteel[2])
        )
        # W_Cu = np.sin(np.deg2rad(Tilt_angle)) * Copper / (2 * np.pi * (D_a * 0.5 - h_s * 0.5))
        # y_ai3r = (
        #    -W_Cu * (D_a * 0.5 - h_s * 0.5) ** 4 / (R_shaft_outer * W_cu[0]) * (W_cu[1] * W_cu[4] / W_cu[3] - W_cu[2])
        # )

        a_ii = r_o - h_total
        M_rb = (
            -wr_disc
            * a_ii ** 2
            / W_ssteel[5]
            * (W_ssteel[6] * 0.5 / (a_ii * R_shaft_outer) * (a_ii ** 2 - R_shaft_outer ** 2) - W_ssteel[8])
        )
        Q_b = wr_disc * 0.5 / R_shaft_outer * (a_ii ** 2 - R_shaft_outer ** 2)

        y_aiir = (
            M_rb * a_ii ** 2 / W_ssteel[0] * W_ssteel[1]
            + Q_b * a_ii ** 3 / W_ssteel[0] * W_ssteel[2]
            - wr_disc * a_ii ** 4 / W_ssteel[0] * W_ssteel[7]
        )

        # I = np.pi * 0.25 * (r_i ** 4 - R_shaft_outer ** 4)
        # F_ecc           = Sigma_normal*2*pi*K_rad*r_g**3
        # M_ar             = F_ecc*L_r*0.5

        outputs["y_ar"] = y_ai1r + y_ai2r + y_aiir + (r_i + h_yr + h_yr_s) * theta_sh  # +M_ar*L_r**2*0/(2*E*I)

        outputs["y_allowable_r"] = l_eff_rotor * y_allow_pcent / 100
        # Torsional deformation of rotor
        J_dr = (1 / 32) * np.pi * (r_o ** 4 - R_shaft_outer ** 4)

        J_cylr = (1 / 32) * np.pi * (r_o ** 4 - r_i ** 4)

        G = 0.5 * E / (1 + v)

        outputs["twist_r"] = 180 / np.pi * T_e / G * (t_rdisc / J_dr + (l_eff_rotor - t_rdisc) / J_cylr)

        outputs["Structural_mass_rotor"] = (
            rho_steel
            * np.pi
            * (((r_i) ** 2 - (R_shaft_outer) ** 2) * t_rdisc + ((r_i + h_yr_s) ** 2 - (r_i ** 2)) * l_eff_rotor)
        )

        outputs["U_rotor_axial_constraint"] = outputs["y_allowable_r"] - outputs["y_ar"]


class LTS_inactive_stator(om.ExplicitComponent):
    """Estimates overall electromagnetic dimensions and Efficiency of PMSG -arms generator."""

    def setup(self):

        self.add_input("R_nose_outer", 0.0, units="m", desc="Nose outer radius ")

        self.add_input("y_bd", units="W", desc="Deflection of the bedplate")
        self.add_input("theta_bd", 0.0, units="m", desc="Slope at the bedplate")

        self.add_input("T_e", 0.0, units="N*m", desc="Electromagnetic torque ")
        self.add_input("D_sc", 0.0, units="m", desc="field coil diameter")
        self.add_input("delta_em", 0.0, units="m", desc="air gap length")
        self.add_input("h_ys", 0.0, units="m", desc="Stator yoke height ")
        # field coil parameters
        self.add_input("h_sc", 0.0, units="m", desc="SC coil height")
        self.add_input("l_eff_stator", 0.0, units="m", desc="stator effective length including end winding")

        self.add_output("r_is", 0.0, units="m", desc="inner radius of stator disc")
        self.add_output("r_os", 0.0, units="m", desc="outer radius of stator disc")
        self.add_output("R_sy", 0.0, units="m", desc="Stator yoke height ")

        # structural design variables
        self.add_input("t_sdisc", 0.0, units="m", desc="stator disc thickness")

        self.add_input("E", 2e11, units="Pa", desc="Young's modulus of elasticity")
        self.add_input("v", 0.3, desc="Poisson's ratio")
        self.add_input("g", 9.8106, units="m/s/s", desc="Acceleration due to gravity")
        self.add_input("rho_steel", 0.0, units="kg/m**3", desc="Structural steel mass density")

        self.add_input("mass_SC", 0.0, units="kg", desc=" SC mass")
        self.add_input("Tilt_angle", 0.0, units="deg", desc=" Main shaft tilt angle")

        self.add_output("W_sy", 0.0, desc=" line load of stator yoke thickness")
        # self.add_input("I_sc", 0.0, units="A", desc="SC current ")

        # Material properties
        self.add_input("Sigma_normal", 0.0, units="Pa", desc="Normal stress ")
        # self.add_input("Sigma_shear", 0.0, units="Pa", desc="Normal stress ")

        self.add_output("U_radial_stator", 0.0, units="m", desc="stator radial deflection")
        self.add_output("U_axial_stator", 0.0, units="m", desc="stator axial deflection")
        self.add_output("U_stator_radial_constraint", 0.0, units="m", desc="Stator radial deflection contraint")
        self.add_output("U_stator_axial_constraint", 0.0, units="m", desc="Stator axial deflection contraint")
        # self.add_input("perc_allowable_radial", 0.0, desc=" Allowable radial % deflection ")
        # self.add_input("perc_allowable_axial", 0.0, desc=" Allowable axial % deflection ")

        self.add_input("Structural_mass_rotor", 0.0, units="kg", desc="rotor disc mass")
        self.add_output("Structural_mass_stator", 0.0, units="kg", desc="Stator mass (kg)")
        self.add_output("mass_total", 0.0, units="kg", desc="stator disc mass")

        # self.add_input("K_rad", desc="Aspect ratio")

        self.add_output("u_as", 0.0, units="m", desc="Radial deformation")
        self.add_output("y_as", 0.0, units="m", desc="Axial deformation")
        self.add_output("twist_s", 0.0, units="deg", desc="Stator torsional twist")

        self.add_input("u_allow_pcent", 0.0, desc="Radial deflection as a percentage of air gap diameter")
        self.add_input("y_allow_pcent", 0.0, desc="Radial deflection as a percentage of air gap diameter")
        # self.add_input("z_allow_deg", 0.0, units="deg", desc="Allowable torsional twist")

        self.add_output(
            "u_allowable_s", 0.0, units="m", desc="Allowable Radial deflection as a percentage of air gap diameter"
        )
        self.add_output(
            "y_allowable_s", 0.0, units="m", desc="Allowable Axial deflection as a percentage of air gap diameter"
        )

    def compute(self, inputs, outputs):
        # Unpack inputs
        R_nose_outer = inputs["R_nose_outer"]
        y_bd = inputs["y_bd"]
        theta_bd = inputs["theta_bd"]
        T_e = inputs["T_e"]
        D_sc = inputs["D_sc"]
        delta_em = inputs["delta_em"]
        h_ys = inputs["h_ys"]
        h_sc = inputs["h_sc"]
        l_eff_stator = inputs["l_eff_stator"]
        t_sdisc = inputs["t_sdisc"]
        E = inputs["E"]
        v = inputs["v"]
        g = inputs["g"]
        rho_steel = inputs["rho_steel"]
        mass_SC = inputs["mass_SC"]
        Tilt_angle = inputs["Tilt_angle"]
        Sigma_normal = inputs["Sigma_normal"]
        # Sigma_shear = inputs["Sigma_shear"]
        # perc_allowable_radial = inputs["perc_allowable_radial"]
        # perc_allowable_axial = inputs["perc_allowable_axial"]
        Structural_mass_rotor = inputs["Structural_mass_rotor"]
        # K_rad = inputs["K_rad"]
        u_allow_pcent = inputs["u_allow_pcent"]
        y_allow_pcent = inputs["y_allow_pcent"]
        # z_allow_deg = inputs["z_allow_deg"]

        # Assign values to universal constants

        # Radial deformation of Stator
        L_s = l_eff_stator + t_sdisc
        outputs["r_os"] = r_os = D_sc * 0.5 + h_sc + 0.25 + h_ys
        outputs["r_is"] = r_is = r_os - h_ys
        outputs["R_sy"] = (r_os + r_is) * 0.5
        R_s = r_is

        D_0, Lambda_0, C_14_0, C_11_0, F_2_0, C_13_0, F_1_0, F_4_0 = shell_constant(R_s, t_sdisc, L_s, 0, v, E)
        D_L, Lambda_L, C_14_L, C_11_L, F_2_L, C_13_L, F_1_L, F_4_L = shell_constant(R_s, t_sdisc, L_s, L_s, v, E)
        f_d_denom1 = (
            R_s / (E * ((R_s) ** 2 - (R_nose_outer) ** 2)) * ((1 - v) * R_s ** 2 + (1 + v) * (R_nose_outer) ** 2)
        )
        f_d_denom2 = (
            t_sdisc
            / (2 * D_0 * (Lambda_0) ** 3)
            * (C_14_0 / (2 * C_11_0) * F_2_0 - C_13_0 / C_11_0 * F_1_0 - 0.5 * F_4_0)
        )
        f = Sigma_normal * (R_s) ** 2 * t_sdisc / (E * (h_ys) * (f_d_denom1 + f_d_denom2))
        outputs["u_as"] = (
            (Sigma_normal * (R_s) ** 2) / (E * (h_ys))
            - f / (D_L * (Lambda_L) ** 3) * ((C_14_L / (2 * C_11_L) * F_2_L - C_13_L / C_11_L * F_1_L - 0.5 * F_4_L))
            + y_bd
        )

        outputs["u_as"] += y_bd

        outputs["u_allowable_s"] = delta_em * u_allow_pcent / 100

        outputs["U_stator_radial_constraint"] = outputs["u_allowable_s"] - outputs["u_as"]

        ###################################################### Electromagnetic design#############################################

        # axial deformation of stator

        W_ssteel = plate_constant(
            R_s + h_ys * 0.5,
            R_nose_outer,
            v,
            R_s + h_ys * 0.5,
            t_sdisc,
            E,
        )
        W_sc = plate_constant(
            D_sc * 0.5 + h_sc * 0.5,
            R_nose_outer,
            v,
            D_sc * 0.5 + h_sc * 0.5,
            t_sdisc,
            E,
        )

        W_is = rho_steel * g * np.sin(np.deg2rad(Tilt_angle)) * (L_s - t_sdisc) * h_ys
        y_ai1s = (
            -W_is
            * (0.5 * h_ys + R_s) ** 4
            / (R_nose_outer * W_ssteel[0])
            * (W_ssteel[1] * W_ssteel[4] / W_ssteel[3] - W_ssteel[2])
        )

        W_field = np.sin(np.deg2rad(Tilt_angle)) * mass_SC / (2 * np.pi * (D_sc * 0.5 + h_sc * 0.5))
        y_ai2s = (
            -W_field
            * (D_sc * 0.5 + h_sc * 0.5) ** 4
            / (R_nose_outer * W_sc[0])
            * (W_sc[1] * W_sc[4] / W_sc[3] - W_sc[2])
        )

        w_disc_s = rho_steel * g * np.sin(np.deg2rad(Tilt_angle)) * t_sdisc

        a_ii = R_s
        r_oii = R_nose_outer
        M_sb = (
            -w_disc_s
            * a_ii ** 2
            / W_ssteel[5]
            * (W_ssteel[6] * 0.5 / (a_ii * R_nose_outer) * (a_ii ** 2 - r_oii ** 2) - W_ssteel[8])
        )
        Q_sb = w_disc_s * 0.5 / R_nose_outer * (a_ii ** 2 - r_oii ** 2)

        y_aiis = (
            M_sb * a_ii ** 2 / W_ssteel[0] * W_ssteel[1]
            + Q_sb * a_ii ** 3 / W_ssteel[0] * W_ssteel[2]
            - w_disc_s * a_ii ** 4 / W_ssteel[0] * W_ssteel[7]
        )

        # I = np.pi * 0.25 * (R_s ** 4 - (R_nose_outer) ** 4)
        # F_ecc           = inputs['Sigma_normal']*2*np.pi*inputs['K_rad']*inputs['r_g']**2
        # M_as             = F_ecc*L_s*0.5
        outputs["y_as"] = y_ai1s + y_ai2s + y_aiis + (R_s + h_ys * 0.5) * theta_bd  # M_as*L_s**2*0/(2*E*I)

        outputs["y_allowable_s"] = L_s * y_allow_pcent / 100

        # Torsional deformation of stator
        J_ds = (1 / 32) * np.pi * ((r_os) ** 4 - R_nose_outer ** 4)

        J_cyls = (1 / 32) * np.pi * ((r_os) ** 4 - r_is ** 4)

        G = 0.5 * E / (1 + v)

        outputs["twist_s"] = 180.0 / np.pi * T_e * L_s / G * (t_sdisc / J_ds + (L_s - t_sdisc) / J_cyls)

        outputs["Structural_mass_stator"] = rho_steel * (
            np.pi * ((R_s + h_ys * 0.5) ** 2 - (R_nose_outer) ** 2) * t_sdisc
            + np.pi * (r_os ** 2 - r_is ** 2) * l_eff_stator
        )

        outputs["U_stator_axial_constraint"] = outputs["y_allowable_s"] - outputs["y_as"]

        outputs["mass_total"] = outputs["Structural_mass_stator"] + Structural_mass_rotor


class LTS_Outer_Rotor_Structural(om.Group):
    def setup(self):
        #        self.linear_solver = lbgs = om.LinearBlockJac() #om.LinearBlockGS()
        #        self.nonlinear_solver = nlbgs = om.NonlinearBlockGS()
        #        nlbgs.options["maxiter"] = 3
        #        nlbgs.options["atol"] = 1e-2
        #        nlbgs.options["rtol"] = 1e-8
        #        nlbgs.options["iprint"] = 2

        ivcs = om.IndepVarComp()
        ivcs.add_output("y_sh", units="W", desc="Deflection at the shaft")
        ivcs.add_output("theta_sh", 0.0, units="rad", desc="slope of shaft deflection")
        ivcs.add_output("y_bd", units="W", desc="Deflection of the bedplate")
        ivcs.add_output("theta_bd", 0.0, units="m", desc="Slope at the bedplate")
        ivcs.add_output("R_shaft_outer", 0.0, units="m", desc=" Main shaft outer radius")
        ivcs.add_output("R_nose_outer", 0.0, units="m", desc=" Bedplate nose outer radius")
        ivcs.add_output("Tilt_angle", 0.0, units="deg", desc=" Main shaft tilt angle")
        ivcs.add_output("h_yr_s", 0.0, units="m", desc="rotor yoke thickness")
        ivcs.add_output("h_ys", 0.0, units="m", desc="stator yoke thickness")
        ivcs.add_output("t_rdisc", 0.0, units="m", desc="rotor disc thickness")
        ivcs.add_output("t_sdisc", 0.0, units="m", desc="stator disc thickness")
        ivcs.add_output("E", 2e11, units="Pa", desc="Young's modulus of elasticity")
        ivcs.add_output("v", 0.3, desc="Poisson's ratio")
        ivcs.add_output("g", 9.8106, units="m/s/s", desc="Acceleration due to gravity")
        ivcs.add_output("rho_steel", 0.0, units="kg/m**3", desc="Structural steel mass density")
        ivcs.add_output("u_allow_pcent", 0.0, desc="Radial deflection as a percentage of air gap diameter")
        ivcs.add_output("y_allow_pcent", 0.0, desc="Radial deflection as a percentage of air gap diameter")
        ivcs.add_output("z_allow_deg", 0.0, units="deg", desc="Allowable torsional twist")
        ivcs.add_output("perc_allowable_radial", 0.0, desc=" Allowable radial % deflection ")
        ivcs.add_output("perc_allowable_axial", 0.0, desc=" Allowable axial % deflection ")

        self.add_subsystem("ivcs", ivcs, promotes=["*"])
        self.add_subsystem("sys1", LTS_inactive_rotor(), promotes=["*"])
        self.add_subsystem("sys2", LTS_inactive_stator(), promotes=["*"])


if __name__ == "__main__":

    prob = om.Problem()
    prob.model = LTS_Outer_Rotor_Structural()

    prob.driver = om.ScipyOptimizeDriver()  # pyOptSparseDriver()
    prob.driver.options["optimizer"] = "SLSQP"  #'COBYLA'
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

    prob.model.add_design_var("h_yr_s", lower=0.0250, upper=0.5, ref=0.3)
    prob.model.add_design_var("h_ys", lower=0.025, upper=0.6, ref=0.35)
    prob.model.add_design_var("t_rdisc", lower=0.025, upper=0.5, ref=0.3)
    prob.model.add_design_var("t_sdisc", lower=0.025, upper=0.5, ref=0.3)
    prob.model.add_objective("mass_total")

    prob.model.add_constraint("U_rotor_radial_constraint", lower=0.01)
    prob.model.add_constraint("U_rotor_axial_constraint", lower=0.01)
    prob.model.add_constraint("U_stator_radial_constraint", lower=0.01)
    prob.model.add_constraint("U_stator_axial_constraint", lower=0.01)

    prob.model.approx_totals(method="fd")

    prob.setup()
    # --- Design Variables ---

    # Initial design variables for a PMSG designed for a 15MW turbine
    prob["Sigma_shear"] = 74.99692029e3
    prob["Sigma_normal"] = 378.45123826e3
    prob["T_e"] = 9e06
    prob["l_eff_stator"] = 1.44142189  # rev 1 9.94718e6
    prob["l_eff_rotor"] = 1.2827137
    prob["D_a"] = 7.74736313
    prob["delta_em"] = 0.0199961
    prob["h_s"] = 0.1803019703
    prob["D_sc"] = 7.78735533
    prob["rho_steel"] = 7850
    prob["rho_Fe"] = 7700
    prob["Tilt_angle"] = 90.0
    prob["R_shaft_outer"] = 1.25
    prob["R_nose_outer"] = 0.95
    prob["u_allow_pcent"] = 50
    prob["y_allow_pcent"] = 20
    prob["h_yr"] = 0.1254730934
    prob["h_yr_s"] = 0.025
    prob["h_ys"] = 0.050
    prob["t_rdisc"] = 0.05
    prob["t_sdisc"] = 0.100
    prob["y_bd"] = 0.00
    prob["theta_bd"] = 0.00
    prob["y_sh"] = 0.00
    prob["theta_sh"] = 0.00

    prob["Copper"] = 60e3
    prob["mass_SC"] = 4000

    prob.model.approx_totals(method="fd")

    prob.run_model()
    # prob.run_driver()

    # prob.model.list_outputs(values = True, hierarchical=True)

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
            prob.get_val("t_rdisc", units="mm"),
            prob.get_val("h_yr_s", units="mm"),
            prob.get_val("t_sdisc", units="mm"),
            prob.get_val("h_ys", units="mm"),
            prob.get_val("u_ar", units="mm"),
            prob.get_val("y_ar", units="mm"),
            prob.get_val("u_as", units="mm"),
            prob.get_val("y_as", units="mm"),
            prob.get_val("Structural_mass_rotor", units="t"),
            prob.get_val("Structural_mass_stator", units="t"),
            prob.get_val("mass_total", units="t"),
        ],
        "Limit": [
            "",
            "",
            "",
            "",
            prob.get_val("u_allowable_r", units="mm"),
            prob.get_val("y_allowable_r", units="mm"),
            prob.get_val("u_allowable_s", units="mm"),
            prob.get_val("y_allowable_s", units="mm"),
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
    # print(raw_data)
    df = pd.DataFrame(raw_data, columns=["Parameters", "Values", "Limit", "Units"])

    print(df)

    df.to_excel("Optimized_structure_LTSG_MW.xlsx")