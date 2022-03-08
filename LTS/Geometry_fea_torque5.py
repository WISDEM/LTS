# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 12:28:24 2021

@author: lsethura
"""
# Wound Copper Coil with an Iron Core
# David Meeker
# dmeeker@ieee.org
#
# This program consider an axisymmetric magnetostatic problem
# of a cylindrical coil with an axial length of 100 mm, an
# inner radius of 50 mm, and an outer radius of 100 mm.  The
# coil has 200 turns and the coil current is 20 Amps. There is
# an iron bar 80 mm long with a radius of 10 mm centered co-
# axially with the coil.  The objective of the analysis is to
# determine the flux density at the center of the iron bar,
# and to plot the field along the r=0 axis. This analysis
# defines a nonlinear B-H curve for the iron and employs an
# asymptotic boundary condition to approximate an "open"
# boundary condition on the edge of the solution domain.

import femm
import numpy as np
import openmdao.api as om

# The package must be initialized with the openfemm command.


def run_post_process(D_a, radius_sc, h_sc, slot_radius, theta_p_r, alpha_r, beta_r, n):

    theta_p_d = np.rad2deg(theta_p_r)

    femm.mo_addcontour((D_a / 2 + (radius_sc - D_a / 2) * 0.5) * np.cos(0), (D_a / 2 + (radius_sc - D_a / 2) * 0.5) * np.sin(0))
    femm.mo_addcontour(
        (D_a / 2 + (radius_sc - D_a / 2) * 0.5) * np.cos(theta_p_r), (D_a / 2 + (radius_sc - D_a / 2) * 0.5) * np.sin(theta_p_r)
    )
    femm.mo_bendcontour(theta_p_d, 0.25)
    femm.mo_makeplot(1, 1500, "gap_" + str(n) + ".csv", 1)
    femm.mo_makeplot(2, 100, "B_r_normal" + str(n) + ".csv", 1)
    femm.mo_makeplot(3, 100, "B_t_normal" + str(n) + ".csv", 1)
    femm.mo_clearcontour()

    #        femm.mo_addcontour((radius_sc)*np.cos(beta_r),(radius_sc)*np.sin(beta_r))
    #        femm.mo_addcontour((radius_sc)*np.cos(alpha_r),(radius_sc)*np.sin(alpha_r))
    #        femm.mo_addcontour((radius_sc+h_sc)*np.cos(alpha_r),(radius_sc+h_sc)*np.sin(alpha_r))
    #        femm.mo_addcontour((radius_sc+h_sc)*np.cos(beta_r),(radius_sc+h_sc)*np.sin(beta_r))
    #        femm.mo_addcontour((radius_sc)*np.cos(beta_r),(radius_sc)*np.sin(beta_r))

    femm.mo_selectblock(
        (radius_sc + h_sc * 0.5) * np.cos(alpha_r + (beta_r - alpha_r) * 0.5),
        (radius_sc + h_sc * 0.5) * np.sin(alpha_r + (beta_r - alpha_r) * 0.5),
    )
    femm.mo_smooth("off")
    numelm = femm.mo_numelements()
    bcoil_area = []
    for k in range(1, numelm):
        p1, p2, p3, x, y, a, g = femm.mo_getelement(k)
        bx, by = femm.mo_getb(x, y)
        bcoil_area.append((bx**2 + by**2) ** 0.5)

    B_coil_max = max(bcoil_area)
    femm.mo_clearblock()

    radius_eps = 1e-3
    femm.mo_addcontour((slot_radius - radius_eps) * np.cos(0), (slot_radius - radius_eps) * np.sin(0))
    femm.mo_addcontour((slot_radius - radius_eps) * np.cos(theta_p_r), (slot_radius - radius_eps) * np.sin(theta_p_r))
    femm.mo_bendcontour(theta_p_d, 0.25)

    femm.mo_makeplot(1, 1500, "core_" + str(n) + ".csv", 1)
    femm.mo_clearcontour()

    B_rymax = np.loadtxt("core_" + str(n) + ".csv")[:,1].max()
    B_g_peak = np.loadtxt("gap_" + str(n) + ".csv")[:,1].max()
    B_r_normal = np.loadtxt("B_r_normal" + str(n) + ".csv")
    B_t_normal = np.loadtxt("B_t_normal" + str(n) + ".csv")

    circ = B_r_normal[-1,0]
    delta_L = np.diff(B_r_normal[:,0])[0]

    force = np.sum((B_r_normal[:,1]) ** 2 - (B_t_normal[:,1]) ** 2) * delta_L
    sigma_n = abs(1 / (8 * np.pi * 1e-7) * force) / circ

    # print ((radius_sc)*np.cos(beta_r),(radius_sc)*np.sin(beta_r),(radius_sc)*np.cos(alpha_r),(radius_sc)*np.sin(alpha_r),(radius_sc+h_sc)*np.cos(alpha_r),(radius_sc+h_sc)*np.sin(alpha_r),(radius_sc+h_sc)*np.cos(beta_r),(radius_sc+h_sc)*np.sin(beta_r),(radius_sc)*np.cos(beta_r),(radius_sc)*np.sin(beta_r))

    return B_g_peak, B_rymax, B_coil_max, sigma_n


# Define the problem type.  Magnetostatic; Units of mm; Axisymmetric;
# Precision of 10^(-8) for the linear solver; a placeholder of 0 for
# the depth dimension, and an angle constraint of 30 degrees


def B_r_B_t(D_a, l_s, p1,delta_em, theta_p_r, I_s, theta_b_t, theta_b_s, layer_1, layer_2, Y_q, N_c, tau_p):

    theta_p_d = np.rad2deg(theta_p_r)
    
    
    femm.openfemm(1)
    femm.opendocument("coil_design_new.fem")
    T_elec = -0.5
    femm.mi_modifycircprop("A+", 1, I_s * np.sin(0))
    femm.mi_modifycircprop("D+", 1, I_s * np.sin(1*np.pi / 6))
    femm.mi_modifycircprop("C-", 1, -I_s * np.sin(-4 * np.pi / 6))
    femm.mi_modifycircprop("F-", 1, -I_s * np.sin(-3*np.pi/6))
    femm.mi_modifycircprop("B+", 1, I_s * np.sin(-8 * np.pi / 6))
    femm.mi_modifycircprop("E+", 1, I_s * np.sin(-7 * np.pi / 6))
    femm.mi_modifycircprop("A-", 1, -I_s * np.sin(0))
    femm.mi_modifycircprop("D-", 1, -I_s * np.sin(np.pi / 6))
    femm.mi_saveas("coil_design_new_I1.fem")
    femm.mi_analyze()
    femm.mi_loadsolution()

    femm.mo_addcontour((D_a / 2 + delta_em*0.5) * np.cos(0), (D_a / 2 + delta_em*0.5) * np.sin(0))
    femm.mo_addcontour((D_a / 2 + delta_em*0.5) * np.cos(theta_p_r), (D_a / 2 + delta_em*0.5) * np.sin(theta_p_r))
    femm.mo_bendcontour(theta_p_d, 0.25)
    femm.mo_makeplot(2, 100, "B_r_1.csv", 1)
    femm.mo_makeplot(3, 100, "B_t_1.csv", 1)
    femm.mo_clearcontour()
    femm.mo_close()
    femm.openfemm(1)
    femm.opendocument("coil_design_new_I1.fem")
    pitch = 1

    Phases = ["D+", "C-", "F-", "B+", "E+", "A-", "D-"]
    #Phases = ["F+", "B+", "E+", "A+", "D+", "C+", "F-"]
    N_c_a1    =[2*N_c,4*N_c,4*N_c,4*N_c,4*N_c,4*N_c,2*N_c]

    count = 0
    angle_r = theta_b_t * 0.5 + theta_b_s * 0.5
    delta_theta = theta_b_t + theta_b_s
    for pitch in range(1, Y_q, 2):
        femm.mi_selectlabel(
            layer_2 * np.cos(angle_r + (pitch - 1) * (delta_theta)), layer_2 * np.sin(angle_r + (pitch - 1) * (delta_theta))
        )
        femm.mi_selectlabel(layer_2 * np.cos(angle_r + pitch * delta_theta), layer_2 * np.sin(angle_r + pitch * delta_theta))
        femm.mi_setblockprop("20 SWG", 1, 1, Phases[count], 0, 8, N_c)
        #femm.mi_setblockprop("20 SWG", 1, 1, Phases[count], 0, 8, N_c_a1[count])
        femm.mi_clearselected()
        count = count + 1

    count = 0
    angle_r = theta_b_t * 0.5 + theta_b_s * 0.5
    delta_theta = theta_b_t + theta_b_s
    for pitch in range(1, Y_q, 2):
        femm.mi_selectlabel(
            layer_1 * np.cos(angle_r + (pitch - 1) * (delta_theta)), layer_1 * np.sin(angle_r + (pitch - 1) * (delta_theta))
        )
        femm.mi_selectlabel(layer_1 * np.cos(angle_r + pitch * delta_theta), layer_1 * np.sin(angle_r + pitch * delta_theta))
        femm.mi_setblockprop("20 SWG", 1, 1, Phases[count + 1], 0, 8, N_c)
        #femm.mi_setblockprop("20 SWG", 1, 1, Phases[count + 1], 0, 8, N_c_a1[count+1])
        femm.mi_clearselected()
        count = count + 1

    femm.mi_modifycircprop("D+", 1, I_s * np.sin(T_elec + np.pi / 6))
    femm.mi_modifycircprop("C-", 1, -I_s * np.sin(T_elec  -4 * np.pi / 6))
    femm.mi_modifycircprop("F-", 1, -I_s * np.sin(T_elec  - 3*np.pi/6))
    femm.mi_modifycircprop("B+", 1, I_s * np.sin(T_elec  - 8 * np.pi / 6))
    femm.mi_modifycircprop("E+", 1,I_s * np.sin(T_elec  - 7 * np.pi / 6))
    femm.mi_modifycircprop("A-", 1, -I_s * np.sin(T_elec))
    femm.mi_modifycircprop("D-", 1, -I_s * np.sin(T_elec+np.pi / 6))
    
    femm.mi_saveas("coil_design_new_I2.fem")
    femm.mi_analyze()
    femm.mi_loadsolution()
    femm.mo_addcontour((D_a / 2 + delta_em*0.5) * np.cos(0), (D_a / 2 + delta_em*0.5) * np.sin(0))
    femm.mo_addcontour((D_a / 2 + delta_em*0.5) * np.cos(theta_p_r), (D_a / 2 + delta_em*0.5) * np.sin(theta_p_r))
    femm.mo_bendcontour(theta_p_d, 0.25)
    femm.mo_makeplot(2, 100, "B_r_2.csv", 1)
    femm.mo_makeplot(3, 100, "B_t_2.csv", 1)
    femm.mo_clearcontour()
    femm.mo_close()

    B_r_1 = np.loadtxt("B_r_1.csv")
    B_t_1 = np.loadtxt("B_t_1.csv")
    B_r_2 = np.loadtxt("B_r_2.csv")
    B_t_2 = np.loadtxt("B_t_2.csv")
    delta_L = np.diff(B_r_1[:,0])[0]
    circ = B_r_1[-1,0]

    force = np.array([np.sum(B_r_1[:,1] * B_t_1[:,1]), np.sum(B_r_2[:,1] * B_t_2[:,1]) ]) * delta_L
    sigma_t = abs(1 / (4 * np.pi * 1e-7) * force) / circ
    torque = np.pi / 2 * sigma_t * D_a ** 2 * l_s
    
    print (torque[0]/1e6,torque[1]/1e6)

    return torque.mean(), sigma_t.mean()


class FEMM_Geometry(om.ExplicitComponent):

    def setup(self):
        self.add_discrete_input("q", 2, desc="slots_per_pole")
        self.add_discrete_input("m", 6, desc="number of phases")

        self.add_input("l_s", 0.0, units="m", desc="Stack length ")
        self.add_input("alpha", 0.0, units="deg", desc="Start angle of field coil")
        self.add_input("beta", 0.0, units="deg", desc="End angle of field coil")
        self.add_input("h_sc", 0.0, units="m", desc="SC coil height")
        self.add_input("p1", 0.0, desc="Pole pairs ")
        self.add_input("D_a", 0.0, units="m", desc="armature diameter ")
        self.add_input("h_s", 0.0, units="m", desc="Slot height ")
        self.add_input("h_yr", 0.0, units="m", desc="rotor yoke height")
        #self.add_input("Y", 0.0, desc="coil pitch")
        self.add_input("D_sc", 0.0, units="m", desc="field coil diameter ")
        #self.add_input("Dia_sc", 0.0, units="m", desc="field coil diameter")
        self.add_input("I_sc", 0.0, units="A", desc="SC current ")
        #self.add_input("N_s", 0.0, desc="Number of turns per phase in series")
        self.add_input("N_sc", 0.0, desc="Number of turns of SC field coil")
        self.add_input("N_c", 0.0, desc="Number of turns per coil")
        self.add_input("delta_em", 0.0, units="m", desc="airgap length ")
        self.add_input("I_s", 0.0, units="A", desc="Generator output phase current")

        self.add_output("B_g", 0.0, desc="Peak air gap flux density ")
        self.add_output("B_rymax", 0.0, desc="Peak Rotor yoke flux density")
        self.add_output("B_coil_max", 0.0, desc="Peak flux density in the field coils")
        #self.add_output("I_sc_out", 0.0, units="A", desc="SC current ")
        #self.add_output("N_sc_out", 0.0, desc="Number of turns of SC field coil")
        self.add_output("Torque_actual", 0.0, units="N*m", desc="Shear stress actual")
        self.add_output("Sigma_shear", 0.0, units="Pa", desc="Shear stress")
        self.add_output("Sigma_normal", 0.0, units="Pa", desc="Normal stress")
        self.add_output("margin_I_c",0.0,units="A", desc="Critical current margin")
        self.add_output("Critical_current_limit",0.0,units="A", desc="Critical current limit") 
        self.add_output("Coil_max_limit",0.0, desc="Critical coil flux density limit") 
        self.add_output("a_m", 0.0, units="m", desc="Coil separation distance")
        self.add_output("W_sc", 0.0, units="m", desc="SC coil width")
        self.add_output("Outer_width", 0.0, units="m", desc="Coil outer width")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        # Unpack variables
        l_s = float(inputs["l_s"])
        alpha_d = float(inputs["alpha"])
        beta_d = float(inputs["beta"])
        alpha_r = np.deg2rad( alpha_d )
        beta_r = np.deg2rad( beta_d )
        h_sc = float(inputs["h_sc"])
        p1 = float(inputs["p1"])
        D_a = float(inputs["D_a"])
        h_yr = float(inputs["h_yr"])
        h_s = float(inputs["h_s"])
        q = discrete_inputs["q"]
        m = discrete_inputs["m"]
        #Y = float(inputs["Y"])
        D_sc = float(inputs["D_sc"])
        #Dia_sc = float(inputs["Dia_sc"])
        N_sc = int(inputs["N_sc"])   # 
        I_sc = float(inputs["I_sc"])
        N_c = int(inputs["N_c"])
        delta_em = float(inputs["delta_em"])
        I_s = float(inputs["I_s"])
        radius_sc = D_sc / 2
        tau_p = np.pi * (radius_sc * 2 + 2 * h_sc) / (2 * p1)
        theta_p_r = tau_p / (radius_sc + h_sc)
        theta_p_d = np.rad2deg(theta_p_r)
        
        if (alpha_d <=0) or (beta_d>theta_p_d*0.5):
           outputs["B_g"] =7
           outputs["B_coil_max"]=12
           outputs["B_rymax"]=5
           outputs["Torque_actual"]=50e+06
           outputs["Sigma_shear"]=300000
           outputs["Sigma_normal"]=200000 
        else:
            femm.openfemm(1)
            femm.newdocument(0)
            femm.mi_probdef(0, "meters", "planar", 1.0e-8, l_s, 30)
    
            slot_radius = D_a * 0.5 - h_s
            yoke_radius = slot_radius - h_yr
            h42 = D_a / 2 - 0.5 * h_s
            Slots = 2 * q * m * p1
            Slots_pp = q * m
            Y_q = int(Slots / (2 * p1))
            tau_s = np.pi * D_a / Slots
    
            #bs_taus = 0.45 #UNUSED
            b_s = 0.45 * tau_s
            b_t = tau_s - b_s
            theta_b_s = b_s / (D_a * 0.5)
            #theta_tau_s = tau_s / (D_a * 0.5) # UNUSED
            theta_b_t = (b_t) / (D_a * 0.5)
    
            
            tau_p = np.pi * (radius_sc * 2 + 2 * h_sc) / (2 * p1)
            Current = 0
    
            
            theta_p_d = np.rad2deg(theta_p_r)
            alphap_taup_r = theta_p_r - 2*beta_r
            alphap_taup_angle_r = beta_r + alphap_taup_r - alpha_r
            alphap_taup_angle_d = np.rad2deg(alphap_taup_angle_r)
    
            # Draw the field coil
            m_1=-1/(np.tan(theta_p_r*0.5))       # slope of the tangent line
            x_coord =radius_sc *np.cos(theta_p_r*0.5)
            y_coord =radius_sc *np.sin(theta_p_r*0.5)
            c1       =y_coord-m_1*x_coord
            
            if alpha_r<=1:
                angle=alpha_r
            else:
                angle=np.tan(alpha_r)
            m_2=angle
            c2       =0
            
            m_3=m_1    # tangent offset
            x_coord3=(radius_sc+h_sc) *np.cos(theta_p_r*0.5)
            y_coord3=(radius_sc+h_sc) *np.sin(theta_p_r*0.5)
            c3      =y_coord3-m_3*x_coord3
 
            mlabel=m_1
           
            x_label =(radius_sc+h_sc*0.5) *np.cos(theta_p_r*0.5)
            y_label =(radius_sc+h_sc*0.5) *np.sin(theta_p_r*0.5)
            clabel=y_label-mlabel*x_label
           
           
            mlabel2=np.tan(alpha_r+(beta_r-alpha_r)*0.5)
            clabel2=0
           
            mlabel3=np.tan(theta_p_r-alpha_r-(beta_r-alpha_r)*0.5)
            clabel3=0
          
            m_6=np.tan(beta_r)
            c6      =0
            
    
            
            x1      =(c1-c2)/(m_2-m_1)
            y1      =m_1*x1+c1
            
            femm.mi_addnode(x1,y1)
            
            m_4     =np.tan(theta_p_r*0.5)
            
            c4      =y1-m_4*x1
            
            x2      =(c3-c4)/(m_4-m_3)
            y2      =m_4*x2+c4
            
            femm.mi_addnode(x2,y2)
            
            x4      =(c1-c6)/(m_6-m_1)
            y4      =m_1*x4+c1
            
                   
            m_5=np.tan(theta_p_r*0.5)
            c5      =y4-m_5*x4
            
            x3      =(c3-c5)/(m_5-m_3)
            y3      =m_3*x3+c3
            
            
            
            m_7=np.tan(theta_p_r-alpha_r)
            c7 =0
             
            x5 =(c1-c7)/(m_7-m_1)
            y5 =m_1*x5+c1
            
            m_8=m_4
            c8 =y5-m_8*x5
             
            x6 =(c3-c8)/(m_8-m_3)
            y6 =m_3*x6+c3
             
             
            m_9=np.tan(theta_p_r-beta_r)
            c9  =0
             
            x8 =(c1-c9)/(m_9-m_1)
            y8 =m_1*x8+c1
             
            m_10 =m_5
            c10 =y8-m_10*x8
             
            x7 =(c3-c10)/(m_10-m_3)
            y7 =m_3*x7+c3
            
            xlabel1=(clabel-clabel2)/(mlabel2-mlabel)
            ylabel1=mlabel*xlabel1+clabel
            
            xlabel2=(clabel-clabel3)/(mlabel3-mlabel)
            ylabel2=mlabel*xlabel2+clabel
            
            
             
            
    
            outputs["a_m"]  =2*(np.sqrt((radius_sc*np.cos(theta_p_r*0.5)-x4)**2+((radius_sc*np.sin(theta_p_r*0.5)-y4)**2)))
            outputs["W_sc"] =np.sqrt((x1-x4)**2+(y1-y4)**2)
            outputs["Outer_width"] = outputs["a_m"] + 2 * outputs["W_sc"]
            
            
            femm.mi_addnode(x3,y3)
            femm.mi_addnode(x4,y4)
            femm.mi_addnode(x5,y5)
            femm.mi_addnode(x6,y6)
            femm.mi_addnode(x7,y7)
            femm.mi_addnode(x8,y8)
            femm.mi_addsegment(x1,y1,x2,y2)
            femm.mi_addsegment(x2,y2,x3,y3)
            femm.mi_addsegment(x3,y3,x4,y4)
            femm.mi_addsegment(x4,y4,x1,y1)
            femm.mi_addsegment(x5,y5,x6,y6)
            femm.mi_addsegment(x8,y8,x7,y7)
            femm.mi_addsegment(x5,y5,x8,y8)
            femm.mi_addsegment(x6,y6,x7,y7)
    
    
  
    
            femm.mi_addnode(0, 0)
        
            # Draw the stator slots and Stator coils
    
            femm.mi_addnode(D_a / 2 * np.cos(0), D_a / 2 * np.sin(0))
            femm.mi_selectnode(D_a / 2 * np.cos(0), D_a / 2 * np.sin(0))
            femm.mi_setgroup(1)
            femm.mi_addnode(slot_radius * np.cos(0), slot_radius * np.sin(0))
            femm.mi_selectnode(slot_radius * np.cos(0), slot_radius * np.sin(0))
            femm.mi_setgroup(1)
            femm.mi_addnode(h42 * np.cos(0), h42 * np.sin(0))
            femm.mi_selectnode(h42 * np.cos(0), h42 * np.sin(0))
            femm.mi_setgroup(1)
            femm.mi_addsegment(D_a / 2 * np.cos(0), D_a / 2 * np.sin(0), h42 * np.cos(0), h42 * np.sin(0))
            femm.mi_addsegment(h42 * np.cos(0), h42 * np.sin(0), slot_radius * np.cos(0), slot_radius * np.sin(0))
            femm.mi_selectsegment(h42 * np.cos(0), h42 * np.sin(0))
    
            femm.mi_selectsegment(slot_radius * np.cos(0), slot_radius * np.sin(0))
            femm.mi_setgroup(1)
    
            femm.mi_addarc(slot_radius * np.cos(theta_b_t * 0.5), slot_radius * np.sin(theta_b_t * 0.5), slot_radius, 0, 5, 1)
            femm.mi_selectarcsegment(slot_radius * np.cos(0), slot_radius * np.sin(0))
            femm.mi_setgroup(1)
            femm.mi_selectsegment(D_a / 2 * np.cos(0), D_a / 2 * np.sin(0))
            femm.mi_setgroup(1)
    
            femm.mi_addnode(D_a * 0.5 * np.cos(theta_b_t * 0.5), D_a * 0.5 * np.sin(theta_b_t * 0.5))
            femm.mi_selectnode(D_a * 0.5 * np.cos(theta_b_t * 0.5), D_a * 0.5 * np.sin(theta_b_t * 0.5))
            femm.mi_setgroup(2)
    
            femm.mi_addnode(slot_radius * np.cos(theta_b_t * 0.5), slot_radius * np.sin(theta_b_t * 0.5))
            femm.mi_setgroup(2)
    
            femm.mi_addnode(h42 * np.cos(theta_b_t * 0.5), h42 * np.sin(theta_b_t * 0.5))
            femm.mi_setgroup(2)
    
            femm.mi_addsegment(
                D_a * 0.5 * np.cos(theta_b_t * 0.5),
                D_a * 0.5 * np.sin(theta_b_t * 0.5),
                h42 * np.cos(theta_b_t * 0.5),
                h42 * np.sin(theta_b_t * 0.5),
            )
    
            femm.mi_addsegment(
                h42 * np.cos(theta_b_t * 0.5),
                h42 * np.sin(theta_b_t * 0.5),
                slot_radius * np.cos(theta_b_t * 0.5),
                slot_radius * np.sin(theta_b_t * 0.5),
            )
            femm.mi_selectarcsegment(h42 * np.cos(theta_b_t * 0.5), h42 * np.sin(theta_b_t * 0.5))
            femm.mi_setgroup(2)
    
            femm.mi_selectsegment(h42 * np.cos(theta_b_t * 0.5), h42 * np.sin(theta_b_t * 0.5))
            femm.mi_setgroup(2)
    
            femm.mi_selectsegment(D_a * 0.5 * np.cos(theta_b_t * 0.5), D_a * 0.5 * np.sin(theta_b_t * 0.5))
            femm.mi_setgroup(2)
    
            femm.mi_selectsegment(slot_radius * np.cos(theta_b_t * 0.5), slot_radius * np.sin(theta_b_t * 0.5))
            femm.mi_setgroup(2)
    
            theta_b_s_new = (b_t * 0.5 + b_s) / (D_a * 0.5)
            femm.mi_addnode(D_a * 0.5 * np.cos(theta_b_s_new), D_a * 0.5 * np.sin(theta_b_s_new))
            femm.mi_selectnode(D_a * 0.5 * np.cos(theta_b_s_new), D_a * 0.5 * np.sin(theta_b_s_new))
            femm.mi_setgroup(2)
    
            femm.mi_addnode(h42 * np.cos(theta_b_s_new), h42 * np.sin(theta_b_s_new))
            femm.mi_selectnode(h42 * np.cos(theta_b_s_new), h42 * np.sin(theta_b_s_new))
            femm.mi_setgroup(2)
    
            femm.mi_addsegment(
                D_a * 0.5 * np.cos(theta_b_s_new),
                D_a * 0.5 * np.sin(theta_b_s_new),
                h42 * np.cos(theta_b_s_new),
                h42 * np.sin(theta_b_s_new),
            )
            femm.mi_selectsegment(D_a * 0.5 * np.cos(theta_b_s_new), D_a * 0.5 * np.sin(theta_b_s_new))
            femm.mi_setgroup(2)
    
            femm.mi_addnode(slot_radius * np.cos(theta_b_s_new), slot_radius * np.sin(theta_b_s_new))
            femm.mi_selectnode(slot_radius * np.cos(theta_b_s_new), slot_radius * np.sin(theta_b_s_new))
            femm.mi_setgroup(2)
    
            femm.mi_addsegment(
                slot_radius * np.cos(theta_b_s_new),
                slot_radius * np.sin(theta_b_s_new),
                h42 * np.cos(theta_b_s_new),
                h42 * np.sin(theta_b_s_new),
            )
            femm.mi_selectsegment(slot_radius * np.cos(theta_b_s_new), slot_radius * np.sin(theta_b_s_new))
            femm.mi_setgroup(2)
    
            femm.mi_addarc(
                D_a * 0.5 * np.cos(theta_b_s_new),
                D_a * 0.5 * np.sin(theta_b_s_new),
                D_a * 0.5 * np.cos(theta_b_t * 0.5),
                D_a * 0.5 * np.sin(theta_b_t * 0.5),
                5,
                1,
            )
            femm.mi_selectarcsegment(D_a * 0.5 * np.cos(theta_b_s_new), D_a * 0.5 * np.sin(theta_b_s_new))
            femm.mi_setgroup(2)
    
            femm.mi_addarc(
                h42 * np.cos(theta_b_s_new), h42 * np.sin(theta_b_s_new), h42 * np.cos(theta_b_t * 0.5), h42 * np.sin(theta_b_t * 0.5), 5, 1
            )
            femm.mi_selectarcsegment(h42 * np.cos(theta_b_s_new), h42 * np.sin(theta_b_s_new))
            femm.mi_setgroup(2)
    
            femm.mi_addarc(
                slot_radius * np.cos(theta_b_t * 0.5),
                slot_radius * np.sin(theta_b_t * 0.5),
                slot_radius * np.cos(theta_b_s_new),
                slot_radius * np.sin(theta_b_s_new),
                2,
                1,
            )
            femm.mi_selectarcsegment(slot_radius * np.cos(theta_b_s_new), slot_radius * np.sin(theta_b_s_new))
            femm.mi_setgroup(2)
    
            femm.mi_selectgroup(2)
            angle_d = np.rad2deg(tau_s / (D_a * 0.5))
            femm.mi_copyrotate(0, 0, angle_d, Slots_pp - 1)
    
            #b_t_new = slot_radius * b_t / ((D_a * 0.5)) #UNUSED
    
            femm.mi_addnode(yoke_radius * np.cos(0), yoke_radius * np.sin(0))
    
            # femm.mi_addnode(yoke_radius*np.cos(alpha_r),yoke_radius*np.sin(alpha_r))
            femm.mi_addnode(yoke_radius * np.cos(theta_p_r), yoke_radius * np.sin(theta_p_r))
    
            # femm.mi_addnode(slot_radius*np.cos(theta_p_r),slot_radius*np.sin(theta_p_r))
    
            femm.mi_addarc(
                yoke_radius * np.cos(0),
                yoke_radius * np.sin(0),
                yoke_radius * np.cos(theta_p_r),
                yoke_radius * np.sin(theta_p_r),
                theta_p_d,
                1,
            )
    
            #    femm.mi_addarc(D_a*0.5*np.cos(0),D_a*0.5*np.sin(0),D_a*0.5*np.cos(theta_b_t*0.5),D_a*0.5*np.sin(theta_b_t*0.5),5,1)
            #    femm.mi_selectarcsegment(D_a*0.5*np.cos(0),D_a*0.5*np.sin(0))
            #    femm.mi_setgroup(1)
            #
            #
            femm.mi_addarc(slot_radius, 0, slot_radius * np.cos(theta_b_t * 0.5), slot_radius * np.sin(theta_b_t * 0.5), 5, 1)
            femm.mi_selectarcsegment(slot_radius * np.cos(0), 0)
            femm.mi_setgroup(1)
    
            femm.mi_selectgroup(1)
            angle1_d = np.rad2deg(tau_p / (radius_sc + h_sc) - theta_b_t * 0.5)
            femm.mi_copyrotate(0, 0, angle1_d, 1)
    
            # femm.mi_addarc(slot_radius*np.cos(theta_b_t_new),slot_radius*np.sin(theta_b_t_new),slot_radius*np.cos(theta_p_r),slot_radius*np.sin(theta_p_r),5,1)
    
            # femm.mi_addsegment(slot_radius*np.cos(theta_p_r),slot_radius*np.sin(theta_p_r),D_a/2*np.cos(theta_p_r),D_a/2*np.sin(theta_p_r))
    
            r_o = (radius_sc + h_sc) * 3
    
            # femm.mi_addsegment(D_a/2*np.cos(0),D_a/2*np.sin(0),radius_sc*np.cos(0),radius_sc*np.sin(0))
    
            femm.mi_addnode(r_o * np.cos(0), r_o * np.sin(0))
            femm.mi_addnode(r_o * np.cos(theta_p_r), r_o * np.sin(theta_p_r))
            #
            ## Add some block labels materials properties
            femm.mi_addmaterial("Air", 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0)
            femm.mi_addmaterial("NbTi", 0.6428571428571428571428, 0.6428571428571428571428, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            femm.mi_getmaterial("M-36 Steel")
            femm.mi_getmaterial("20 SWG")
            femm.mi_addcircprop("A1+", I_sc, 1)
            femm.mi_addcircprop("A1-", -1 * I_sc, 1)
            femm.mi_addcircprop("A+", Current, 1)
            femm.mi_addcircprop("A-", Current, 1)
            femm.mi_addcircprop("B+", Current, 1)
            femm.mi_addcircprop("B-", Current, 1)
            femm.mi_addcircprop("C+", Current, 1)
            femm.mi_addcircprop("C-", Current, 1)
            femm.mi_addcircprop("D+", Current, 1)
            femm.mi_addcircprop("D-", Current, 1)
            femm.mi_addcircprop("E+", Current, 1)
            femm.mi_addcircprop("E-", Current, 1)
            femm.mi_addcircprop("F+", Current, 1)
            femm.mi_addcircprop("F-", Current, 1)
    
            femm.mi_addboundprop("Dirichlet", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            femm.mi_addboundprop("apbc1", 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0)
            femm.mi_addboundprop("apbc2", 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0)
            femm.mi_addboundprop("apbc3", 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0)
            femm.mi_addboundprop("apbc4", 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0)
            femm.mi_addboundprop("apbc5", 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0)
    
            femm.mi_addarc(r_o * np.cos(0), r_o * np.sin(0), r_o * np.cos(theta_p_r), r_o * np.sin(theta_p_r), theta_p_d, 1)
            femm.mi_selectarcsegment(r_o * np.cos(0), r_o * np.sin(0))
            femm.mi_setarcsegmentprop(5, "Dirichlet", 0, 5)
    
            femm.mi_addsegment(yoke_radius * np.cos(0), yoke_radius * np.sin(0), slot_radius * np.cos(0), slot_radius * np.sin(0))
            femm.mi_addsegment(D_a * 0.5 * np.cos(0), D_a * 0.5 * np.sin(0), r_o * np.cos(0), r_o * np.sin(0))
            femm.mi_addsegment(0, 0, yoke_radius * np.cos(0), yoke_radius * np.sin(0))
    
            femm.mi_selectsegment(0.0, 0)
            femm.mi_setsegmentprop("apbc1", 100, 0, 0, 6)
            femm.mi_clearselected()
    
            femm.mi_selectsegment(slot_radius * 0.999 * np.cos(0), slot_radius * 0.999 * np.sin(0))
            femm.mi_setsegmentprop("apbc2", 100, 0, 0, 6)
            femm.mi_clearselected()
            femm.mi_selectsegment(h42 * 0.99 * np.cos(0), h42 * 0.99 * np.sin(0))
            femm.mi_setsegmentprop("apbc3", 100, 0, 0, 6)
            femm.mi_clearselected()
            femm.mi_selectsegment(0.99 * D_a * 0.5 * np.cos(0), 0.99 * D_a * 0.5 * np.sin(0))
            femm.mi_setsegmentprop("apbc4", 1, 0, 0, 6)
            femm.mi_clearselected()
            femm.mi_selectsegment(r_o * 0.95 * np.cos(0), r_o * 0.95 * np.sin(0))
            femm.mi_setsegmentprop("apbc5", 100, 0, 0, 6)
            femm.mi_clearselected()
    
            femm.mi_selectgroup(6)
            femm.mi_copyrotate(0, 0, theta_p_d, 1)
    
            femm.mi_selectsegment(h42 * 0.99 * np.cos(theta_p_r), h42 * 0.99 * np.sin(theta_p_r))
            femm.mi_setsegmentprop("apbc3", 100, 0, 0, 6)
            femm.mi_clearselected()
            femm.mi_selectsegment(D_a * 0.5 * 0.99 * np.cos(theta_p_r), D_a * 0.5 * 0.99 * np.sin(theta_p_r))
            femm.mi_setsegmentprop("apbc4", 100, 0, 0, 6)
            femm.mi_clearselected()
    
            iron_label = yoke_radius + (slot_radius - yoke_radius) * 0.5
    
            femm.mi_addblocklabel(iron_label * np.cos(theta_p_r * 0.5), iron_label * np.sin(theta_p_r * 0.5))
            femm.mi_selectlabel(iron_label * np.cos(theta_p_r * 0.5), iron_label * np.sin(theta_p_r * 0.5))
            femm.mi_setblockprop("M-36 Steel", 1, 1, "incircuit", 0, 7, 0)
            femm.mi_clearselected()
    
            select_angle_r = theta_b_s_new + theta_b_t * 0.9
    
            femm.mi_addarc(
                slot_radius * np.cos(theta_b_s_new),
                slot_radius * np.sin(theta_b_s_new),
                slot_radius * np.cos(theta_b_s_new + theta_b_t),
                slot_radius * np.sin(theta_b_s_new + theta_b_t),
                5,
                1,
            )
            femm.mi_selectarcsegment(slot_radius * np.cos(select_angle_r), slot_radius * np.sin(select_angle_r))
            femm.mi_setgroup(10)
    
            #    femm.mi_addarc(D_a*0.5*np.cos(theta_b_s_new),D_a*0.5*np.sin(theta_b_s_new),D_a*0.5*np.cos(theta_b_s_new+theta_b_t),D_a*0.5*np.sin(theta_b_s_new+theta_b_t),5,1)
            #    femm.mi_selectarcsegment(D_a*0.5*np.cos(select_angle_r*0.9),D_a*0.5*np.sin(select_angle_r*0.9))
            #    femm.mi_setgroup(3)
    
            femm.mi_selectgroup(10)
            femm.mi_copyrotate(0, 0, (theta_b_s_new + theta_b_t * 0.5) * 180 / np.pi, Slots_pp - 2)
    
            femm.mi_clearselected()
    
            femm.mi_addblocklabel(xlabel1,ylabel1)
            femm.mi_addblocklabel(xlabel2,ylabel2)
            femm.mi_selectlabel(xlabel1,ylabel1)
            femm.mi_setblockprop("NbTi", 1, 1, "A1+", 0, 10, N_sc)
            femm.mi_clearselected()
            femm.mi_selectlabel(xlabel2,ylabel2)
            femm.mi_setblockprop("NbTi", 1, 1, "A1-", 0, 7, N_sc)
            femm.mi_clearselected()
    
            layer_1 = slot_radius + (h42 - slot_radius) * 0.5
    
            femm.mi_addblocklabel(layer_1 * np.cos(theta_b_t * 0.5 + theta_b_s * 0.5), layer_1 * np.sin(theta_b_t * 0.5 + theta_b_s * 0.5))
            femm.mi_selectlabel(layer_1 * np.cos(theta_b_t * 0.5 + theta_b_s * 0.5), layer_1 * np.sin(theta_b_t * 0.5 + theta_b_s * 0.5))
            femm.mi_copyrotate(0, 0, (theta_b_s + theta_b_t) * 180 / np.pi, Slots_pp - 1)
    
            layer_2 = h42 + (D_a * 0.5 - h42) * 0.5
    
            femm.mi_addblocklabel(layer_2 * np.cos(theta_b_t * 0.5 + theta_b_s * 0.5), layer_2 * np.sin(theta_b_t * 0.5 + theta_b_s * 0.5))
            femm.mi_selectlabel(layer_2 * np.cos(theta_b_t * 0.5 + theta_b_s * 0.5), layer_2 * np.sin(theta_b_t * 0.5 + theta_b_s * 0.5))
            femm.mi_copyrotate(0, 0, (theta_b_s + theta_b_t) * 180 / np.pi, Slots_pp - 1)
    
            #    femm.mi_addblocklabel(h42*np.cos(theta_b_t*0.25),h42*np.sin(theta_b_t*0.25))
            #    femm.mi_selectlabel(h42*np.cos(theta_b_t*0.25),h42*np.sin(theta_b_t*0.25))
            #    femm.mi_copyrotate(0,0,(theta_p_r-theta_b_t*0.5)*180/np.pi,1)
            #
            #    femm.mi_addblocklabel(layer_2*np.cos(theta_b_s*0.25),layer_2*np.sin(theta_b_s*0.25))
            #    femm.mi_selectlabel(layer_2*np.cos(theta_b_s*0.25),layer_2*np.sin(theta_b_s*0.25))
            #    femm.mi_copyrotate(0,0,(theta_p_r-theta_b_t*0.5)*180/np.pi,1)
    
            femm.mi_addblocklabel(yoke_radius * 0.5 * np.cos(theta_p_r * 0.5), yoke_radius * 0.5 * np.sin(theta_p_r * 0.5))
            femm.mi_selectlabel(yoke_radius * 0.5 * np.cos(theta_p_r * 0.5), yoke_radius * 0.5 * np.sin(theta_p_r * 0.5))
            femm.mi_setblockprop("Air", 1, 1, "incircuit", 0, 7, 0)
            femm.mi_clearselected()
    
            femm.mi_addblocklabel(r_o * 0.75 * np.cos(theta_p_r * 0.5), r_o * 0.75 * np.sin(theta_p_r * 0.5))
            femm.mi_selectlabel(r_o * 0.75 * np.cos(theta_p_r * 0.5), r_o * 0.75 * np.sin(theta_p_r * 0.5))
            femm.mi_setblockprop("Air", 1, 1, "incircuit", 0, 7, 0)
            femm.mi_clearselected()
    
            pitch = 1
    
            Phases = ["A+", "D+", "C-", "F-", "B+", "E+", "A-", "D-", "C-"]
            N_c_a    =[2*N_c,4*N_c,4*N_c,4*N_c,4*N_c,4*N_c,2*N_c,2*N_c,2*N_c]
    
            count = 0
            angle_r = theta_b_t * 0.5 + theta_b_s * 0.5
            delta_theta = theta_b_t + theta_b_s
    
            for pitch in range(1, Y_q, 2):
                femm.mi_selectlabel(
                    layer_2 * np.cos(angle_r + (pitch - 1) * (delta_theta)), layer_2 * np.sin(angle_r + (pitch - 1) * (delta_theta))
                )
                femm.mi_selectlabel(layer_2 * np.cos(angle_r + pitch * delta_theta), layer_2 * np.sin(angle_r + pitch * delta_theta))
                femm.mi_setblockprop("20 SWG", 1, 1, Phases[count], 0, 8, N_c)
                #femm.mi_setblockprop("20 SWG", 1, 1, Phases[count], 0, 8, N_c_a[count])
                femm.mi_clearselected()
                count = count + 1
    
            count = 0
            angle_r = theta_b_t * 0.5 + theta_b_s * 0.5
            delta_theta = theta_b_t + theta_b_s
            for pitch in range(1, Y_q, 2):
                femm.mi_selectlabel(
                    layer_1 * np.cos(angle_r + (pitch - 1) * (delta_theta)), layer_1 * np.sin(angle_r + (pitch - 1) * (delta_theta))
                )
                femm.mi_selectlabel(layer_1 * np.cos(angle_r + pitch * delta_theta), layer_1 * np.sin(angle_r + pitch * delta_theta))
                femm.mi_setblockprop("20 SWG", 1, 1, Phases[count + 1], 0, 8, N_c)
                #femm.mi_setblockprop("20 SWG", 1, 1, Phases[count + 1], 0, 8, N_c_a[count+1])
                femm.mi_clearselected()
                count = count + 1
    
            # Now, the finished input geometry can be displayed.
            femm.mi_zoomnatural()
            ## We have to give the geometry a name before we can analyze it.
            femm.mi_saveas("coil_design_new.fem")
            
            #try:
            femm.mi_analyze()
            #except:
            #    breakpoint()
            femm.mi_loadsolution()
            n = 0
            outputs["B_g"], outputs["B_rymax"], outputs["B_coil_max"], outputs["Sigma_normal"] = run_post_process(D_a, radius_sc, h_sc, slot_radius, theta_p_r, alpha_r, beta_r, n)
            Load_line_slope = I_sc / float(outputs["B_coil_max"])
            print("Computing load line with slope {}".format(Load_line_slope))
            a = 5.8929
            b = -(Load_line_slope + 241.32)
            c = 1859.9
            B_o = (-b - np.sqrt(b**2 - 4 * a * c)) / 2 / a
            #I_c = B_o * Load_line_slope #UNUSED
    
    
            
            
            outputs["margin_I_c"] = float(3.5357 * outputs["B_coil_max"]**2 - 144.79 * outputs["B_coil_max"] + 1116.0)
            outputs["Critical_current_limit"]=outputs["margin_I_c"]-I_sc
            outputs["Coil_max_limit"]=B_o-outputs["B_coil_max"]
            outputs["Torque_actual"], outputs["Sigma_shear"] = B_r_B_t(D_a, l_s,p1,delta_em, theta_p_r, I_s, theta_b_t, theta_b_s, layer_1, layer_2, Y_q, N_c, tau_p)
            
            #print (D_a, l_s,p1,delta_em, theta_p_r, I_s, theta_b_t, theta_b_s, layer_1, layer_2, Y_q, N_c, tau_p)
            #print ("Outputs:",alpha_r*180/np.pi,outputs["Torque_actual"]/1e6, outputs["Sigma_shear"]/1e3, outputs["B_coil_max"],outputs["Sigma_normal"])
            #femm.mo_close()
            '''
            if float(outputs["B_g"]) > 2.1:
                print("Peak air gap flux density {} > 2.1 Tesla".format(outputs["B_g"]))
                if I_sc >= margin_I_c:
                    # No more ability to adjust design, send back to optimizer with violated constraints
                    pass
            else:
                while float(outputs["B_g"]) <= 2.10:
                    print(
                        " B_o is {} Tesla; coil field is {} Tesla; Max air-gap field is {}; Iteration #{}".format(
                            B_o, outputs["B_coil_max"], outputs["B_g"], n
                        )
                    )
                    femm.opendocument("coil_design_new.fem")
                    margin_I_c = float(3.5357 * outputs["B_coil_max"]**2 - 144.79 * outputs["B_coil_max"] + 1116.0)
                    if I_sc >= margin_I_c:
                        I_sc = float(margin_I_c)
                        print("Critical current limit reached:")
                        print("60% margin current:", margin_I_c, "Operating current:", I_sc, "B_op:", outputs["B_coil_max"])
                        print("Increasing the turn number")
    
                        femm.mi_modifycircprop("A1+", 1, I_sc)
                        femm.mi_modifycircprop("A1-", 1, -I_sc)
                        diff = 2500 - N_sc
                        if (diff <= 0) & (I_sc == margin_I_c):
                            print("Max turns and critical currents reached")
                            break
                        elif diff > 0 and diff < 125:
                            N_sc = N_sc + diff
                        else:
                            N_sc = N_sc + 125
                        print("to turns:", N_sc)
                        femm.mi_selectlabel(
                            (radius_sc + h_sc * 0.5) * np.cos(alpha_r + beta_r * 0.25),
                            (radius_sc + h_sc * 0.5) * np.sin(alpha_r + beta_r * 0.25),
                        )
                        femm.mi_setblockprop("NbTi", 1, 1, "A1+", 0, 10, N_sc)
                        femm.mi_clearselected()
                        femm.mi_selectlabel(
                            (radius_sc + h_sc * 0.5) * np.cos(alphap_taup_angle_r + beta_r),
                            (radius_sc + h_sc * 0.5) * np.sin(alphap_taup_angle_r + beta_r),
                        )
                        femm.mi_setblockprop("NbTi", 1, 1, "A1-", 0, 7, N_sc)
                        femm.mi_clearselected()
                        femm.mi_analyze()
                        femm.mi_loadsolution()
                        outputs["B_g"], outputs["B_rymax"], outputs["B_coil_max"], outputs["Sigma_normal"] = run_post_process(D_a, radius_sc, h_sc, slot_radius, theta_p_r, alpha_r, beta_r, n)
                        if outputs["B_coil_max"] > B_o:
                            break
                        else:
                            print("B_g:", outputs["B_g"])
                            pass
                    else:
                        if float(outputs["B_g"]) < B_o:
                            I_sc = I_sc + 25
                            print("Increasing operating current now:")
                            femm.mi_modifycircprop("A1+", 1, I_sc)
                            femm.mi_modifycircprop("A1-", 1, -I_sc)
                            femm.mi_analyze()
                            femm.mi_loadsolution()
                            outputs["B_g"], outputs["B_rymax"], outputs["B_coil_max"], outputs["Sigma_normal"] = run_post_process(D_a, radius_sc, h_sc, slot_radius, theta_p_r, alpha_r, beta_r, n)
                            n = n + 1
            '''
    
            
    
    
            #outputs["I_sc_out"] = I_sc
            #outputs["N_sc_out"] = N_sc
        #    except Exception :
        #        outputs["B_g"] =7
        #        outputs["B_coil_max"]=12
        #        outputs["B_rymax"]=5
        #        outputs["Torque_actual"]=50e+06
        #        outputs["sigma_shear"]=300000
        #        return outputs["B_g"],outputs["B_rymax"],outputs["B_coil_max"],N_sc,I_sc,outputs["Torque_actual"],outputs["sigma_shear"]
