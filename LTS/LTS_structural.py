"""LTS_outer_stator.py
Created by Latha Sethuraman
Copyright (c) NREL. All rights reserved.
Electromagnetic design based on conventional magnetic circuit laws
Structural design based on {Structural mass in direct-drive permanent magnet electrical generators by
McDonald,A.S. et al. IET Renewable Power Generation(2008),2(1):3 http://dx.doi.org/10.1049/iet-rpg:20070071 """

import pandas as pd
import numpy as np
import openmdao.api as om



def shell_constant(R,t,l,x,v,E):
    
    Lambda     = (3*(1-v**2)/(R**2*t**2))**0.25
    D          = E*t**3/(12*(1-v**2))
    C_14       = (np.sinh(Lambda*l))**2+ (np.sin(Lambda*l))**2
    C_11       = (np.sinh(Lambda*l))**2- (np.sin(Lambda*l))**2
    F_2        = np.cosh(Lambda*x)*np.sin(Lambda*x) + np.sinh (Lambda*x)* np.cos(Lambda*x)
    C_13       = np.cosh(Lambda*l)*np.sinh(Lambda*l) - np.cos(Lambda*l)* np.sin(Lambda*l)
    F_1        = np.cosh(Lambda*x)*np.cos(Lambda*x)
    F_4        = np.cosh(Lambda*x)*np.sin(Lambda*x)-np.sinh(Lambda*x)*np.cos(Lambda*x)
    
    return D,Lambda,C_14,C_11,F_2,C_13,F_1,F_4
        
def plate_constant(a,b,v,r_o,t,E):
    
    D          = E*t**3/(12*(1-v**2))
    C_2        = 0.25*(1-(b/a)**2*(1+2*np.log10(a/b)))
    C_3        = 0.25*(b/a)*(((b/a)**2+1)*np.log10(a/b)+(b/a)**2 -1)
    C_5        = 0.5*(1-(b/a)**2)
    C_6        = 0.25*(b/a)*((b/a)**2-1+2*np.log10(a/b))
    C_8        = 0.5*(1+v+(1-v)*(b/a)**2)
    C_9        = (b/a)*(0.5*(1+v)*np.log10(a/b)+0.25*(1-v)*(1-(b/a)**2))
    L_11       = (1/64)*(1+4*(r_o/a)**2-5*(r_o/a)**4-4*(r_o/a)**2*(2+(r_o/a)**2)*np.log10(a/r_o))
    L_17       = 0.25*(1-0.25*(1-v)*((1-(r_o/a)**4)-(r_o/a)**2*(1+(1+v)*np.log10(a/r_o))))
            
    return D,C_2,C_3,C_5,C_6,C_8,C_9,L_11,L_17


class LTS_inactive_rotor(om.ExplicitComponent):

    """ Estimates overall electromagnetic dimensions and Efficiency of PMSG -arms generator. """
    
    def setup(self):
    
        
        self.add_input('y_sh', units ='W', desc='Deflection at the shaft')
        self.add_input('theta_sh', 0.0, units = 'rad', desc='slope')
        self.add_input("D_a", 0.0, units="m", desc="armature outer diameter ")
        self.add_input("h_s", 0.0, units="m", desc="Slot height ")
        self.add_input('delta_em',0.0, units ='m', desc='air gap length')

        self.add_input("h_yr", 0.0, units="m", desc="rotor yoke height")
        self.add_input("h_yr_s", 0.0, units="m", desc="rotor yoke disc thickness")
        self.add_input("l_eff_rotor", 0.0, units="m", desc="rotor effective length including end winding")
        
        
        
        self.add_input("t_rdisc", 0.0, units="m", desc="rotor disc thickness")

        self.add_discrete_input("E", 2e11, desc="Young's modulus of elasticity")
        self.add_discrete_input("v", 0.3, desc="Poisson's ratio")
        self.add_discrete_input("g", 9.8106, desc="Acceleration due to gravity")
        self.add_input("rho_steel", 0.0, units="kg/m**3", desc="Mass density")
        self.add_input("T_e", 0.0, units="N*m", desc="Electromagnetic torque")
        
        
        self.add_output("R_ry", 0.0, units="m", desc="mean radius of the rotor yoke")
        self.add_output("r_o", 0.0, units="m", desc="Outer radius of rotor yoke")
        self.add_output("r_i", 0.0, units="m", desc="inner radius of rotor yoke")
        
        self.add_input("R_shaft_outer",0.0, units="m", desc=" Main shaft outer radius")
        self.add_input("R_nose_outer",0.0, units="m", desc=" Bedplate nose outer radius")
        self.add_output("W_ry",0.0, desc=" line load of rotor yoke thickness")
        self.add_input("Copper",0.0, units="kg", desc=" Copper mass")
        self.add_input("Mass_SC",0.0, units="kg", desc=" SC mass")
        self.add_input("Tilt_angle",0.0, units="deg", desc=" Main shaft tilt angle")
        
        

        # Material properties
        self.add_input("Sigma_normal", 0.0, units="N/(m**2)", desc="Normal stress ")
        self.add_input("Sigma_shear", 0.0, units="N/(m**2)", desc="Normal stress ")

        # Deflection
        self.add_output("u_ar", 0.0, units="m", desc="rotor radial deflection")
        self.add_output("y_ar", 0.0, units="m", desc="rotor axial deflection")
        self.add_output("U_rotor_radial_constraint",0.0,units="m", desc="Stator radial deflection contraint")
        self.add_output("U_rotor_axial_constraint",0.0,units="m", desc="Rotor axial deflection contraint")

        # Mass Outputs

        self.add_input('u_allow_pcent',0.0,desc='Radial deflection as a percentage of air gap diameter')
        self.add_input('y_allow_pcent',0.0,desc='Radial deflection as a percentage of air gap diameter')
        self.add_input('z_allow_deg',0.0,units='deg',desc='Allowable torsional twist')
        
        
        # structural design variables
        
        self.add_input('K_rad', desc='Aspect ratio')
        
        # Material properties
        
        self.add_output('Structural_mass_rotor',0.0, units='kg', desc='Rotor mass (kg)')

        self.add_output('twist_r', 0.0, units ='deg', desc='torsional twist')
               
        
        self.add_output('u_allowable_r',0.0,units='m',desc='Allowable Radial deflection')
        self.add_output('y_allowable_r',0.0,units='m',desc='Allowable Radial deflection')


        
        self.declare_partials("*", "*", method="fd")
        
    def compute(self, inputs, outputs, discrete_inputs,discrete_outputs):
        
       
        # Radial deformation of rotor
        
        ###################################################### Rotor disc radial deflection#############################################
        outputs["r_o"]= inputs["D_a"]*0.5-inputs["h_s"]
        
        
        h_total         = inputs["h_yr"]+inputs["h_yr_s"]
        outputs["r_i"]  = outputs["r_o"]-h_total
        outputs["R_ry"] =(outputs["r_o"]+outputs["r_i"])*0.5
        R               = outputs["r_i"]+h_total
        
        print (outputs["r_o"],outputs["r_i"], R)
        
        L_r             = inputs['l_eff_rotor']+inputs['t_rdisc']
        constants_x_0   = shell_constant(R,inputs['t_rdisc'],L_r,0,discrete_inputs["v"],discrete_inputs["E"])
        constants_x_L   = shell_constant(R,inputs['t_rdisc'],L_r,L_r,discrete_inputs["v"],discrete_inputs["E"])
        
       
        f_d_denom1      = R/(discrete_inputs["E"]*((R)**2-(inputs['R_shaft_outer'])**2))*((1-discrete_inputs["v"])*R**2+(1+discrete_inputs["v"])*(inputs['R_shaft_outer'])**2)
        
        
        f_d_denom2      = inputs['t_rdisc']/(2*constants_x_0[0]*(constants_x_0[1])**3)*(constants_x_0[2]/(2*constants_x_0[3])*constants_x_0[4]-constants_x_0[5]/constants_x_0[3]*constants_x_0[6]-0.5*constants_x_0[7])
        
        f               = inputs['Sigma_normal']*(R)**2*inputs['t_rdisc']/(discrete_inputs["E"]*(h_total)*(f_d_denom1+f_d_denom2))
        
        u_d             =f/(constants_x_L[0]*(constants_x_L[1])**3)*((constants_x_L[2]/(2*constants_x_L[3])*constants_x_L[4] -constants_x_L[5]/constants_x_L[3]*constants_x_L[6]-0.5*constants_x_L[7]))+inputs['y_sh']
        
 
        outputs['u_ar'] = (inputs['Sigma_normal']*(R)**2)/(discrete_inputs["E"]*(h_total))-u_d
        
        print (outputs['u_ar'], inputs["h_yr_s"],inputs['t_rdisc'])
                
        outputs['u_ar'] = abs(outputs['u_ar'] + inputs['y_sh'])
        
        outputs['u_allowable_r'] =inputs['delta_em']*inputs['u_allow_pcent']/100
        
        outputs["U_rotor_radial_constraint"]=outputs['u_allowable_r']-outputs["u_ar"]
        
        
        ###################################################### Electromagnetic design#############################################
        #return D,C_2,C_3,C_5,C_6,C_8,C_9,L_11,L_17
        # axial deformation of rotor
        W_back_iron     =  plate_constant(outputs["r_i"]+inputs['h_yr']*0.5,inputs['R_shaft_outer'],discrete_inputs["v"],0.5*inputs['h_yr']+R,inputs['t_rdisc'],discrete_inputs["E"])
        W_ssteel        =  plate_constant(outputs["r_i"]+inputs['h_yr']+inputs['h_yr_s']*0.5,inputs['R_shaft_outer'],discrete_inputs["v"],inputs['h_yr']+outputs['r_i']+inputs['h_yr_s']*0.5,inputs['t_rdisc'],discrete_inputs["E"])
        W_cu          =  plate_constant(inputs["D_a"]*0.5-inputs["h_s"]*0.5,inputs['R_shaft_outer'],discrete_inputs["v"],inputs["D_a"]*0.5-inputs["h_s"]*0.5,inputs['t_rdisc'],discrete_inputs["E"])
        
        outputs["W_ry"] =inputs["rho_steel"]*discrete_inputs["g"]*np.sin(np.deg2rad(inputs["Tilt_angle"]))*(inputs["l_eff_rotor"]-inputs["t_rdisc"])*inputs['h_yr']
        
        wr_disc        =inputs["rho_steel"]*discrete_inputs["g"]*np.sin(np.deg2rad(inputs["Tilt_angle"]))*inputs["t_rdisc"]
        
       
        y_ai1r          = -outputs["W_ry"]*(R)**4/(inputs['R_shaft_outer']*W_back_iron[0])*(W_back_iron[1]*W_back_iron[4]/W_back_iron[3]-W_back_iron[2])
        W_sr            =  inputs['rho_steel']*discrete_inputs["g"]*np.sin(np.deg2rad(inputs["Tilt_angle"]))*(inputs["l_eff_rotor"]-inputs['t_rdisc'])*inputs['h_yr_s']
        y_ai2r          = -W_sr*(outputs["r_i"]+inputs["h_yr"]+inputs["h_yr_s"]*0.5)**4/(inputs['R_shaft_outer']*W_ssteel[0])*(W_ssteel[1]*W_ssteel[4]/W_ssteel[3]-W_ssteel[2])
        W_Cu            =  np.sin(np.deg2rad(inputs["Tilt_angle"]))*inputs['Copper']/(2*np.pi*(inputs["D_a"]*0.5-inputs["h_s"]*0.5))
        y_ai3r          = -W_Cu*(inputs["D_a"]*0.5-inputs["h_s"]*0.5)**4/(inputs['R_shaft_outer']*W_cu[0])*(W_cu[1]*W_cu[4]/W_cu[3]-W_cu[2])
        
              
        a_ii            = outputs["r_o"]
        r_oii           = inputs['R_shaft_outer']
        M_rb            = -wr_disc *a_ii**2/W_ssteel[5]*(W_ssteel[6]*0.5/(a_ii*inputs['R_shaft_outer'])*(a_ii**2-r_oii**2)-W_ssteel[8])
        Q_b             =  wr_disc *0.5/inputs['R_shaft_outer']*(a_ii**2-r_oii**2)
        
        y_aiir          =  M_rb*a_ii**2/W_ssteel[0]*W_ssteel[1]+Q_b*a_ii**3/W_ssteel[0]*W_ssteel[2]-wr_disc*a_ii**4/W_ssteel[0]*W_ssteel[7]
        
        I               =  np.pi*0.25*(R**4-(inputs['R_shaft_outer'])**4)
        #F_ecc           = inputs['Sigma_normal']*2*pi*inputs['K_rad']*inputs['r_g']**3
        #M_ar             = F_ecc*L_r*0.5
               
        
        outputs['y_ar'] =(y_ai1r+y_ai2r+y_ai3r)+y_aiir+(outputs["r_i"]+inputs['h_yr']+inputs['h_yr_s'])*inputs['theta_sh']  #+M_ar*L_r**2*0/(2*discrete_inputs["E"]*I)
        
        outputs['y_allowable_r'] =inputs['l_eff_rotor']*inputs['y_allow_pcent']/100
        # Torsional deformation of rotor
        J_dr            =(1/32)*np.pi*((outputs["r_o"])**4-inputs['R_shaft_outer']**4)
        
        J_cylr          =(1/32)*np.pi*(outputs["r_o"]**4-R**4)
        
        G=0.5*discrete_inputs["E"]/(1+discrete_inputs["v"])
        
        outputs['twist_r']=180/np.pi*inputs['T_e']/G*(inputs['t_rdisc']/J_dr+(inputs['l_eff_rotor']-inputs['t_rdisc'])/J_cylr)
        
        outputs['Structural_mass_rotor'] = inputs['rho_steel']*np.pi*(((outputs['r_i']+inputs['h_yr_s'])**2-(inputs['R_shaft_outer'])**2)*inputs['t_rdisc']+\
                                           ((outputs['r_i']+inputs['h_yr_s'])**2-(outputs['r_i']**2)*inputs['l_eff_rotor']))
                                           
        outputs["U_rotor_axial_constraint"]=outputs['y_allowable_r']-outputs["y_ar"]
     
          
          			
class LTS_inactive_stator(om.ExplicitComponent):
    """ Estimates overall electromagnetic dimensions and Efficiency of PMSG -arms generator. """
    
    def setup(self):
    
        self.add_input('R_nose_outer',0.0, units ='m', desc='Nose outer radius ')
    
        self.add_input('y_bd', units ='W', desc='Deflection of the bedplate')
        self.add_input('theta_bd', 0.0, units = 'm', desc='Slope at the bedplate')
        
        self.add_input('T_e',0.0, units = 'N*m', desc='Electromagnetic torque ')
        self.add_input('D_sc',0.0, units ='m', desc='field coil diameter')
        self.add_input('delta_em',0.0, units ='m', desc='air gap length')
        self.add_input('h_ys',0.0, units ='m', desc='Stator yoke height ')
        # field coil parameters
        self.add_input("h_sc", 0.0, units="m", desc="SC coil height")
        self.add_input("l_eff_stator", 0.0, units="m", desc="stator effective length including end winding")
        
        
        self.add_output("r_is", 0.0, units="m", desc="inner radius of stator disc")
        self.add_output("r_os", 0.0, units="m", desc="outer radius of stator disc")
        self.add_output('R_sy',0.0, units ='m', desc='Stator yoke height ')
        
        # structural design variables
        self.add_input("t_sdisc", 0.0, units="m", desc="stator disc thickness")
        
        self.add_discrete_input("E", 2e11, desc="Young's modulus of elasticity")
        self.add_discrete_input("v", 0.3, desc="Poisson's ratio")
        self.add_discrete_input("g", 9.8106, desc="Acceleration due to gravity")
        self.add_input("rho_steel", 0.0, units="kg/m**3", desc="Mass density")
        
        
        self.add_input("Mass_SC",0.0, units="kg", desc=" SC mass")
        self.add_input("Tilt_angle",0.0, units="deg", desc=" Main shaft tilt angle")
        
        self.add_output("W_sy",0.0, desc=" line load of stator yoke thickness")
        #self.add_input("I_sc", 0.0, units="A", desc="SC current ")

        # Material properties
        self.add_input("Sigma_normal", 0.0, units="N/(m**2)", desc="Normal stress ")
        self.add_input("Sigma_shear", 0.0, units="N/(m**2)", desc="Normal stress ")
        
        self.add_output("U_radial_stator", 0.0, units="m", desc="stator radial deflection")
        self.add_output("U_axial_stator", 0.0, units="m", desc="stator axial deflection")
        self.add_output("U_stator_radial_constraint",0.0,units="m", desc="Stator radial deflection contraint")
        self.add_output("U_stator_axial_constraint",0.0,units="m", desc="Stator axial deflection contraint")
        self.add_input("perc_allowable_radial",0.0, desc=" Allowable radial % deflection ")
        self.add_input("perc_allowable_axial",0.0, desc=" Allowable axial % deflection ")
        
        self.add_input('Structural_mass_rotor', 0.0, units="kg", desc="rotor disc mass")
        self.add_output('Structural_mass_stator',0.0, units='kg', desc='Stator mass (kg)')
        self.add_output("mass_total",0.0, units="kg", desc="stator disc mass")

        self.add_input('K_rad', desc='Aspect ratio')
        
        
        # Material properties
        self.add_input('rho_Fes',0.0, units='kg/(m**3)',desc='Structural Steel density')
        

        self.add_output('u_as',0.0,units='m', desc='Radial deformation')
        self.add_output('y_as',0.0, units ='m', desc='Axial deformation')
        self.add_output('twist_s', 0.0, units ='deg', desc='Stator torsional twist')
        
        self.add_input('u_allow_pcent',0.0,desc='Radial deflection as a percentage of air gap diameter')
        self.add_input('y_allow_pcent',0.0,desc='Radial deflection as a percentage of air gap diameter')
        self.add_input('z_allow_deg',0.0,units='deg',desc='Allowable torsional twist')
       
      
        self.add_output('u_allowable_s',0.0,units='m',desc='Allowable Radial deflection as a percentage of air gap diameter')
        self.add_output('y_allowable_s',0.0,units='m',desc='Allowable Axial deflection as a percentage of air gap diameter')
        
        
    def compute(self, inputs, outputs, discrete_inputs,discrete_outputs):
        
        #Assign values to universal constants
        
       
        # Radial deformation of Stator
        L_s             = inputs['l_eff_stator']+inputs['t_sdisc']
        outputs["r_os"]=inputs["D_sc"]*0.5+inputs["h_sc"]+0.25+inputs["h_ys"]
        outputs["r_is"]=outputs["r_os"]-inputs["h_ys"]
        outputs["R_sy"]=(outputs["r_os"]+outputs["r_is"])*0.5
        R_s            = outputs["R_sy"]
        constants_x_0   = shell_constant(R_s,inputs['t_sdisc'],L_s,0,discrete_inputs["v"],discrete_inputs["E"])
        constants_x_L   = shell_constant(R_s,inputs['t_sdisc'],L_s,L_s,discrete_inputs["v"],discrete_inputs["E"])
        f_d_denom1      = R_s/(discrete_inputs["E"]*((R_s)**2-(inputs['R_nose_outer'])**2))*((1-discrete_inputs["v"])*R_s**2+(1+discrete_inputs["v"])*(inputs['R_nose_outer'])**2)
        f_d_denom2      = inputs['t_sdisc']/(2*constants_x_0[0]*(constants_x_0[1])**3)*(constants_x_0[2]/(2*constants_x_0[3])*constants_x_0[4]-constants_x_0[5]/constants_x_0[3]*constants_x_0[6]-0.5*constants_x_0[7])
        f               = inputs['Sigma_normal']*(R_s)**2*inputs['t_sdisc']/(discrete_inputs["E"]*(inputs['h_ys'])*(f_d_denom1+f_d_denom2))
        outputs['u_as'] = (inputs['Sigma_normal']*(R_s)**2)/(discrete_inputs["E"]*(inputs['h_ys']))-f*0/(constants_x_L[0]*(constants_x_L[1])**3)*((constants_x_L[2]/(2*constants_x_L[3])*constants_x_L[4] -constants_x_L[5]/constants_x_L[3]*constants_x_L[6]-1/2*constants_x_L[7]))+inputs['y_bd']
        
        outputs['u_as'] = abs(outputs['u_as'] + inputs['y_bd'])
        
        outputs['u_allowable_s'] =inputs['delta_em']*inputs['u_allow_pcent']/100
        
        outputs["U_stator_radial_constraint"]=outputs['u_allowable_s']-outputs["u_as"]
        
        ###################################################### Electromagnetic design#############################################
        
        # axial deformation of stator
        W_ssteel        =  plate_constant(R_s+inputs['h_ys']*0.5,inputs['R_nose_outer'],discrete_inputs["v"],R_s+inputs['h_ys']*0.5,inputs['t_sdisc'],discrete_inputs["E"])
        W_sc       =  plate_constant(inputs['D_sc']*0.5+inputs['h_sc']*0.5,inputs['R_nose_outer'],discrete_inputs["v"],inputs['D_sc']*0.5+inputs['h_sc']*0.5,inputs['t_sdisc'],discrete_inputs["E"])
        
        W_is            =  inputs['rho_steel']*discrete_inputs["g"]*np.sin(np.deg2rad(inputs["Tilt_angle"]))*(L_s-inputs['t_sdisc'])*inputs['h_ys']
        y_ai1s           = -W_is*(0.5*inputs['h_ys']+R_s)**4/(inputs['R_nose_outer']*W_ssteel[0])*(W_ssteel[1]*W_ssteel[4]/W_ssteel[3]-W_ssteel[2])
                
        W_field            =  np.sin(np.deg2rad(inputs["Tilt_angle"]))*inputs['Mass_SC']/(2*np.pi*(inputs['D_sc']*0.5+inputs['h_sc']*0.5))
        y_ai2s           = -W_field*(inputs['D_sc']*0.5+inputs['h_sc']*0.5)**4/(inputs['R_nose_outer']*W_sc[0])*(W_sc[1]*W_sc[4]/W_sc[3]-W_sc[2])
        
        w_disc_s        = inputs['rho_steel']*discrete_inputs["g"]*np.sin(np.deg2rad(inputs["Tilt_angle"]))*inputs['t_sdisc']
        
        a_ii            = R_s
        r_oii           = inputs['R_nose_outer']
        M_rb            = -w_disc_s*a_ii**2/W_ssteel[5]*(W_ssteel[6]*0.5/(a_ii*inputs['R_nose_outer'])*(a_ii**2-r_oii**2)-W_ssteel[8])
        Q_b             =  w_disc_s*0.5/inputs['R_nose_outer']*(a_ii**2-r_oii**2)
        
        y_aiis          =  M_rb*a_ii**2/W_ssteel[0]*W_ssteel[1]+Q_b*a_ii**3/W_ssteel[0]*W_ssteel[2]-w_disc_s*a_ii**4/W_ssteel[0]*W_ssteel[7]
        
        I               =  np.pi*0.25*(R_s**4-(inputs['R_nose_outer'])**4)
        #F_ecc           = inputs['Sigma_normal']*2*np.pi*inputs['K_rad']*inputs['r_g']**2
        #M_as             = F_ecc*L_s*0.5
        
        outputs['y_as'] =y_ai1s+y_ai2s+y_aiis+(R_s+inputs['h_ys']*0.5)*inputs['theta_bd'] #M_as*L_s**2*0/(2*discrete_inputs["E"]*I)
        
        
        outputs['y_allowable_s'] =L_s*inputs['y_allow_pcent']/100
        
        # Torsional deformation of stator
        J_ds            =(1/32)*np.pi*((outputs["r_os"])**4-inputs['R_nose_outer']**4)
        
        J_cyls          =(1/32)*np.pi*((outputs["r_os"])**4-outputs["r_is"]**4)
        
        G=0.5*discrete_inputs["E"]/(1+discrete_inputs["v"])
        
        outputs['twist_s']=180.0/np.pi*inputs['T_e']*L_s/G*(inputs['t_sdisc']/J_ds+(L_s-inputs['t_sdisc'])/J_cyls)
        
        outputs['Structural_mass_stator'] = inputs['rho_steel']*(np.pi*((R_s+inputs['h_ys']*0.5)**2-(inputs['R_nose_outer'])**2)*inputs['t_sdisc']+\
                                            np.pi*((outputs['r_os'])**2-outputs['r_is']**2)*inputs['l_eff_stator'])
        
        outputs["U_stator_axial_constraint"]=outputs['y_allowable_s']-outputs["y_as"]
        
        outputs["mass_total"]=outputs['Structural_mass_stator']+inputs['Structural_mass_rotor']
   

class LTS_Outer_rotor_Opt(om.Group):
    def setup(self):
#        self.linear_solver = lbgs = om.LinearBlockJac() #om.LinearBlockGS()
#        self.nonlinear_solver = nlbgs = om.NonlinearBlockGS()
#        nlbgs.options["maxiter"] = 3
#        nlbgs.options["atol"] = 1e-2
#        nlbgs.options["rtol"] = 1e-8
#        nlbgs.options["iprint"] = 2

        ivcs = om.IndepVarComp()
        ivcs.add_output("h_yr_s", 0.0, units="m", desc="rotor yoke thickness")
        ivcs.add_output("h_ys", 0.0, units="m", desc="stator yoke thickness")
        ivcs.add_output("t_rdisc", 0.0, units="m", desc="rotor disc thickness")
        ivcs.add_output("t_sdisc", 0.0, units="m", desc="stator disc thickness")
        ivcs.add_discrete_input("E", 2e11, desc="Young's modulus of elasticity")
        ivcs.add_discrete_input("v", 0.3, desc="Poisson's ratio")
        ivcs.add_discrete_input("g", 9.8106, desc="Acceleration due to gravity")
  
        ivcs.add_output("rho_Fe", 0.0, units="kg/(m**3)", desc="Structural Steel density ")
        self.add_subsystem("ivcs", ivcs, promotes=["*"])
        self.add_subsystem("sys1", LTS_inactive_rotor(), promotes=["*"])
        self.add_subsystem("sys2", LTS_inactive_stator(), promotes=["*"])


if __name__ == "__main__":

    prob = om.Problem()
    prob.model = LTS_Outer_rotor_Opt()

    prob.driver = om.ScipyOptimizeDriver()  # pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'SLSQP' #'COBYLA'
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
    prob["T_e"] = 9e+06
    prob["l_eff_stator"] = 1.44142189  # rev 1 9.94718e6
    prob["l_eff_rotor"] = 1.2827137 
    prob["D_a"]  = 7.74736313
    prob["delta_em"]  =0.0199961
    prob["h_s"]  =0.1803019703
    prob["D_sc"]=7.78735533
    prob["rho_steel"]=7700
    prob["Tilt_angle"]=90.0
    prob["R_shaft_outer"] =1.5
    prob["R_nose_outer"] =0.95
    prob["u_allow_pcent"]=50
    prob['y_allow_pcent']=20
    prob["h_yr"]  =0.1254730934
    prob["h_yr_s"]  =0.191
    prob["h_ys"]  =0.050
    prob["t_rdisc"]  =0.025
    prob["t_sdisc"]  =0.100
    prob["y_bd"]=0.00
    prob["theta_bd"]=0.00
    prob["y_sh"]=0.00
    prob["theta_sh"]=0.00
    
    prob["Copper"]=60e3
    prob["Mass_SC"]=4000
    

    prob.model.approx_totals(method="fd")


    prob.run_model()
    #prob.run_driver()

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
            prob.get_val('Structural_mass_rotor', units="t"),
            prob.get_val("Structural_mass_stator", units="t"),
            prob.get_val("mass_total", units="t"),
                    ],
        "Limit": [
            "",
            "",
            "",
            "",
            prob.get_val('u_allowable_r', units="mm"),
            prob.get_val('y_allowable_r', units="mm"),
            prob.get_val('u_allowable_s', units="mm"),
            prob.get_val('y_allowable_s', units="mm"),
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
    #print(raw_data)
    df = pd.DataFrame(raw_data, columns=["Parameters", "Values", "Limit", "Units"])

    print(df)

    df.to_excel("Optimized_structure_LTSG_MW.xlsx")

