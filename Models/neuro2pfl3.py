# -*- coding: utf-8 -*-
"""
Created on Sat May 16 15:31:20 2026

@author: dowel

This class variable will allow you to model a goal and heading signal as 
PFL3 neuron output. The PFL3 neurons have been characterised as per
the FAFB connectome. Future iterations can sub in other connectome derived data
or even averages across connectomes. The script for deriving the PFL3 variables 
can be found in PropagateEPG_Wedge.py in the Connectome-Analysis repo. 

The angular tuning of the EB, FSB and PB has followed my imaging conventions. 
Namely, EB wedges are numbered from -pi -> pi from bottom counter-clockwise.
This follows that the rightmost and leftmost parts of the FSB are at +- pi 
and that the centre glomeruli (gloms L and R 1 as per Hulse et al. 2021)
are also at +- pi. This logic flows from assigning anatomical phase to 
EPG neurons based upon their angular position around the EB. This then defines
glomerular tuning in the PB, with the logic above falling into place. The FSB is
defined as if there were no anatomical shifts as defined by PFGs/PFR_as. 

On ideal data the model does a good job at outputting a turn signal as per
Mussels-Pires at al. 2024. There is a slight bias that becomes more apparent with 
very strong goal signals offset by about 90 degrees from heading. I think this
has a neglible effect on modelling of real neural data. I have not looked into
its origin, but I suspect it may come from assigning angles to EPG neurons. 
It could be that there is some slight over/under sampling of angles because of
ring anisotropies. I tried to account for this by projecting the EB coordinates
into PCA space, but this will only do a linear warp. Future iterations may want
to make adjustments to this. Alternatively angles can be defined based upon the
PB glomeruli instead. 



"""
from Utilities.utils_general import utils_general as ug
import numpy as np
import os
class pfl3_model:
    def __init__(self,connectome_data='FAFB'):
        dataname = 'D:\ConnectomeData\FlywireWholeBrain\PFL3_info.pkl'
        self.pfl3_data  = ug.load_pick(dataname)
        
        dataname = 'D:\ConnectomeData\FlywireWholeBrain\Delta7_GlomAdv.pkl'
        self.d7_data = ug.load_pick(dataname)
        
        dataname = 'D:\ConnectomeData\FlywireWholeBrain\EPG_GlomAdv.pkl'
        self.epg_data = ug.load_pick(dataname)
    
    def model_pfl3_phase(self,phase_eb,phase_goal,goal_weight=2,eb_function='cosine',goal_function='cosine'):
        """
        Runs an idealised model based upon phase inputs
        
        Inputs:
            phase_eb: elipsoid body phase in radians
            phase_goal: goal phase in radians
            goal_weight: a scalar to multiply the goal signal by
            eb_function/goal_function: cosine for now but can be replaced with functions closer to what has been recorded
            
        Returns
        -------
        
        
        None.

        """
        # Get pfl3 info from structure
        col12_theta= self.pfl3_data['col12_theta']
        PFL3_inputmat = self.pfl3_data['PFL3_inputmat']
        PFL3_inputmatN = PFL3_inputmat/np.sum(PFL3_inputmat,axis=0)
        PFL3_input_angles = self.pfl3_data['PFL3_input_angles']
        activation_sign = self.pfl3_data['activation_sign']
        pfl3_thetas  = self.pfl3_data['PFL3_thetas']
        PFL3_LAL = self.pfl3_data['PFL3_LAL']
        
        poffset = ug.circ_subtract(col12_theta[0],-np.pi)
        eb = ug.circ_subtract(phase_eb,-poffset) # add phase offset to match anatomy, this should be close to zero. From my data it is 18 degrees
        phase = ug.circ_subtract(phase_goal,-poffset)

        # np.apply_along_axis(lambda row: np.interp(x_new, x_old, row), 1, fsb)
        L = np.zeros(len(eb))
        R = np.zeros(len(eb))
        for ie,e in enumerate(eb):
            act_vector = (np.cos(PFL3_input_angles-e)+1)*activation_sign
            pact = np.matmul(act_vector,PFL3_inputmatN)
            
            gact = goal_weight*(np.cos(-phase[ie]+pfl3_thetas)+1)
            tact = np.exp(pact+gact) #Using exponential of sums as output. It is crucial that the heading and goal signals are multiplied

            L[ie] = np.sum(tact[PFL3_LAL==-1])
            R[ie] = np.sum(tact[PFL3_LAL==1])
        turn= (R-L)/(R+L)
        return L,R,turn
    
    def model_pfl3_fsb_data(self,phase_eb,fsb,goal_weight=2,eb_function='cosine',goal_function='cosine'):
        """
        
        Runs a model based on a cosine function of heading and measured FSB activity
        
        Inputs:
            phase_eb: elipsoid body phase in radians
            fsb: input fsb columnar activity to the model. This allows modelling of non idealised goal signals
            goal_weight: a scalar to multiply the goal signal by
            eb_function: cosine for now but can be replaced with functions closer to what has been recorded
            
        Returns
        -------
        
        
        None.

        """
        # Get pfl3 info from structure
        col12_theta= self.pfl3_data['col12_theta']
        PFL3_inputmat = self.pfl3_data['PFL3_inputmat']
        PFL3_inputmatN = PFL3_inputmat/np.sum(PFL3_inputmat,axis=0)
        PFL3_input_angles = self.pfl3_data['PFL3_input_angles']
        activation_sign = self.pfl3_data['activation_sign']
        PFL3_LAL = self.pfl3_data['PFL3_LAL']
        pfl3_col_id = self.pfl3_data['PFL3_col_id']
        
        poffset = ug.circ_subtract(col12_theta[0],-np.pi)
        eb = ug.circ_subtract(phase_eb,-poffset) # add phase offset to match anatomy, this should be close to zero. From my data it is 18 degrees
        # Interpolate FSB signal onto new 12 columns
        x_old = np.linspace(-np.pi, np.pi, 16)
        x_new = np.linspace(-np.pi, np.pi, 12)
        #x_old = ug.circ_subtract(x_old,)
        x_new = ug.circ_subtract(x_new,poffset)

        fsb_interp =ug.circular_column_interp(fsb,x_old,x_new)
        
      

        turn = np.zeros(len(eb))
        L = np.zeros(len(eb))
        R = np.zeros(len(eb))
        for ie,e in enumerate(eb):
            act_vector = (np.cos(PFL3_input_angles-e)+1)*activation_sign
            pact = np.matmul(act_vector,PFL3_inputmatN)
            
            
            gact = fsb_interp[ie,pfl3_col_id]*goal_weight
            
            tact = np.exp(pact+gact)
            
            L[ie] = np.sum(tact[PFL3_LAL==-1])
            R[ie] = np.sum(tact[PFL3_LAL==1])
        turn= (R-L)/(R+L)
           
        return L,R,turn
    
    def model_pfl3_all_data(self,ebw,fsb,goal_weight=2,d7weight=1,epgweight=1):
        """
        Runs based upon measured activity with no phase transformations
        
        Inputs:
            ebw: input eb wedge data. This allows modelling of raw bump data
            fsb: input fsb columnar activity to the model. This allows modelling of non idealised goal signals
            goal_weight: a scalar to multiply the goal signal by
            eb_function: cosine for now but can be replaced with functions closer to what has been recorded
            
        Returns
        -------
        
        
        None.

        """
        
        # Not applying offsets to data, could add if needs be
        
        
        activation_sign = self.pfl3_data['activation_sign']
        PFL3_inputmat = self.pfl3_data['PFL3_inputmat']
        PFL3_inputmatN = PFL3_inputmat/np.sum(PFL3_inputmat,axis=0)*activation_sign[:,np.newaxis]
        
        PFL3_LAL = self.pfl3_data['PFL3_LAL']
        pfl3_col_id = self.pfl3_data['PFL3_col_id']
        
        pbglom = self.epg_data['PB_glomeruli']
        
        d7_inputmat = self.d7_data['input_gloms'] # input from gloms 1-16 of EB
        d7_inputmat = d7_inputmat/np.sum(d7_inputmat,axis=0)
        eb_to_pb = [1,3,5,7,9,11,13,15,0,2,4,6,8,10,12,14]
        
        # Transform ebwedges into EGP and d7 population activity
        ebpb = ebw[:,eb_to_pb]
        d7_act = np.matmul(ebpb,d7_inputmat)*d7weight
        
        epg_act = ebw[:,pbglom]*epgweight
        
        epg_d7 = np.append(epg_act,d7_act,axis=1)
        # Get pb activation
        pact = np.matmul(epg_d7,PFL3_inputmatN)
        
        x_old = np.linspace(-np.pi, np.pi, 16)
        x_new = np.linspace(-np.pi, np.pi, 12)
        #x_old = ug.circ_subtract(x_old,)
        
        fsb_interp =ug.circular_column_interp(fsb,x_old,x_new)
        gact = fsb_interp[:,pfl3_col_id]*goal_weight
        
        
        tact = np.exp(pact+gact)
        
        L = np.sum(tact[:,PFL3_LAL==-1],axis=1)
        R = np.sum(tact[:,PFL3_LAL==1],axis=1)
        turn= (R-L)/(R+L)
        
        return L,R,turn
#%% Tests
# from analysis_funs.CX_analysis_col import CX_a
# from EdgeTrackingOriginal.ETpap_plots.ET_paper import ET_paper
# import matplotlib.pyplot as plt
# datadir ="Y:\Data\FCI\Hedwig\FC2_maimon2\\240514\\f1\\Trial2"
# etp = ET_paper(datadir)

# mdl = pfl3_model()
# phase_eb = etp.cxa.pdat['phase_eb']
# phase_goal = etp.cxa.pdat['phase_fsb_upper']


# L,R,turns = mdl.model_pfl3_phase(phase_eb,phase_goal,goal_weight=2,eb_function='cosine',goal_function='cosine')
    
# plt.plot(turns,color='k')    
# fsb  = etp.cxa.pdat['wedges_fsb_upper']

# L,R,turns2 =  mdl.model_pfl3_fsb_data(phase_eb,fsb,goal_weight=2,eb_function='cosine',goal_function='cosine')
# plt.plot(turns2,color='r')
    
# ebw = etp.cxa.pdat['wedges_eb']
# L,R,turns3 = mdl.model_pfl3_all_data(ebw,fsb,goal_weight=2)
    
    
    
    