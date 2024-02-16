# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 10:11:43 2024

@author: dowel
"""

import numpy as np
import pandas as pd
import src.utilities.funcs as fc
from analysis_funs.optogenetics import opto 
import os
import matplotlib.pyplot as plt 
#%%
rootdir = "Y:\Data\Optogenetics\AndyData\FB4R inhibition"
lname = os.listdir(rootdir)
savedirs = ['05312022-144853_FB4R_GtACR1_preinhibition_Fly1.log',
'05312022-145654_FB4R_GtACR1_outsideinhibition_Fly1.log',
'05312022-150708_FB4R_GtACR1_Insideinhibition_Fly1.log',
'05312022-151524_FB4R_GtACR1_postinhibition_Fly1.log',
'05312022-154237_FB4R_GtACR1_preinhibition_Fly2(lost tracking).log',
'05312022-155304_FB4R_GtACR1_Insideinhibition_Fly2.log',
'05312022-155934_FB4R_GtACR1_outsideinhibition_Fly2.log',
'05312022-170143_FB4R_GtACR1_preinhibition_Fly3(lost tracking).log',
'05312022-171834_FB4R_GtACR1_outsideinhibition_Fly3(lost tracking).log',
'05312022-172412_FB4R_GtACR1_insideinhibition_Fly3(lost tracking).log',
'05312022-175724_FB4R_GtACR1_preinhibition_Fly4(lost tracking).log',
'05312022-180221_FB4R_GtACR1_insideinhibition_Fly4(lost tracking).log',
'05312022-181623_FB4R_GtACR1_outsideinhibition_Fly4(lost tracking).log',
'05312022-185234_FB4R_GtACR1_preinhibition_Fly5.log',
'05312022-190449_FB4R_GtACR1_outsideinhibition_Fly5.log',
'05312022-193709_FB4R_GtACR1_insideinhibition_Fly5.log',
'05312022-204320_FB4R_GtACR1_preinhibition_Fly6(lost tracking).log',
'05312022-205208_FB4R_GtACR1_insideinhibition_Fly6.log',
'05312022-210315_FB4R_GtACR1_outsideinhibition_Fly6.log',
'05312022-213104_FB4R_GtACR1_preinhibition_Fly7(lost tracking).log',
'05312022-214030_FB4R_GtACR1_outsideinhibition_Fly7(lost tracking).log',
'05312022-214735_FB4R_GtACR1_insideinhibition_Fly7.log']

meta_data = {'stim_type': 'plume',
             'act_inhib':'inhib',
    'ledONy': 'all',
             'ledOffy':'all',
             'LEDoutplume': True,
             'LEDinplume': False,
             'PlumeWidth': float(50),
             'PlumeAngle': 0,
             }
for s in savedirs:
    datadir =os.path.join(rootdir,s)
    df = fc.read_log(datadir)
    op = opto()
    op.plot_plume_simple(meta_data,df)
    st = s.split('_')
    plt.title(st[-2])
    
#%% h delta C inhibition
plt.close('all')
rootdir = "Y:\Data\Optogenetics\AndyData\hDeltaC inhibition"
lname = os.listdir(rootdir)
savedirs = ['11012022-110656_HdC_inhibition_thresh500_Fly1.log',
 '11012022-113023_HdC_inhibition_nothresh_Fly1.log',
 '11012022-120434_HdC_inhibition_Thresh500_Fly2.log',
 '11012022-122430_hdC_inhibition_noThresh_Fly2.log',
 '11092022-151404_HDC_inhibition_nothresh_Fly1.log',
 '11092022-151958_HDC_inhibition_outside_500thresh_Fly1.log',
 '11102022-140033_HDC_inhibition_inside_500thresh_Fly1.log',
 '11102022-141441_HDC_inhibition_inside_500thresh_Fly1.log',
 '11102022-145807_HDC_inhibition_outside_500Thresh_Fly2.log',
 '11102022-160444_HDC_inhibition_inside_500thresh_Fly3.log',
 '11102022-161937_HDC_inhibition_outside_500thresh_Fly3.log',
 '11102022-162850_HDC_inhibition_inside_500thresh_Fly3.log',
 '11102022-171526_HDC_inhibition_outside_500thresh_Fly4.log']

meta_data = {'stim_type': 'plume',
             'act_inhib':'inhib',
    'ledOny': 'all',
             'ledOffy':'all',
             'LEDoutplume': True,
             'LEDinplume': False,
             'PlumeWidth': float(50),
             'PlumeAngle': 0,
             }
for s in savedirs:
    datadir =os.path.join(rootdir,s)
    df = fc.read_log(datadir)
    op = opto()
    #op.plot_plume_simple(meta_data,df)
    op.plot_traj_scatter(df)
    st = s.split('_')
    plt.title(st[-2])