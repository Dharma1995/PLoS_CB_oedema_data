#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import statsmodels.api as sm
from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D
from skimage.measure import EllipseModel
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms 
import random

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):

    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ellipse 

data = pd.read_csv('clean_data.csv',
        header=0,
        usecols=["patient id", "midline shift","ICP","Side of lesion (1=L, 2=R, 3=Both)", 'Swelling', 'nICP_BB_L','nICP_BB_R','ABP','CPP', 'FV_L', 'FV_R'])

ids = data['patient id'].tolist()
mls = data['midline shift'].tolist() 
icp = data['ICP'].tolist()       
side = data['Side of lesion (1=L, 2=R, 3=Both)'].tolist() 
swell = data['Swelling'].tolist() 
r = data['nICP_BB_R'].tolist() 
l = data['nICP_BB_L'].tolist() 
abp = data['ABP'].tolist()
cpp = data['CPP'].tolist() 
fvl = data['FV_L'].tolist() 
fvr = data['FV_R'].tolist() 

patients = []
hashtable = {}
for i in range(len(ids)):
   
   if ids[i] not in patients:   
      rows = [] 
      patients.append(ids[i])
   rows.append(i)  
   hashtable[ids[i]] = rows

list_mls = []
list_icp = []
list_side = []
list_sos  = []
list_swell = []
list_r = []
list_l = []
list_abp = []
list_cpp = []
list_fvl = []
list_fvr = []
for i in hashtable.keys(): 
   num = 0; mls_sum = 0; icp_sum = 0; side_sum = 0; swell_sum = 0; r_sum = 0; l_sum = 0; abp_sum = 0;cpp_sum = 0;fvr_sum= 0; fvl_sum = 0; 
   for j in hashtable[i]: 
      num += 1
      mls_sum += mls[j]
      icp_sum += icp[j]
      side_sum+= side[j]
      swell_sum+= swell[j]
      r_sum += r[j]
      l_sum += l[j]
      abp_sum += abp[j] 
      cpp_sum += cpp[j]
      fvr_sum += fvr[j]
      fvl_sum += fvl[j]
   list_mls.append(mls_sum/num)
   list_icp.append(icp_sum/num)
   list_side.append(side_sum/num)
   list_swell.append(swell_sum/num)
   list_l.append(l_sum/num)
   list_r.append(r_sum/num)
   list_abp.append(abp_sum/num)
   list_cpp.append(cpp_sum/num)
   list_fvr.append(fvr_sum/num)
   list_fvl.append(fvl_sum/num)

l_r = list(np.array(list_l) - np.array(list_r))   
dict = {'ID': patients, 'mls': list_mls, 'icp': list_icp, 'side':list_side, 'swell':list_swell, 'nICPl': list_l, 'nICPr': list_r, 'abp': list_abp, 'fvr': list_fvr, 'fvl': list_fvl}
df = pd.DataFrame(dict) 
df.to_csv('patients_data.csv')

rgt_mls = []
rgt_icp = []
rgt_lr = []
rgt_abp = []
rgt_cpp = []
fvlr = []
fvrr = []
for i in range(len(list_side)): 
   if list_side[i] == 2:
      rgt_lr.append(l_r[i])
      rgt_mls.append(list_mls[i])
      rgt_icp.append(list_icp[i])
      rgt_abp.append(list_abp[i])
      rgt_cpp.append(list_cpp[i])
      fvlr.append(list_fvl[i])
      fvrr.append(list_fvr[i])
             
lft_cpp = []     
lft_mls = []
lft_icp = []
lft_lr = []
lft_abp = []
fvrl = []
fvll = []

for i in range(len(list_side)): 
   if list_side[i] == 1:
      lft_lr.append(l_r[i])
      lft_mls.append(list_mls[i])
      lft_icp.append(list_icp[i])
      lft_abp.append(list_abp[i])
      lft_cpp.append(list_cpp[i])
      fvll.append(list_fvl[i])
      fvrl.append(list_fvr[i])

ax = plt.gca()
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.ylabel('MLS (mm)', fontsize = 15)
 
all_data = [np.random.normal(0, std, size=100) for std in range(1, 4)]
df1 = pd.DataFrame(rgt_mls, columns=['A'])
df2 = pd.DataFrame(lft_mls, columns=['B'])

df0 = [np.array(rgt_mls),np.array(lft_mls)]
label = ["Clinical", "In-silico"]

mlss = [-0.0012142518173476062, -0.002504755586330487, -0.003872269551252101, -0.00483053334019517, -0.005547795979391772, -0.006086045139702435,  -0.000739509859620237, -0.006330985992568836, -0.0014438323010226046, -0.0020775118642399524, -0.0026275712178251763, -0.0030956161178456493, -0.0034895815029153273, -0.003819266549380461, -0.004094347741993375, -0.0009768808388056116, -0.001960913517221096, -0.002862290175147256, -0.0036372099928493116, -0.004282077285823421, -0.0047977350200028345, -0.0051560796069103345, -0.0055523881011988185, -0.0005021388832449539, -0.0009532374650686785, -0.0013482077902606721, -0.0016887550068532464, -0.0019798213200459534, -0.0022274338915323905, -0.0024376125402458797, -0.002615880856168416] 

mlss = 1000*np.absolute(mlss)
neg_mls=map(lambda x:-x, lft_mls)

mls = list(rgt_mls) + list(neg_mls)

bplot = plt.boxplot([mls, mlss],patch_artist = 'True', labels = label)#, boxprops=dict(facecolor='pink', color='pink')) # ,lft_mls

colors = ['pink', 'lightgreen'] 
for _, line_list in bplot.items():
    for line in line_list:
        line.set_color('black')
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)


mls.sort()
del mls[-1]
del mls[-1]
del mls[0]

x = np.random.normal(2, 0.02, len(mlss))
y = np.random.normal(1, 0.02, len(mls))
plt.scatter(x, mlss, zorder=10, alpha=0.5, s = 15, color = 'tab:grey')
plt.scatter(y, mls, zorder=10, alpha=0.5, s= 15)

mticks = np.arange(-15, 15, 30)
bwith = 1.5       
ax.yaxis.grid(True)
plt.ylim((-6,15))
plt.yticks(range(-6,15, 3), fontsize=15)
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
plt.xticks(fontsize=15)
plt.show()

ax = plt.gca()
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.title('MLS-ICP relationship');

#del rgt_icp[6]
#del rgt_icp[8]
#del rgt_abp[6]
#del rgt_abp[8]

ratio = np.array(rgt_abp)/np.array(rgt_icp)
ratio1 = np.array(lft_abp)/np.array(lft_icp)
plt.scatter(lft_abp, lft_icp, s=60, marker = 'x' , color="tab:blue",  label = 'left')
plt.scatter(rgt_abp, rgt_icp, s=60, marker = 'o' , color="tab:red", facecolor = 'None', label = 'right')

ax = plt.gca()
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.ylabel('ICP (mmHg)', fontsize = 15)
plt.xlabel('ABP (mmHg)', fontsize = 15)

#elipsefit
#90hemi = [1649.882278055356, 1897.2896336697509, 2124.72164942145, 2324.9734025419552, 2498.5475594376667, 2648.6435241012223]
pca90 = [1578.285574550121, 1781.6551346312847, 1949.9791997740504, 2090.766478219456, 2209.107680855084, 2308.762967361517, 2392.5613888884054, 2462.861652670137]
pca95 = [1684.359734768451, 1975.6502150258154, 2217.5554421075685, 2420.8761279510472, 2591.6079859610977, 2735.541735908776, 2856.5326327905896, 2959.9630015414923]#[1674.286414897032, 1944.7099062440127, 2192.278848373548, 2407.6118598585967, 2594.9492698558956, 2757.7419868958173]
pca100 = [1790.4339010290437, 2169.683091609991, 2485.653312117397, 2750.9759030242976, 2974.1350620321246, 3163.6248542505186, 3323.4389294238317, 3459.5931542285284]#[1861.1922415814338, 2287.748413112656, 2631.4057816988384, 2908.0317504390055, 3130.0520620509374, 3307.551776879443]
pca105 = [1896.5080710381096, 2363.851328328061, 2755.3195216345416, 3084.2429776177955, 3361.778893417554, 3595.1390208529206, 3794.3701455512487]#, 3962.088948477208]#[1900.5406187555673, 2358.7018762494595, 2727.8380921816806, 3024.9974376032474, 3263.5246855632786, 3454.2460686491745]

dfp = pd.read_csv('patients_data_abp_icp.csv')
rgt = dfp[['icpr', 'abpr']].values.reshape(-1,2)
lft = dfp[['icpl', 'abpl']].values.reshape(-1,2)

silico_abp90 = list(90*np.ones(8)) 
silico_abp95 = list(95*np.ones(8)) 
silico_abp100 = list(100*np.ones(8)) 
silico_abp105 = list(105*np.ones(7))
pca90 = list(np.array(pca90/(133*np.ones(8))))
pca95 = list(np.array(pca95/(133*np.ones(8))))
pca100 = list(np.array(pca100/(133*np.ones(8))))
pca105 = list(np.array(pca105/(133*np.ones(7))))
silico_icp = pca90 + pca95 + pca100 + pca105
silico_abp = silico_abp90 + silico_abp95 + silico_abp100 + silico_abp105
plt.scatter(silico_abp, silico_icp, s=60, marker = 'o' , color="tab:blue",label = 'in-silico',alpha=0.5)
 
rgt = df[['abp','icp']].values.reshape(-1,2)
x = rgt[:, 0]
y = rgt[:, 1]
ell = EllipseModel()
ell.estimate(rgt)

xc, yc, a, b, theta = ell.params

ellipse = confidence_ellipse(x, y, ax, n_std = 1.6, edgecolor='black')
ax.add_patch(ellipse) 

regr = LinearRegression()
x = x.reshape((len(x), 1))
y = y.reshape((len(y), 1))
regr.fit(x, y)
So = np.linspace(80,118,100)
So = So.reshape((len(So), 1))
plt.plot(So, regr.predict(So), color='grey',  linewidth=1, linestyle = '--') 

plt.xlim((80,118))
plt.ylim((5,38))
ell_patch = Ellipse((xc, yc), 2*a, 2*b, theta*180/np.pi, edgecolor='grey', facecolor='none')

plt.legend(fontsize = 15) 
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

