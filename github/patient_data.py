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
    print('nmdelipse')
    return ellipse #ax.add_patch(ellipse)

pstd = [1330, 1402.7803870716289, 1465.234091088783, 1528.4227569572543, 1590.5415008096927, 1651.1239138279577, 1709.9875501224535, 1767.1809801801753, 1822.7509963932957, 1876.74262294551, 1929.1989332233325,1980.1578838422527, 2029.6621014345956, 2077.7472386893332, 2124.4470313814695, 2169.7935364254226, 2213.8169887227104, 2256.545993994331, 2298.0077679140713, 2338.229844693493, 2377.236571565183,2450.5598154261315, 2519.4259424352945, 2583.9983772263595, 2644.4627728768087, 2701.0043616149333, 2753.8079306487475, 2803.0572774353363, 2848.934485979981, 2891.6191521152873, 2931.2875989058784,2968.111331651251, 3002.256772584303, 3033.888632204921, 3063.1625076668824, 3090.229225622183, 3115.233419568411, 3138.3133637497886]
mstd = [0, -0.0003462075380862476, -0.0006771375479682884, -0.0009951828704965059, -0.001301025516369137, -0.0015948207094722976, -0.0018765946507005428, -0.002146395558959459, -0.00240434756741449, -0.002650660954336787,-0.0028856227243990117, -0.0031225261485497095, -0.0033423076202778223, -0.003553532020647367, -0.0037568408584879743, -0.003949414683116419, -0.004133666558009961, -0.004308860451414463,-0.004476175533277569, -0.004645340568169699, -0.0047936958672025855, -0.004997324715696654, -0.005239609411009533, -0.005457181200262797, -0.005654012576770834, -0.005831651938181525,-0.005990959259112024, -0.006136490217087352, -0.006269911286510448, -0.006388235238930223, -0.006497170587045179, -0.006592596190699756, -0.006694498563063488, -0.006760035112258868, -0.00684760971772057,-0.0069039314635477534, -0.006979229398085933, -0.007035609965885738]

paca = [1330, 1461.905662994244, 1585.4140995535, 1699.8581604977664, 1804.708078565474, 1899.7879993856486, 1985.282750248375, 2061.649234972431, 2129.5135609616227, 2189.5865887361138, 2242.6041476033247, 2325.6803452094523, 2391.0115095325255, 2442.337156966226, 2483.6192536032686, 2516.849552361078, 2543.1916554196014, 2564.077845558219, 2580.6500190411093, 2593.8142234848156, 2604.2869283443565]
maca = [0, -0.00016402239653391093, -0.0003102015262068239, -0.00044152956365426483, -0.0005601297391332323, -0.0006676389432152356, -0.0007649754691543175, -0.0008539023451756427, -0.0009350021794232633, -0.0010087706116297052, -0.0010759330939681888, -0.0011977122940370459, -0.001308643327465611, -0.0014018102827403765, -0.0014806649095233306, -0.0015479298279034652, -0.0016055896310935571, -0.0016552344989441362, -0.0016981601324554476, -0.0017354250231315447, -0.00176789776318462]

pmca = [1330, 1402.2912382455143, 1464.4635379083645, 1528.1881408417516, 1590.5322337757482, 1651.094224381187, 1709.911022392363, 1767.0131068054911, 1822.4258884144263, 1876.1716040689687, 1928.2712016301641, 2026.0379895990498, 2117.5939944257107, 2203.212129736648, 2283.1797586565212, 2357.778612410665, 2426.5575603795182, 2490.96863586273, 2551.0668356237625, 2607.0110623146174, 2659.0780777992386,2707.528546643161, 2752.6509767366483, 2794.7920403384737, 2834.0155910173253, 2870.517654671841, 2904.7591157257666, 2936.8435613877687, 2966.6974852486, 2994.473573476035, 3020.2795929677063,3044.274236456353, 3066.559736533726, 3087.240610860782, 3106.4421027379, 3124.262790999295, 3140.803378324089, 3156.156538388909, 3170.5049195800493, 3183.858377997091, 3196.261266949854]
mmca = [0, -0.00017064849068776928, -0.0003380328000655643, -0.0005025320814128496, -0.0006641670463469621, -0.0008228281654180838, -0.000978361915811983, -0.0011306082742935832, -0.001279418222988007, -0.0014246619251199316, -0.0015662321549287624,-0.0018244661090702574, -0.00209082658426755, -0.002339485866397122, -0.0025728892267491916, -0.002791057324846667,-0.003007158172088059, -0.0032130322100239936, -0.0034076423773024152, -0.0035868075103488275, -0.0037522636123161787, -0.003922855319416084,-0.004083342996015701, -0.004228679962883977, -0.00436220358115058, -0.004484599386796617, -0.004610040769038436, -0.004710892790632739, -0.004800558597732492, -0.004884476548470095, -0.0049670161191479436, -0.0050529817519986775, -0.005136890874809532, -0.005209407555112371, -0.005277620400624689, -0.00534735871931758, -0.005394817607654751, -0.0054431305933973495, -0.00549762745083747, -0.005538677478157435, -0.005576131925217678]

ppca = [1330, 1462.041964461516, 1586.208150367803, 1702.296315906832, 1810.0945647448286, 1909.5166235733304, 2000.6529107239433, 2083.809381481337, 2159.315220414577, 2227.6054916833496, 2289.1584573407167, 2344.472138014291, 2394.233463687503, 2438.8240446576565, 2478.6498737725283, 2514.156158305235, 2545.76121559902, 2573.8536549348432, 2598.791402610514, 2620.9020551089275, 2640.484055520271,2685.7029545728365, 2717.858340685955, 2740.5118039157546, 2756.455056378657, 2767.656128290953, 2775.5055804969716, 2780.990114816347, 2784.8110940855504, 2787.466239468826, 2789.307405664007]
mpca = [0, -3.0866296106994615e-05, -5.9934837754063684e-05, -8.511945452739227e-05, -0.00010642014328541596, -0.0001243080450290858, -0.000139314789058761, -0.00015192226846459758, -0.00016253930017531127, -0.00017150436341124175, -0.00017909529871808197, -0.0001856786185250775, -0.000191283792807584, -0.0001960719184117995, -0.00020016845413690213, -0.00020367925965330297, -0.0002066932295774153, -0.0002092849298330842, -0.00021151695227957198, -0.0002134419516128326, -0.000215104341408648, -0.00021943132904762605, -0.00022225827115435963, -0.00022414886443262734, -0.00022541508143908065, -0.0002262683717892589, -0.00022684639519172055, -0.00022723984791722936, -0.0002275088870436871, -0.00022769365866537314, -0.00022782110177757556]

data = pd.read_csv('cleanfile.csv',
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
print(rgt_icp,rgt_mls)      
#plt.scatter(rgt_icp, rgt_mls, s=60, marker = 'x' , edgecolors="k")   
#plt.show()     
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
#print(rgt_icp,rgt_mls)      
#plt.scatter(lft_icp, lft_mls, s=60, marker = 'x' , edgecolors="k")   
#plt.show()

ax = plt.gca()
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
#plt.xlabel(fontsize = 15)
plt.ylabel('MLS (mm)', fontsize = 15)

#plt.scatter(lft_icp, np.abs(lft_mls), s=60, marker = 'x' , color="tab:blue", label = 'left lesion')  
all_data = [np.random.normal(0, std, size=100) for std in range(1, 4)]
print(all_data)
df1 = pd.DataFrame(rgt_mls, columns=['A'])
#bplot1 = plt.boxplot(df['A'],patch_artist = 'True')
#bplot1['boxes'].set_facecolor('pink')
df2 = pd.DataFrame(lft_mls, columns=['B'])
#df1 = np.array(df1)
#df2 = np.array(df2)
print(df1, df2)
#df1 = df1.reshape(1,len(df1))
#df2 = df2.reshape(1,len(df2))

df0 = [np.array(rgt_mls),np.array(lft_mls)]
label = ["Clinical", "In-silico"]
print(df0, df1, df2)

mlss = [-0.0012142518173476062, -0.002504755586330487, -0.003872269551252101, -0.00483053334019517, -0.005547795979391772, -0.006086045139702435,  -0.000739509859620237, -0.006330985992568836, -0.0014438323010226046, -0.0020775118642399524, -0.0026275712178251763, -0.0030956161178456493, -0.0034895815029153273, -0.003819266549380461, -0.004094347741993375, -0.0009768808388056116, -0.001960913517221096, -0.002862290175147256, -0.0036372099928493116, -0.004282077285823421, -0.0047977350200028345, -0.0051560796069103345, -0.0055523881011988185, -0.0005021388832449539, -0.0009532374650686785, -0.0013482077902606721, -0.0016887550068532464, -0.0019798213200459534, -0.0022274338915323905, -0.0024376125402458797, -0.002615880856168416] #[-0.0005096788794920102, -0.0009354899394307639, -0.0012884407829782457,-0.0007042360416579162, -0.0013117096235972898, -0.0018232563068603948, -0.002247110603718545, -0.002595019803472898, -0.0028792683114732195, -0.000898793200347896, -0.0016984301276107255, -0.0023799809814053644, -0.0029446225201111478, -0.003405425935478358, -0.0037792372445241153, -0.0010933503556616992, -0.0020955846070992564, -0.0029570319390471783, -0.00366657330218072, -0.004239521889401694, -0.00469767829653705, -0.002503088461718364, -0.003552584010964038, -0.004431810194855945, -0.0050555979088024935, -0.005538251659588704, -0.00512508401578029, -0.005800693263898938, -0.006333443212528869]
#-0.0068172173517954925,


#-0.0005021388832449539, -0.0009532374650686785, -0.0013482077902606721, -0.0016887550068532464, -0.0019798213200459534, -0.0022274338915323905, -0.0024376125402458797, -0.002615880856168416]
#[-0.000739509859620237, -0.0014438323010226046, -0.0020775118642399524, -0.0026275712178251763, -0.0030956161178456493, -0.0034895815029153273, -0.003819266549380461, -0.004094347741993375,
# -0.0009768808388056116, -0.001960913517221096, -0.002862290175147256, -0.0036372099928493116, -0.004282077285823421, -0.0047977350200028345, -0.0051560796069103345, -0.0055523881011988185, 
# [-0.0012142518173476062, -0.002504755586330487, -0.003872269551252101, -0.00483053334019517, -0.005547795979391772, -0.006086045139702435, -0.006330985992568836, -0.0068172173517954925, 


print('..............', np.max(mlss), np.min(mlss), np.average(mlss))

mlss = 1000*np.absolute(mlss)
neg_mls=map(lambda x:-x, lft_mls)

mls = list(rgt_mls) + list(neg_mls)


bplot = plt.boxplot([mls, mlss],patch_artist = 'True', labels = label)#, boxprops=dict(facecolor='pink', color='pink')) # ,lft_mls


#mlss.sort()
#print(mlss[0], mlss[-1], np.average(mlss))
#print(rgt_mls[1], rgt_mls[-2], np.average(rgt_mls))

colors = ['pink', 'lightgreen'] # 
#bplot2['boxes'].set_facecolor('lightgreen')
print(bplot.items())
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

print('...............', np.max(mls), np.min(mls), np.max(mlss), np.min(mlss), np.average(mls), np.average(mlss))

mticks = np.arange(-15, 15, 30)
bwith = 1.5       
ax.yaxis.grid(True)
#plt.legend(fontsize = 15) 
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

del rgt_icp[6]
del rgt_icp[8]
del rgt_mls[6]
del rgt_mls[8]
del rgt_abp[6]
del rgt_abp[8]

ratio = np.array(rgt_abp)/np.array(rgt_icp)
ratio1 = np.array(lft_abp)/np.array(lft_icp)
plt.scatter(lft_abp, lft_icp, s=60, marker = 'x' , color="tab:blue",  label = 'left')
plt.scatter(rgt_abp, rgt_icp, s=60, marker = 'o' , color="tab:red", facecolor = 'None', label = 'right')
#plt.scatter(lft_abp, lft_mls, s=60, marker = 'x' , color = "tab:blue")  

ax = plt.gca()
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
#plt.xlabel('ICP (mmHg)', fontsize = 15)
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
print(len(silico_abp), len(silico_icp))
plt.scatter(silico_abp, silico_icp, s=60, marker = 'o' , color="tab:blue",label = 'in-silico',alpha=0.5)
 
rgt = df[['abp','icp']].values.reshape(-1,2)
x = rgt[:, 0]
y = rgt[:, 1]
ell = EllipseModel()
ell.estimate(rgt)

xc, yc, a, b, theta = ell.params

ellipse = confidence_ellipse(x, y, ax, n_std = 1.6, edgecolor='black')
ax.add_patch(ellipse) 

print("center = ",  (xc, yc))
print("angle of rotation = ",  theta)
print("axes = ", (a,b))
regr = LinearRegression()
print(x,y)
x = x.reshape((len(x), 1))
y = y.reshape((len(y), 1))
regr.fit(x, y)
print(x,y)
So = np.linspace(80,118,100)
So = So.reshape((len(So), 1))
plt.plot(So, regr.predict(So), color='grey',  linewidth=1, linestyle = '--') 

plt.xlim((80,118))
plt.ylim((5,38))
ell_patch = Ellipse((xc, yc), 2*a, 2*b, theta*180/np.pi, edgecolor='grey', facecolor='none')

#ax.add_patch(ell_patch)
plt.legend(fontsize = 15) 
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

