#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import log
from scipy.optimize import curve_fit
import math


#data
pstd = [1330, 1748.9229668175838, 2045.4501348868507, 2260.591270921171, 2416.7156106313537, 2531.460700351373, 2616.2826820090368, 2678.532255281759, 2724.0433546054837]

mstd = [0, -0.0008311157304833141, -0.0015341752828098368, -0.0020775453844619676, -0.002482702141065397, -0.0027821105323242263, -0.0030033289197561146, -0.003167453820292292, -0.003289928318694662]

maca = [0,-0.00026786066162216713, -0.0004619935403235322, -0.0006037283112078957, -0.0007087467519080637, -0.0007874488813148826, -0.000848110855965303, -0.0008954222538890091, -0.0009329373551641119]
paca = [1330, 1699.5665382185784, 1956.5546118943312, 2133.8500443462995, 2255.357519572504, 2337.8430815032348, 2393.5113023093954, 2430.9392548449637, 2456.0639526936534]


mmca = [0,-0.0004542794355762786, -0.0008381445201462366, -0.0011507713781194102, -0.0014008176682288108, -0.0015995050900811545, -0.001757391875798042, -0.001883365717303345, -0.0019845342330160514]
pmca = [1330, 1745.8665546387454, 2038.635789024633, 2248.7869080492173, 2400.963263092665, 2512.4580880136546, 2593.575056904869, 2652.5459226735907, 2695.46606389639]


mpca = [0,-6.0248805872222e-05, -9.808319937720525e-05, -0.0001274541197438143, -0.0001493008380283293, -0.0001663175923252243, -0.00018070312234073267, -0.00019331350061512804, -0.0002040617200164077]
ppca = [1330, 1704.0727152880374, 1967.1096281110665, 2152.10936955632, 2281.832249593658, 2372.1387518796514, 2434.649970072482, 2477.7203891643844, 2507.305972646395]


plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.figure(figsize = (7,5))
#ax=fig.add_axes([0,0,1,1])
sk = []
S = np.linspace(0,30,100)
S = S[:,None]

for i in [[pstd,mstd,'Hemishperic','o'], [paca,maca,'ACA','x'], [pmca,mmca,'MCA','s'],[ppca,mpca,'PCA','v']]:#,[p125lp,m125lp],[p075lp,m075lp],[p05lp,m05lp],[p025lp,m025lp], [p5lp,m5lp], [p8lp,m8lp], [p35lp,m35lp]]: #[[p15lp,m15lp]]:#
   a = i[1]
   m = np.absolute(a)
   m = m*1000
   di = np.ones(len(a))
   di = di*133
   b = i[0]
   p = b/di
   sample_weight = np.ones(len(a)) 
   sample_weight[0] *= 200
   plt.scatter(p, m,  marker = i[3], label= i[2])  #color = 'black',
   p = p.reshape((len(a), 1))
   m = m.reshape((len(a), 1))
   regr = LinearRegression()
   regr.fit(p, m, sample_weight)
   plt.plot(S, regr.predict(S), color='grey', linewidth=2) 
   v = (regr.predict(p)[1])/(p[1]-10)
   sk.append(v[0])

font1 = {'family' : 'Arial','weight' : 'normal','size'   : 15}
plt.xlabel('ICP (mmHg)', fontsize = 15)
plt.ylabel('MLS (mm)', fontsize = 15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlim((10, 25))
plt.ylim((0, 4))
plt.legend(frameon = False, fontsize = 15)
bwith = 1.5 
TK = plt.gca()
TK.spines['bottom'].set_linewidth(bwith)
TK.spines['left'].set_linewidth(bwith)
TK.spines['top'].set_linewidth(bwith)
TK.spines['right'].set_linewidth(bwith)
plt.show()


