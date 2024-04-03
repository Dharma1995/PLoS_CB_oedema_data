#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import log
from scipy.optimize import curve_fit
import math


#data
p025lp = [1330, 1519.6636523170823, 1679.6056195506756, 1814.6737982967595, 1928.7214945980581, 2026.4182128734583, 2109.2511595699175, 2178.9890400107524, 2237.53301844494]
m025lp = [0, -0.0003575457621582986, -0.0006612108248180098, -0.0009180085698247169, -0.0011343179116383674, -0.0013159715494832447, -0.0014681701595131266, -0.0015955059226189128, -0.0017019383963450703]

p05lp = [1330, 1523.3699142994556, 1688.4601496317712, 1830.953385679893, 1953.637562972744, 2060.154345412878, 2152.6116059428573, 2232.2916408559145, 2301.114111873118]
m05lp = [0, -0.0003824006456779, -0.0007231521448651507, -0.0010240990236343253, -0.0012881185562870349, -0.0015186338526349083, -0.0017192020261201208, -0.0018933214131395847, -0.002044258283929849]

pstd = [1330, 1578.285574550121, 1781.6551346312847, 1949.9791997740504, 2090.766478219456, 2209.107680855084, 2308.762967361517, 2392.5613888884054, 2462.861652670137]
mstd = [0, -0.0005021388832449539, -0.0009532374650686785, -0.0013482077902606721, -0.0016887550068532464, -0.0019798213200459534, -0.0022274338915323905, -0.0024376125402458797, -0.002615880856168416]


p2lp = [1330,1754.9384449008717, 2049.0602540198574, 2262.0869570802124, 2418.13778763892, 2533.966058484134, 2621.7547206850436, 2687.6084805572614, 2737.3388228401077]
m2lp = [0, -0.0008633001098159579, -0.0016098238081403466, -0.0021954212084314985, -0.0026372779947373493, -0.0029687126769379684, -0.0032184100894180503, -0.0034082072311849214, -0.0035539136616881033]

p5lp = [1330, 1877.6714269370054, 2197.273235199092, 2383.230399632126, 2497.126142206108, 2596.054096025221, 2669.6480410518243, 2723.993461641341, 2764.417727658072, 2794.8591331989096, 2818.4600812706685, 2837.096056600078, 2851.697662119949]
m5lp = [0, -0.0010277598863662753, -0.0019042915967040575, -0.0025482738969315366, -0.0029984990952917896, -0.003316117699461721, -0.0035448088009511925, -0.0037344451860457594, -0.003833033688608889, -0.003946254929103804, -0.004023646288315311, -0.004090237076342048, -0.004144647721937421]

m8lp = [0, -0.0011598229227230627, -0.0021273667791612974, -0.0027887239303404305, -0.0032171469976596347, -0.003502857292921974, -0.0036999644433331367, -0.0038429863002307834, -0.003950980194209469, -0.00403598352677092, -0.00410468169717021, -0.004162097891747198, -0.004210492335172793]
p8lp = [1330, 1962.0508837451587, 2314.8625444764957, 2509.1013291915206, 2614.7020395275335, 2672.9041536705954, 2706.683495489451, 2739.004951706358, 2770.9520879296974, 2796.1758599843547, 2815.959647907707, 2831.731273689496, 2844.48370143414]

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.figure(figsize = (7,5))
sk = []
S = [0.25, 0.5,1 ,2, 5]
So = np.linspace(10,30,100)
So = So[:,None]
for i in [[p025lp,m025lp,'D','0.25Lp', 'tab:blue'], [p05lp,m05lp,'v','0.5Lp', 'tab:grey'], [pstd,mstd, 'o','Lp', 'tab:purple'], [p2lp,m2lp,'s','2Lp', 'tab:brown'],  [p5lp,m5lp,'H','5Lp', 'black'], [p8lp,m8lp,'p','8Lp', 'tab:green']]: 
   a = i[1]
   m = np.absolute(a)
   m = m*1000
   di = np.ones(len(a))
   di = di*133
   b = i[0]
   p = b/di
   sample_weight = np.ones(len(a)) 
   sample_weight[0] *= 200
   plt.scatter(p, m,marker = i[2], s = 50, facecolor = 'None', edgecolor = i[4], label=i[3], lw = 1)  
   p = p.reshape((len(a), 1))
   m = m.reshape((len(a), 1))
   regr = LinearRegression()
   regr.fit(p, m, sample_weight)
   plt.plot(So, regr.predict(So), c = i[4], linewidth=2) 
   v = (regr.predict(p)[1])/(p[1]-10)
   sk.append(v[0])

font1 = {'family' : 'Arial','weight' : 'normal','size'   : 15}
plt.xlabel('ICP (mmHg)', fontsize = 15)
plt.ylabel('MLS (mm)', fontsize = 15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlim((10, 25))
plt.ylim((0, 4.5))
plt.legend(frameon = False, fontsize = 15)
bwith = 1.5 
TK = plt.gca()
TK.spines['bottom'].set_linewidth(bwith)
TK.spines['left'].set_linewidth(bwith)
TK.spines['top'].set_linewidth(bwith)
TK.spines['right'].set_linewidth(bwith)
plt.show()
