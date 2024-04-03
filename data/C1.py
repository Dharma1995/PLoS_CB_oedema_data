#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import copy

def best_fit(X, Y):

    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) # or len(Y)

    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2

    b = numer / denum
    a = ybar - b * xbar

    return a, b    
 
#std
a = [0, -0.0014007633838715248, -0.0027782249611590783, -0.0038313416729631442, -0.004302486322535252, -0.005266058760157659, -0.005933492357988066, -0.006345874335838731, -0.0068] # [0, -0.0012142518173476062, -0.002504755586330487, -0.003872269551252101, -0.00483053334019517, -0.005547795979391772, -0.006086045139702435, -0.006330985992568836, -0.0068172173517954925] # 
b = [1330, 1896, 2363, 2755, 3084, 3361, 3595, 3794, 3962]

b = [1330, 1896, 2361, 2744, 3063, 3327, 3548, 3733, 3886]
a = [0, -0.0012620858144254423, -0.0026084949756760793, -0.0036937277397660266, -0.004668939244940287, -0.005317277892494812, -0.0058717224863042115, -0.00630174690151477, -0.006637534764628965]

#sp_new
#(0, 0.029, -0.005)  #!!! light blue
a1 = [0, -0.0008909519679719379, -0.0017466976794795208, -0.0023632637433088958, -0.002748742145826488, -0.0030386806719613817, -0.0032076564695949867, -0.0033250685982688937, -0.0034007543722609575]

# (0, 0.045, -0.005)
a = [0, -0.0011432666744379174, -0.002237847706917337, -0.0030393842695795397, -0.0036056327072590886, -0.004049762285460945, -0.004377562393094653, -0.004626798616641718, -0.004814447542948385]


# (0, 0.0425, -0.007) #!!! blue
a2 = [0, -0.0010522404952377665, -0.002104305258085549, -0.0028829740740735983, -0.003397572204668822, -0.0038076015391830817, -0.0040765240149956425, -0.004273515587883073, -0.004423423670443431]

#(0, 0.017, 0.007)
a = [0, -0.0010370491420527914, -0.0021530320007463665, -0.0030564114877798415, -0.003868040985821531, -0.004416253047937571, -0.004886307224790261, -0.0052526332746542816, -0.005540359925424667]

#(0, 0.04, 0.007)
a = [0,-0.0010343895032503092, -0.0020169716262416967, -0.0027307096981461065, -0.0032423782210326347, -0.0036917139586998284, -0.004015625657445604, -0.00425617791194794, -0.004445546776102803]

#(0, 0.025, 0)
a = [0,-0.001158095193624019, -0.0023190189446396494, -0.003214036255993004, -0.003942834313721684, -0.004448153812657779, -0.0048267503968707545, -0.005088561065334352, -0.005268867100008862]

#(0, 0.046, 0)
a= [0,-0.0011387255412448262, -0.0022078831982951935, -0.002983791013721933, -0.003556474753548594, -0.004033633260589537, -0.004402957121185334, -0.0046823823280915345, -0.004893504883149995]


# (0, 0.0465, -0.001)
a = [0, -0.0011218858207349405, -0.0021567621861303837, -0.002902490032082795, -0.0034472163318898766, -0.0039019416904515465, -0.004257359875076337, -0.00453063898013255, -0.0047384431002722265]

# (0, 0.0325, 0.012) !!! green
a3 = [0, -0.001013682157407724, -0.001981386658243263, -0.002683365499078157, -0.003171499875151966, -0.003610158698056826, -0.003933193232668962, -0.004178017458518708, -0.004367913850873769]

#(0, 0.045, -0.005) 
a=[0,-0.0011432666744379174, -0.002237847706917337, -0.0030393842695795397, -0.0036056327072590886, -0.004049762285460945, -0.004377562393094653, -0.004626798616641718, -0.004814447542948385]

#(0, 0.0055, 0.012)
a = [0, -0.0007004111258039727, -0.0014817410979155584, -0.002096555612235124, -0.0026126966277560462, -0.0029748177504735846, -0.003272798616208339, -0.003497920667517139, -0.003670399082629746]

# (0, 0.013, 0.01) #!!! orange
a4 = [0, -0.000877552092921842, -0.0018589649283206025, -0.002659872494836464, -0.0033629481483462044, -0.0038488831750890253, -0.004247417571465358, -0.004550636572631429, -0.004787164804371821]

#(0,0.0055, 0.0135) #!!! yellow
a5 = [0,-0.0007874626902327789, -0.0016491708422169138, -0.002308143488868193, -0.0028175208706318375, -0.0031891713673011615, -0.00347704115693619, -0.003694766157764495, -0.0038607398242232263]

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
ax = plt.gca()
bwith = 1       
plt.ylim((-6,15))
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
plt.xticks(fontsize=12)

lis = [a1,a2,a3,a4]#,a5] #[a1,a3,a4]# 
color = ['brown', 'dodgerblue', 'lime', 'orange']#, 'yellow']#['paleturquoise', 'lime', 'orange']#
label = ['Periventricular point 1', 'Periventricular point 2', 'Periventricular point 3', 'Periventricular point 4']

for i in range(len(lis)): 
   X = [1330, 1896, 2363, 2755]
   a_n = lis[i]
   Y = a_n[:4] 
   Y = copy.deepcopy(Y)

   sample_weight = np.ones(len(X)) 
   sample_weight[0] *= 100

   Y = np.absolute(Y)
   Y = Y*1000
   di = np.ones(len(X))
   di = di*133
   X = X/di   
   
   plt.scatter(X, Y, s=sample_weight, c='grey')
   
   X = np.reshape(X, (-1, 1))
   Y = np.reshape(Y, (-1, 1))

   # The weighted model
   regr = LinearRegression()
   regr.fit(X, Y, sample_weight)

   lin = np.array(range(10, 31))
   lin = np.reshape(lin, (-1, 1))
   plt.plot(lin, regr.predict(lin), color='grey', linewidth=1.5, linestyle = '--')

   m = np.absolute(a_n)
   m = m*1000
   di = np.ones(len(a_n))
   di = di*133
   p = b/di
   plt.plot(p, m, color = color[i], linewidth=2, label = label[i])
   plt.scatter(p, m, color = 'grey', linewidth=1.5, marker = 'x') 
    
   plt.ylabel('MLS (mm)', fontsize = 15)
   plt.xlabel('ICP (mmHg)', fontsize = 15)
   plt.xlim((10, 29.25))  
   plt.ylim((0.01, 5))  
   plt.legend(fontsize=15, loc='upper left', framealpha=0) 
   plt.show()