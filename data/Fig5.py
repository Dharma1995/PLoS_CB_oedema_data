#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import log
from scipy.optimize import curve_fit
import math
import pandas as pd


#data

dictp = {}
dictm = {}

for i in [3,5,6,14,25,28,32,38,42,47,51,53,56,60,62,67,69]:
   dfp = pd.read_csv('./IDS/'+str(i)+'/icp_mls.csv')
   p = dfp['ICP'].tolist()
   p.insert(0, 1330)
   dictp['p'+str(i)] = p
   m = dfp['MLS'].tolist()
   m.insert(0,0)
   dictm['m'+str(i)] = m
   
dictp['pstd'] = [1330, 1748.9229668175838, 2045.4501348868507, 2260.591270921171, 2416.7156106313537, 2531.460700351373, 2616.2826820090368, 2678.532255281759, 2724.0433546054837]
dictm['mstd'] = [0, -0.0008311157304833141, -0.0015341752828098368, -0.0020775453844619676, -0.002482702141065397, -0.0027821105323242263, -0.0030033289197561146, -0.003167453820292292, -0.003289928318694662]

dict0 ={}
dict0.update(dictp)
dict0.update(dictm)
list_names = ['std','3','5','6','14','25','28','32','38','42','47','51','53','56','60','62','67','69']
x = ['p']
y = ['m']
list_namesx = []
list_namesy = []
for i in list_names: 
   list_namesx.append(x[0] + i)
   list_namesy.append(y[0] + i)

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.figure(figsize = (7,5))
sk = []
S = np.linspace(0,30,100)
S = S[:,None]
  
for i in [[dict0[list_namesx[0]],dict0[list_namesy[0]],'id:'+str(list_names[0]),'o','tab:blue'],  [dict0[list_namesx[3]],dict0[list_namesy[3]],'id:'+str(list_names[3]),'s','tab:red'], [dict0[list_namesx[6]],dict0[list_namesy[6]],'id:'+str(list_names[6]),'s','tab:green'],  [dict0[list_namesx[13]],dict0[list_namesy[13]],'id:'+str(list_names[13]),'s','tab:orange'],[dict0[list_namesx[14]],dict0[list_namesy[14]],'id:'+str(list_names[14]),'s','grey']]: #[[dict0[list_namesx[0]],dict0[list_namesy[0]],'id:'+str(list_names[0]),'o','tab:blue'], [dict0[list_namesx[1]],dict0[list_namesy[1]],'id:'+str(list_names[1]),'x','tab:green'], [dict0[list_namesx[2]],dict0[list_namesy[2]],'id:'+str(list_names[2]),'s','brown'], [dict0[list_namesx[3]],dict0[list_namesy[3]],'id:'+str(list_names[3]),'s','grey'],[dict0[list_namesx[4]],dict0[list_namesy[4]],'id:'+str(list_names[4]),'s','black'], [dict0[list_namesx[5]],dict0[list_namesy[5]],'id:'+str(list_names[5]),'s','grey'], [dict0[list_namesx[6]],dict0[list_namesy[6]],'id:'+str(list_names[6]),'s','tab:red'], [dict0[list_namesx[7]],dict0[list_namesy[7]],'id:'+str(list_names[7]),'s','grey'], [dict0[list_namesx[8]],dict0[list_namesy[8]],'id:'+str(list_names[8]),'s','grey'], [dict0[list_namesx[9]],dict0[list_namesy[9]],'id:'+str(list_names[9]),'s','grey'], [dict0[list_namesx[10]],dict0[list_namesy[10]],'id:'+str(list_names[10]),'s','grey'],  [dict0[list_namesx[11]],dict0[list_namesy[11]],'id:'+str(list_names[11]),'s','tab:red'], [dict0[list_namesx[12]],dict0[list_namesy[12]],'id:'+str(list_names[12]),'s','grey'], [dict0[list_namesx[13]],dict0[list_namesy[13]],'id:'+str(list_names[13]),'s','grey'],[dict0[list_namesx[14]],dict0[list_namesy[14]],'id:'+str(list_names[14]),'s','grey'],[dict0[list_namesx[15]],dict0[list_namesy[15]],'id:'+str(list_names[15]),'s','grey'],[dict0[list_namesx[16]],dict0[list_namesy[16]],'id:'+str(list_names[16]),'s','grey'],[dict0[list_namesx[17]],dict0[list_namesy[17]],'id:'+str(list_names[17]),'s','grey']]:     
   a = i[0]
   p = np.absolute(a)
   p = p/133
   b = i[1]
   m = np.absolute(b)
   m = m*1000
   sample_weight = np.ones(len(a)) 
   sample_weight[0] *= 200
   plt.scatter(p, m, facecolor = 'None', edgecolor = i[4], marker = i[3], label = i[2])  
   p = p.reshape((len(a), 1))
   m = m.reshape((len(a), 1))
   regr = LinearRegression()
   regr.fit(p, m, sample_weight)
   plt.plot(S, regr.predict(S), color=i[4], linewidth=2) 
   v = (regr.predict(p)[1])/(p[1]-10)
   sk.append(v[0])

font1 = {'family' : 'Arial','weight' : 'normal','size'   : 15}
plt.xlabel('ICP (mmHg)', fontsize = 15)
plt.ylabel('MLS (mm)', fontsize = 15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlim((10, 23))
plt.ylim((0, 4))
plt.legend(frameon = False, fontsize = 15)
bwith = 1.5 
TK = plt.gca()
TK.spines['bottom'].set_linewidth(bwith)
TK.spines['left'].set_linewidth(bwith)
TK.spines['top'].set_linewidth(bwith)
TK.spines['right'].set_linewidth(bwith)
plt.show()



