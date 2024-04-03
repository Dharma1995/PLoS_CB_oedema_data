#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import log
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import math


#data
slope = [0.3067480584256511, 0.29759606914103187, 0.305358637702461, 0.2855053760705389, 0.30920190594766483, 0.2952383542990821, 0.2743078342210074, 0.2939634844634492, 0.3127414094944135, 0.300577284542405, 0.28349834567105764, 0.3259656246572466, 0.2930169655668207]
slope = [0.3067480584256511, 0.29759606914103187, 0.305358637702461, 0.2855053760705389, 0.30920190594766483, 0.2952383542990821, 0.2743078342210074, 0.29818412715596093, 0.2939634844634492, 0.3127414094944135, 0.300577284542405, 0.3259726197900175, 0.300955600038574, 0.28349834567105764, 0.3259656246572466, 0.30465571280807, 0.299050727793454, 0.2930169655668207]

volume = [0.0005582531138033461, 0.0003980307063621607, 0.0004524894491344372, 0.00042628780031446375, 0.000488338679988666, 0.0004415321081366241, 0.00045182316254749897, 0.00048699174568303673, 0.000596997638558557, 0.0004976889820328703, 0.00043416624041002966, 0.0005959045052264422, 0.0004442735934057021]
volume_w = [0.000989390738218688, 0.001124717672803724, 0.001061366578736137, 0.0014476634326674885, 0.001217045211870505, 0.001100284198634725, 0.0011239422094981278, 0.0012102582571324137, 0.0014810975758344789, 0.00123803935265701, 0.0010795243784843332, 0.0014764880567801934, 0.001106348357004217]

volume = [0.0005377821300612, 0.0003980307063621607, 0.0004524894491344372, 0.00042628780031446375, 0.000488338679988666, 0.0004415321081366241, 0.00045182316254749897, 0.00046240340790509226, 0.00048699174568303673, 0.000596997638558557, 0.0004976889820328703, 0.000523818031837518, 0.0004577077799892785, 0.00043416624041002966, 0.0005959045052264422, 0.00048322019054141817, 0.0004523453075104379, 0.0004442735934057021]
volume_w = [0.0013358736983638618, 0.000989390738218688, 0.001124717672803724, 0.001061366578736137, 0.001217045211870505, 0.001100284198634725, 0.0011239422094981278, 0.0011514608964006234, 0.0012102582571324137, 0.0014810975758344789, 0.00123803935265701, 0.0013018749341434984, 0.001139798871167204, 0.0010795243784843332, 0.0014764880567801934, 0.0012014163227687736, 0.0011261263547293777, 0.001106348357004217]

volume = list(np.array(volume)*1000000*np.ones(len(volume)))
volume_w = list(np.array(volume_w)*1000000*np.ones(len(volume)))

ids = [57, 6, 28, 56, 3, 69, 5, 25, 38, 47, 14, 0, 42, 60, 7]

length = [0.1748304, 0.1506490918553972, 0.16502128185089474, 0.15787238524088004, 0.18285489345821615, 0.16764927808407765, 0.16926103506573587, 0.17471557247130665, 0.180801701776117, 0.1802435071742105, 0.1621895060039399, 0.17451114232735795, 0.16927221929005637]
width = [0.1390632, 0.11817271095005137, 0.12545729922268808, 0.13026174666310641, 0.12912077993255755, 0.12630633439404262, 0.12638674832233332, 0.12994339990023046, 0.13889611082865894, 0.1325646167366831, 0.12765709538900616, 0.1411734702227151, 0.12611294192720962]
height = [0.1559985, 0.14927060026147826, 0.15257625605026384, 0.14699925596945013, 0.146538664822143, 0.1483605034938565, 0.1476813896441191, 0.14998983147386397, 0.16763378714136046, 0.14655456348685386, 0.14801208378117386, 0.17005161989380327, 0.14670762733996573]

length1 = [0.1738507, 0.1502853050698901, 0.16461964128930465, 0.1576783961489283, 0.19028207301769545, 0.18236714876482057, 0.16687268720338144, 0.16877852492975487, 0.1743493765391258, 0.17960057110304228, 0.17972910505620096, 0.16186235867871765, 0.17421532221444894]
width1 = [0.07173997, 0.06174645365786067, 0.06597759625162672, 0.06716834701880615, 0.06703175799654666, 0.06634275083888114, 0.06593780798896438, 0.06570980796459935, 0.06644389123599552, 0.07077823310313146, 0.06725314447654593, 0.06460972547058151, 0.07179583446481638]
height1 = [0.1329383, 0.11482435636600347, 0.11973531116226932, 0.11975195210181748, 0.14071408058179158, 0.1234172776599411, 0.12286806594924335, 0.12936893421471873, 0.12001655092625718, 0.14342813240449492, 0.12566637283717014, 0.12552051076344956, 0.14378956100228266]


length = [0.1748304, 0.1506490918553972, 0.16502128185089474, 0.15787238524088004, 0.18285489345821615, 0.16764927808407765, 0.16926103506573587, 0.17387905943324655, 0.17471557247130665, 0.180801701776117, 0.1802435071742105, 0.18187830162846308, 0.169313786709839, 0.1621895060039399, 0.17451114232735795, 0.17378890288244164, 0.17000618450877802, 0.16927221929005637]
width = [0.1390632, 0.11817271095005137, 0.12545729922268808, 0.13026174666310641, 0.12912077993255755, 0.12630633439404262, 0.12638674832233332, 0.12441836254328324, 0.12994339990023046, 0.13889611082865894, 0.1325646167366831, 0.12566023899702844, 0.12526569919716674, 0.12765709538900616, 0.1411734702227151, 0.12590175083271932, 0.1253734796418962, 0.12611294192720962]
height = [0.1559985, 0.14927060026147826, 0.15257625605026384, 0.14699925596945013, 0.146538664822143, 0.1483605034938565, 0.1476813896441191, 0.15098000395504507, 0.14998983147386397, 0.16763378714136046, 0.14655456348685386, 0.16131771486378832, 0.1525382880953213, 0.14801208378117386, 0.17005161989380327, 0.15630787220528097, 0.15058746879845938, 0.14670762733996573]
length1 = [0.1738507, 0.1502853050698901, 0.16461964128930465, 0.1576783961489283, 0.18236714876482057, 0.16687268720338144, 0.16877852492975487, 0.1733782735259198, 0.1743493765391258, 0.17960057110304228, 0.17972910505620096, 0.1804916158830256, 0.16805867114333412, 0.16186235867871765, 0.17421532221444894, 0.17314772996075217, 0.16879650956298636, 0.16857160707825292]
width1 = [0.070140823, 0.06174645365786067, 0.06597759625162672, 0.06716834701880615, 0.06634275083888114, 0.06593780798896438, 0.06570980796459935, 0.06377884783956873, 0.06644389123599552, 0.07077823310313146, 0.06725314447654593, 0.06385431652196884, 0.06556170543294748, 0.06460972547058151, 0.07179583446481638, 0.06426716100499937, 0.06686697055927154, 0.06352779320502377]
height1 = [0.1329383, 0.11482435636600347, 0.11973531116226932, 0.11975195210181748, 0.1234172776599411, 0.12286806594924335, 0.12936893421471873, 0.12815088826263846, 0.12001655092625718, 0.14342813240449492, 0.12566637283717014, 0.1391494039566732, 0.13045459023813885, 0.12552051076344956, 0.14378956100228266, 0.12874168216769305, 0.12659466369087463, 0.12578929596718263]


length1 = list(np.array(length1)*100*np.ones(len(volume)))
width1 = list(np.array(width1)*100*np.ones(len(volume)))
height1 = list(np.array(height1)*100*np.ones(len(volume)))
length = list(np.array(length)*100*np.ones(len(volume)))
width = list(np.array(width)*100*np.ones(len(volume)))
height = list(np.array(height)*100*np.ones(len(volume)))

ratio1 = list(np.array(volume)/np.array(length1))
ratio2 = list(np.array(volume)/np.array(width1))
ratio3 = list(np.array(volume)/np.array(height1))

ratio1 = list(np.array(height)*np.array(length))
ratio2 = list(np.array(width)*np.array(length))
ratio3 = list(np.array(width)*np.array(height))


fig, ax = plt.subplots()
fig.subplots_adjust(right=0.7)

twin1 = ax.twinx()
twin2 = ax.twinx()
twin3 = ax.twinx()

twin2.spines.right.set_position(("axes", 1.2))
twin3.spines.right.set_position(("axes", 1.4))

sk = []
S = np.linspace(0.2,0.35,100)
S = S[:,None]
slope = np.array(slope).reshape((len(slope), 1))
volume = np.array(volume).reshape((len(slope), 1))
regr = LinearRegression()
regr.fit(slope, volume)
ax.plot(S, regr.predict(S), color='black', alpha = 0.5, linewidth=2) 

p1 = ax.scatter(np.array(slope), np.array(volume), edgecolor = 'black', facecolor = 'None', label="Volume, R^2 = "+ str('%.3f') % r2_score(volume, regr.predict(slope)))

ratio1 = np.array(ratio1).reshape((len(slope), 1))
regr1 = LinearRegression()
regr1.fit(slope, ratio1)
twin1.plot(S, regr1.predict(S), alpha = 0.5, color='r', linewidth=2) 
p2 = twin1.scatter(np.array(slope), np.array(ratio1), edgecolor = "r", facecolor = 'None', label="L*H, R^2 = "+ str('%.3f') % r2_score(ratio1, regr1.predict(slope)))

ratio2 = np.array(ratio2).reshape((len(slope), 1))
regr2 = LinearRegression()
regr2.fit(slope, ratio2)
twin2.plot(S, regr2.predict(S), alpha = 0.5, color='g', linewidth=2) 
p3 = twin2.scatter(np.array(slope), np.array(ratio2), edgecolor = "g", facecolor = 'None', label="L*W, R^2 = " + str('%.3f') % r2_score(ratio2, regr2.predict(slope)))

ratio3 = np.array(ratio3).reshape((len(slope), 1))
regr3 = LinearRegression()
regr3.fit(slope, ratio3)
twin3.plot(S, regr3.predict(S), alpha = 0.5, color='b', linewidth=2) 
p4 = twin3.scatter(np.array(slope), np.array(ratio3), edgecolor = "b", facecolor = 'None', label = 'W*H, R^2 = ' + str('%.3f') % r2_score(ratio3, regr3.predict(slope)))

ax.set_xlim(np.min(slope)-0.005, np.max(slope)+0.005)
ax.set_ylim(np.min(volume) - 5, np.max(volume) + 5)
twin3.set_ylim(np.min(ratio3) - 5, np.max(ratio3) +5)
twin1.set_ylim(np.min(ratio1) - 5, np.max(ratio1) +5)
ax.set_xlabel("Slope (mm/mmHg)")
ax.set_ylabel("Volume (ml)")

twin1.set_ylabel("L*H (cm^2)")
twin2.set_ylabel("L*W (cm^2)")
twin3.set_ylabel("W*H (cm^2)")


ax.yaxis.label.set_color('black')
twin1.yaxis.label.set_color("r")
twin2.yaxis.label.set_color('g')
twin3.yaxis.label.set_color('b')

tkw = dict(size=4, width=1.5)
#plt.legend(frameon = False, fontsize = 15)
#plt.xticks(fontsize=15)
#plt.yticks(fontsize=15)
ax.tick_params(axis='y', colors= 'black', **tkw)
twin1.tick_params(axis='y', colors= 'r', **tkw)
twin2.tick_params(axis='y', colors= 'g', **tkw)
twin3.tick_params(axis='y', colors= 'b', **tkw)
ax.tick_params(axis='x', **tkw)

ax.legend(handles=[p1, p2, p3, p4], loc = 'upper left')

plt.show()

