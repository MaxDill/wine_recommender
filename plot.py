import matplotlib.pyplot as plt
import seaborn as sns

# [0.19509156839723768, 0.1687602783082704, 0.15764858165423817, 0.15006848606167839, 0.14850310181802115]

# Wines dataset taster-specific RS taste-related features only
raw_data1 = [(0.18912689582517203, 0.0014523067077875339), (0.15941375545254052, 0.0013027270626956937), (0.15621290986335434, 0.0009652146514093564), (0.15829555216449895, 0.000766458157006716), (0.15785796122951873, 0.0005151330598457043)]

# Wines dataset global RS taste-related features only
#raw_data2 = [(0.20894182181823556, 0.003075895477116529), (0.17853339820327688, 0.0018520486819711384), (0.17232119960303507, 0.0007895344077408466), (0.16881601857684345, 0.00026181951615423236), (0.1696734227130918, 0.0003192612791211413)]

# Threshold of .99
#raw_data5 = [(0.19251592670276962, 0.001921339727132408), (0.18258517043527187, 0.0020593439331325044), (0.18342574294846437, 0.0017935583715740274), (0.1803901974818608, 0.0010939726701274643), (0.18041894001811323, 0.00032586760822430824)]

# Threshold of .7
#raw_data4 = [(0.1883743348347076, 0.0011845494464899633), (0.1643731203217471, 0.00086779551243825), (0.1574442360758403, 0.0005865298090496308), (0.1588272209073012, 0.00021555268069762182), (0.15950441446929134, 0.00040719488621936855)]

# Treshold of .8
raw_data2 = [(0.19142023463042887, 0.003100644573614243), (0.1645315204293458, 0.0014095723850388537), (0.1627387244994478 , 0.0003823369357987597), (0.15982079371450927, 0.0007643117042925703), (0.16082747582161552, 0.0010173554557865452)]
# Extend feature selection (nation and type)
#raw_data2 = [(0.18591385169713118, 0.0014864039108083878), (0.16700200433819, 0.0016296378660600435), (0.1610308152470845, 0.0008814322729360381), (0.15704033682168994, 0.0007350483041337857), (0.15636016893785304, 0.0005461863163322801)]

# Review dataset only training
#raw_data2 = [0.19509156839723768, 0.1687602783082704, 0.15764858165423817, 0.15006848606167839, 0.14850310181802115]

# Wines dataset random RS taste-related features only
#raw_data3 = [(0.3392680322827317, 0.0007214621206006272), (0.3403347024868394, 0.0023703322127400033), (0.3396447418992782, 0.0014194543568927973), (0.3399517820433096, 0.0014530176335575986), (0.33988323979274687, 0.0011920446698172977)]

# Threshold of .95
raw_data3 = [(0.1776998785627553, 0.0021019444195709016), (0.16117373260933132, 0.001080166526521576), (0.15967216463244988, 0.0009951356665101351), (0.15807051515849452, 0.0005504464832401083), (0.1581827046701867, 0.000251258171390367)]

y = [1,3,5,10,15]
x1 = [i[0] for i in raw_data1]
x2 = [i[0] for i in raw_data2]
x3 = [i[0] for i in raw_data3]
x4 = [i[0] for i in raw_data4]
x5 = [i[0] for i in raw_data5]
#
plt.plot(y, x1, label='.9')
plt.plot(y, x2, label = '.8')
plt.plot(y, x3, label = '.95')
plt.plot(y, x4, label='.7')
plt.plot(y, x5, label = '.99')
#
plt.xlabel("k")
plt.ylabel("RMSE")
plt.title("RMSE for different k values and different threshold values")

plt.legend()
plt.show()

def plot_distribution(x, bins, title, ylabel):
    plt.hist(x, bins = bins)
    plt.gca().set(title=title, ylabel=ylabel)
    plt.show()

def plot_scatter_taster_rmse():
    #val = {483: 0.1531696839225115, 185: 0.19487551532197042, 129: 0.18319545795366754, 118: 0.14531222766148558, 137: 0.14696468006477395, 109: 0.15338165274787083, 69: 0.1230695223048074, 37: 0.15971061067509443, 39: 0.11432185814365779, 31: 0.12905976854472506}
    plt.scatter(val.keys(), val.values())
    plt.xlabel('Number of entries')
    plt.ylabel('RMSE')
    plt.title('RMSE for different number of entries')
    plt.show()