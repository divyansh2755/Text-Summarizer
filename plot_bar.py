import numpy as np
import matplotlib.pyplot as plt

N = 10
developed_summ = (0.38788,0.50781,0.47489,0.54167,0.29508,0.45106,0.36364,0.30588,0.4093,0.38017)


ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, developed_summ, width, color='r')

online_summ = (0.70238,0.39333,0.46715,0.37874,0.39456,0.43448,0.45693,0.48175,0.33607,0.42294)

rects2 = ax.bar(ind+width, online_summ, width, color='y')

# add some text for labels, title and axes ticks
ax.set_ylabel('FScore')
ax.set_title('Scores by document and tool used')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10') )

ax.legend( (rects1[0], rects2[0]), ('LSA', 'tools4noobs/summarizer') )



plt.show()
