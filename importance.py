import numpy as np
import matplotlib.pyplot as plt

# new
# mt = [202, 278, 198, 200, 320, 302]
# el = [140, 240, 244, 210, 342, 324]
# pos = [116, 248, 267, 242, 316, 311]
# dep = [331, 268, 174, 166, 221, 340]

# baseline
mt = [206, 390,  63, 277, 419, 145]
el = [119, 247, 419, 159, 218, 338]
pos = [126, 320, 257, 158, 208, 431]
dep = [246, 259, 306, 158, 208, 323]





features = ['GENETIC','SYNTACTIC','FEATURAL','PHONOLOGICAL','INVENTORY','GEOGRAPHIC']

# Normalize importance values
mt = np.array(mt) / np.sum(mt)
el = np.array(el) / np.sum(el)
pos = np.array(pos) / np.sum(pos)
dep = np.array(dep) / np.sum(dep)

x = np.arange(len(features))  # the label locations
width = 0.15  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.barh(x - width*1.5, mt, width, label='MT')
rects2 = ax.barh(x - width*0.5, el, width, label='EL')
rects3 = ax.barh(x + width*0.5, pos, width, label='POS')
rects4 = ax.barh(x + width*1.5, dep, width, label='DEP')

# Add some text for labels, title and axes ticks
ax.set_xlabel('Normalized Importance')
ax.set_yticks(x)
ax.set_yticklabels(features)
ax.legend()

plt.tight_layout()
plt.show()