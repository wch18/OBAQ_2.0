import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

map_w = 12
map_h = 12

W_bwmap = np.load('./tmp2/W_30.npy',allow_pickle=True)
GA_bwmap = np.load('./tmp/GA_10.npy', allow_pickle=True)

def bwmap_trans_G(bwmap):
    # return np.expand_dims((1-bwmap[:map_w, :map_h]/4*180/255), 2)
    # mask = 1 - ((np.random.rand(map_w, map_h) > 0.4) & (bwmap[:map_w, :map_h] < 2.4))

    return np.expand_dims((1-(np.round(bwmap[:map_w, :map_h])/4*180/255)), 2)

R = np.ones(shape=(map_w, map_h, 1))
# G = 1 - (np.random.rand(map_w, map_h, 1)>0.8)*180/255
# G = 1 - np.ones(shape=(map_w, map_h, 1)) * 180/255
G = bwmap_trans_G(W_bwmap)
# G = 1 - np.array(np.arange(0,10,2).reshape(5,1,1))/8*180/255
# print(G)
print(R.shape, G.shape)
# print()

B = np.ones(shape=(map_w, map_h, 1))
# RGB_map = np.array(R,G,B)
RGB_map = np.concatenate((R,G,B), -1)
print(RGB_map[:,:,0])
print(RGB_map.dtype)
# print(RGB_map)
ax = plt.gca()
ax.set_xticks(np.arange(-.5, map_w, 1))
ax.set_yticks(np.arange(-.5, map_h, 1))
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
ax.grid(color='k', linestyle='-', linewidth=1)
plt.imshow(RGB_map, interpolation='none')
plt.savefig('test.png')