# Modeling standard plenoptic camera by an equivalent camera array
To render camera array images use [this](https://github.com/ehedayati/blender-addon). Seahorse5x5 is an example scene.

# Example usage
```
from LFProcess import *
#loading and generating 4D light field
sceneName='seahorse5x5'
lf = LF(sceneName)
lf.storeLf(1000000)
lf.loadLf()
print(lf.lf.shape)

# Center view
myAif = lf.allInFocusImage()

# Depth of field Image
myDofOrg = lf.depthOfFieldImage(lf.aperture,False)

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
RGBmyAif = cv2.cvtColor(myAif.astype(np.uint8), cv2.COLOR_BGR2RGB)
RGBmyDOF = cv2.cvtColor(myDofOrg.astype(np.uint8), cv2.COLOR_BGR2RGB)
ax1.imshow(RGBmyAif)
ax1.axis('off')
ax2.imshow(RGBmyDOF)
ax2.axis('off')
plt.show()

# Refocusing with alpha = -0.4
lf.refocus(-.4)

# Depth of field of the refocused LF
myDof = lf.depthOfFieldImage(lf.aperture,True)

plt.imshow(cv2.cvtColor(myDof.astype(np.uint8), cv2.COLOR_BGR2RGB))
plt.show()

```

This simple code is based on the work of Eisa Hedayati.
If you used this code please cite our paper "Modeling standard plenoptic camera by an equivalent camera array"

@article{}

@inproceedings{hedayati2018simulation,
  title={Simulation of light fields captured by a plenoptic camera using an equivalent camera array},
  author={Hedayati, Eisa and Bos, Jeremy P},
  booktitle={Laser Communication and Propagation through the Atmosphere and Oceans VII},
  volume={10770},
  pages={107700R},
  year={2018},
  organization={International Society for Optics and Photonics}
}

