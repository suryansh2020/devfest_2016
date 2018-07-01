import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np

XX, YY = np.mgrid[0:4:800j, 0:4:800j]
Z = model.predict(np.c_[XX.ravel(), YY.ravel()])

Z = Z.reshape(XX.shape)
plt.pcolormesh(XX, YY, Z, cmap=plt.cm.Paired)

plt.scatter([d[0] for i, d in enumerate(data) if classes[i] == 0], [d[1] for i, d in enumerate(data) if classes[i] == 0])
plt.scatter([d[0] for i, d in enumerate(data) if classes[i] == 1], [d[1] for i, d in enumerate(data) if classes[i] == 1], color="red")
