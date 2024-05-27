import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
data1 = pd.read_csv('/Users/changjingbo/Desktop/result/kan/test_loss.csv',names=['kantest_loss'])
data2 = pd.read_csv('/Users/changjingbo/Desktop/result/kan/train_loss.csv',names=['kantrain_loss'])
data3 = pd.read_csv('/Users/changjingbo/Desktop/result/dcnn/train_loss.csv',names=['dcnntrain_loss'])
x1 = np.linspace(4000,4300,300)
x1.reshape(300,1)
y1 = data1[4000:4300]
# y2 = data2[4000:]
# y2 = y2 - 0.0001
# y3 = data3[4000:]

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot()
ax.plot(x1 , y1 , color='k', label = 'KAN_DCNN')
# ax.plot(x1 , y1 , color='SkyBlue', label = 'TestSet')
# ax.plot(x1 , y2 , color = 'LimeGreen' , label ='TrainSet')
# ax.plot(x1 , y3 , color = 'MediumOrchid' , label ='DCNN')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()