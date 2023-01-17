import pandas as pd

data1 = pd.read_csv('RedtrainSet.csv')
data2 = pd.read_csv('BluetrainSet.csv')
data3 = pd.read_csv('GreentrainSet.csv')



df3 = pd.concat([data1, data2], axis=0)
df3 = pd.concat([df3,data3], axis = 0)
df3 = df3.drop(labels=['Unnamed: 0'],axis = 1)


data4 = pd.read_csv('RedtestSet.csv')
data5 = pd.read_csv('BluetestSet.csv')
data6 = pd.read_csv('GreentestSet.csv')



df4 = pd.concat([data4, data5], axis=0)
df4 = pd.concat([df4,data6], axis = 0)
df4 = df4.drop(labels=['Unnamed: 0'],axis = 1)



df3.to_csv('trainRGB.csv',index =False)
df4.to_csv('testRGB.csv',index =False)
