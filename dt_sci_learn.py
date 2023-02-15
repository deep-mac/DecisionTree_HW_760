from sklearn import tree
import pandas as pd
import sys

f = open(sys.argv[1], "r")
data = f.readlines()
#print(data)
inp = []
label = []
for datastr in data:
    inp_t = datastr.split(" ")
    inp_x = []
    inp_x.append(inp_t[0])
    inp_x.append(inp_t[1])
    inp.append(inp_x)
    label.append(inp_t[2][:-1])
#print (inp)
#print (label)

#X, y = iris.data, iris.target
clf = tree.DecisionTreeClassifier()
clf = clf.fit(inp, label)

df = pd.read_csv(sys.argv[1], sep=" ", header=None)
df.columns = ["0", "1", "label"]
print (df)

x = df["0"].values.tolist()
y = df["1"].values.tolist()
labels = df["label"].values.tolist()
for i in range (df.shape[0]):
    tmp = []
    tmp2 = []
    tmp.append(x[i])
    tmp.append(y[i])
    tmp2.append(tmp)
    yp = clf.predict(tmp2)
    #print(yp[0])
    if (labels[i] != int(yp[0])):
        print ("ERROR in prediction")
