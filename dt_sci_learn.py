from sklearn import tree
import pandas as pd
import sys

#f = open(sys.argv[1], "r")
#data = f.readlines()
##print(data)
#inp = []
#label = []

def getErr (clf, df_test):
    x = df_test["0"].values.tolist()
    y = df_test["1"].values.tolist()
    llabels = df_test["label"].values.tolist()
    err = 0
    for i in range (df_test.shape[0]):
        tmp = []
        tmp2 = []
        tmp.append(x[i])
        tmp.append(y[i])
        tmp2.append(tmp)
        yp = clf.predict(tmp2)
        #print(yp[0])
        if (llabels[i] != int(yp[0])):
            err = err+1
    return err/df_test.shape[0]

df = pd.read_csv(sys.argv[1], sep=" ", header=None)
df.columns = ["0", "1", "label"]

df_train = df.sample(n=8192)
print (df_train)
df_test = df.drop(df_train.index)
print (df_test)

df_train0 = df_train.sample(n=32)
df_train_rem = df_train.drop(df_train0.index)
x_train0 = df_train0["0"].values.tolist()
y_train0 = df_train0["1"].values.tolist()
labels = df_train0["label"].values.tolist()
inp = list(zip(x_train0, y_train0))

print (len(inp), len(labels))
print(df_train0.shape[0])
clf = tree.DecisionTreeClassifier()
clf = clf.fit(inp, labels)
print ("total nodes = ", clf.tree_.node_count)
err = getErr(clf, df_test)
print("Err = ", err)

df_train1 = df_train_rem.sample(n=96)
df_train_rem = df_train_rem.drop(df_train1.index)
df_merge = [df_train0, df_train1]
df_train1 = pd.concat(df_merge)
x_train1 = df_train1["0"].values.tolist()
y_train1 = df_train1["1"].values.tolist()
labels = df_train1["label"].values.tolist()
inp = list(zip(x_train1, y_train1))

print(df_train1.shape[0])
clf = tree.DecisionTreeClassifier()
clf = clf.fit(inp, labels)
print ("total nodes = ", clf.tree_.node_count)
err = getErr(clf, df_test)
print("Err = ", err)

df_train2 = df_train_rem.sample(n=384)
df_train_rem = df_train_rem.drop(df_train2.index)
df_merge = [df_train1, df_train2]
df_train2 = pd.concat(df_merge)
x_train2 = df_train2["0"].values.tolist()
y_train2 = df_train2["1"].values.tolist()
labels = df_train2["label"].values.tolist()
inp = list(zip(x_train2, y_train2))

print(df_train2.shape[0])
clf = tree.DecisionTreeClassifier()
clf = clf.fit(inp, labels)
print ("total nodes = ", clf.tree_.node_count)
err = getErr(clf, df_test)
print("Err = ", err)

df_train3 = df_train_rem.sample(n=1536)
df_train_rem = df_train_rem.drop(df_train3.index)
df_merge = [df_train2, df_train3]
df_train3 = pd.concat(df_merge)
x_train3 = df_train3["0"].values.tolist()
y_train3 = df_train3["1"].values.tolist()
labels = df_train3["label"].values.tolist()
inp = list(zip(x_train3, y_train3))

print(df_train3.shape[0])
clf = tree.DecisionTreeClassifier()
clf = clf.fit(inp, labels)
print ("total nodes = ", clf.tree_.node_count)
err = getErr(clf, df_test)
print("Err = ", err)

df_merge = [df_train3, df_train_rem]
df_train4 = pd.concat(df_merge)
x_train4 = df_train4["0"].values.tolist()
y_train4 = df_train4["1"].values.tolist()
labels = df_train4["label"].values.tolist()
inp = list(zip(x_train4, y_train4))

print(df_train4.shape[0])
clf = tree.DecisionTreeClassifier()
clf = clf.fit(inp, labels)
print ("total nodes = ", clf.tree_.node_count)
err = getErr(clf, df_test)
print("Err = ", err)
