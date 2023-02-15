import pandas as pd
import sys
import math
import matplotlib.pyplot as plt
import numpy as np

DEBUG = False
COUNT = [8]
class DT:
    def __init__(self):
        self.left = None
        self.right = None
        self.value = None
        self.feature = -1

def print2DUtil(root, space):
 
    # Base case
    if (root == None):
        return
 
    # Increase distance between levels
    space += COUNT[0]
 
    # Process right child first
    print2DUtil(root.right, space)
 
    # Print current node after space
    # count
    #print()
    for i in range(COUNT[0], space):
        print(end=" ")
    print("-> ", root.feature, root.value)
 
    # Process left child
    print2DUtil(root.left, space)
 
# Wrapper over print2DUtil()
 
 
def print2D(root):
 
    # space=[0]
    # Pass initial space count as 0
    print2DUtil(root, 0)
 


def getEntropy(n0, n1):
    h = 0
    p = []
    p.append(n0/(n0+n1))
    p.append(n1/(n0+n1))
    for pi in p:
        if (pi != 0):
            h = h-(pi * math.log2(pi))
    return h

def FindBestSplit(df, c_split):
    if DEBUG:
        print ("Finding best split for df = \n", df)
        print ("With splits = \n", c_split)

    #First find Hy
    #Hy = getEntropy(df["label"].value_counts()[0], df["label"].value_counts()[1])
    Hy = getEntropy(df[df.label == 0].shape[0], df[df.label == 1].shape[0])
    #print (df[df.label == 0].shape[0])
    HyFeature = []
    ValueFeature = []
    for i in range(num_features):
        df_t = df.sort_values(by=[str(i)])
        if DEBUG:
            print("df_t sorted = \n", df_t)
        HyF_dict = {}
        val_dict = {}
        for sp in c_split[i]:
            value = df_t.iloc[sp, i]
            df_left = df_t[df_t[str(i)] >= value]
            df_lcount = df_left.shape[0];
            HyLeft = getEntropy(df_left[df_left.label == 0].shape[0], df_left[df_left.label == 1].shape[0])
            df_right = df_t[df_t[str(i)] <  value]
            df_rcount = df_right.shape[0];
            if (df_lcount == 0 or df_rcount == 0):
                continue
            HyRight = getEntropy(df_right[df_right.label == 0].shape[0], df_right[df_right.label == 1].shape[0])
            HyF = df_left.shape[0]/(df_left.shape[0]+df_right.shape[0])*HyLeft + df_right.shape[0]/(df_left.shape[0]+df_right.shape[0])*HyRight
            Hs = getEntropy(df_lcount, df_rcount)
            HyF_dict[sp] = (Hy-HyF)/Hs
            val_dict[sp] = value
        HyFeature.append(HyF_dict)
        ValueFeature.append(val_dict)

    if DEBUG:
        print ("Hy = ", Hy)
        print ("HyFeature = ", HyFeature)
        print ("ValueFeature = ", ValueFeature)
    minSplit = []
    for i in range(num_features):
        if(HyFeature[i]):
            minSplitKey =  max(HyFeature[i], key= lambda x: HyFeature[i][x])
            minSplit.append(minSplitKey)
        else:
            minSplit.append(-1)

    if DEBUG:
        print ("minSplit = ", minSplit)

    if (HyFeature[1] and HyFeature[0]):
        if (HyFeature[1][minSplit[1]] >  HyFeature[0][minSplit[0]]):
            feature = 1
            value = ValueFeature[1][minSplit[1]]
        else:
            feature = 0
            value = ValueFeature[0][minSplit[0]]
    elif (HyFeature[1]):
        feature = 1
        value = ValueFeature[1][minSplit[1]]
    elif (HyFeature[0]):
        feature = 0
        value = ValueFeature[0][minSplit[0]]
    else:
        print("ERROR cannot split, the program will crash")
        
    if DEBUG:
        print ("return feature = ", feature)
        print ("value = ", value)
        print ("minSplit = ", minSplit)
    return feature,value

def DetermineCandidateSplit(df):
    #print ("Finding split for df = \n", df)
    c_split = []
    if df.empty: 
        return c_split
    for i in range (num_features):
        df_t = df.sort_values(by=[str(i)])
        #print (df_t)
        label_0 = df_t.iloc[0]["label"]
        curr_feature_splits = []
        for j in range (df_t.shape[0]): #this just gives total number of rows
            if (label_0 != df_t.iloc[j]["label"]):
                curr_feature_splits.append(j);
                label_0 = df_t.iloc[j]["label"]
        c_split.append(curr_feature_splits)
    return c_split


def MakeTree (df):
    #Stopping criteria
    c_split = []
    c_split = DetermineCandidateSplit(df)
    if DEBUG:
        print("Inside MakeTree, df = \n", df)
        print("csplit = ", c_split)
    node = DT()

    if not any(c_split):
        node.value = df.iloc[0]["label"]
        node.feature = 'L'
        node.left = None
        node.right = None
        print("Child node")
    else:
        print("Subtree node")
        feature,value = FindBestSplit(df, c_split)
        df_t = df.sort_values(by=str(feature))
        df_t0 = df_t[df_t[str(feature)] >= value]
        node0 = MakeTree(df_t0)
        df_t1 = df_t[df_t[str(feature)] < value]
        node1 = MakeTree(df_t1)
        node.left = node0
        node.right = node1
        node.feature = feature
        node.value = value
    return node

def predict (root, value):
    f0 = value[0]
    f1 = value[1]
    curr_root = root;
    stop = 0
    while (not stop or (curr_root == None)):
        if (curr_root.feature == 'L'):
            return curr_root.value
        elif (value[curr_root.feature] >= curr_root.value):
            next_root = curr_root.left
        elif(value[curr_root.feature] < curr_root.value):
            next_root = curr_root.right
        else:
            next_root = None
        curr_root = next_root

def totalNodes(root):
    if(root == None):
        return 0
 
    l = totalNodes(root.left)
    r = totalNodes(root.right)
 
    return 1 + l + r

num_features = 2
df = pd.read_csv(sys.argv[1], sep=" ", header=None)
df.columns = ["0", "1", "label"]

DTree = MakeTree(df)
print2D(DTree)
print("Total nodes = ", totalNodes(DTree))

x = df["0"].values.tolist()
y = df["1"].values.tolist()
labels = df["label"].values.tolist()
for i in range (df.shape[0]):
    tmp = []
    tmp.append(x[i])
    tmp.append(y[i])
    yp = predict(DTree, tmp)
    if (labels[i] != yp):
        print ("ERROR in prediction")

#Plot
#data scatter plot
lab0 = df[df["label"] == 0]
x0 = lab0["0"].values.tolist()
y0 = lab0["1"].values.tolist()
lab1 = df[df["label"] == 1]
x1 = lab1["0"].values.tolist()
y1 = lab1["1"].values.tolist()
#fig,ax = plt.subplots()
plt.scatter(x0, y0, c = "blue")
plt.scatter(x1, y1, c = "red")
plt.savefig('q6_'+sys.argv[1]+'.pdf', format="pdf")


#decision boundary scratter plot
x = np.random.uniform(0.0, 1.0, 10000)
y = np.random.uniform(0.0, 1.0, 10000)
pred = []
for i in range (len(x)):
    tmp = []
    tmp.append(x[i])
    tmp.append(y[i])
    yp = predict(DTree, tmp)
    pred.append(yp)

df = pd.DataFrame(list(zip(x, y, pred)),
               columns =['0', '1', 'label'])
lab0 = df[df["label"] == 0]
x0 = lab0["0"].values.tolist()
y0 = lab0["1"].values.tolist()
lab1 = df[df["label"] == 1]
x1 = lab1["0"].values.tolist()
y1 = lab1["1"].values.tolist()
#fig,ax = plt.subplots()
plt.scatter(x0, y0, c = "blue")
plt.scatter(x1, y1, c = "red")
plt.savefig('q6_boundary_'+sys.argv[1]+'.pdf', format="pdf")

