import pandas as pd
import math

df = pd.read_csv("5402_dataset.csv", dtype=str)

# This is the grouping that we're considering
cc = "1" # 1 => closed
oo = "2" # 2 => open

df = df.replace("ALMOST_CLOSD", "ALMOST_CLOSED") # only MV304 has a typo

df2 = df.copy()
df2['MV304'] = df['MV304'].map({"CLOSED":cc, "OPEN":oo, "ALMOST_CLOSED":cc, "SEMI_CLOSED":cc, "ALMOST_OPEN":oo, "SEMI_OPEN":oo})
df2['MV301'] = df['MV301'].map({"CLOSED":cc, "OPEN":oo, "ALMOST_CLOSED":cc, "SEMI_CLOSED":cc, "ALMOST_OPEN":oo, "SEMI_OPEN":oo})
df2['MV302'] = df['MV302'].map({"CLOSED":cc, "OPEN":oo, "ALMOST_CLOSED":cc, "SEMI_CLOSED":cc, "ALMOST_OPEN":oo, "SEMI_OPEN":oo})
df2['MV101'] = df['MV101'].map({"CLOSED":cc, "OPEN":oo, "ALMOST_CLOSED":cc, "SEMI_CLOSED":cc, "ALMOST_OPEN":oo, "SEMI_OPEN":oo})
df2['MV201'] = df['MV201'].map({"CLOSED":cc, "OPEN":oo, "ALMOST_CLOSED":cc, "SEMI_CLOSED":cc, "ALMOST_OPEN":oo, "SEMI_OPEN":oo})
df2['MV303'] = df['MV303'].map({"CLOSED":cc, "OPEN":oo, "ALMOST_CLOSED":cc, "SEMI_CLOSED":cc, "ALMOST_OPEN":oo, "SEMI_OPEN":oo})

######################################
# replace NAs with most common value #
######################################
df = df.fillna(df.mode().iloc[0]) # https://stackoverflow.com/a/32619781
df2 = df2.fillna(df2.mode().iloc[0]) # https://stackoverflow.com/a/32619781

############################
# compute entropy function #
############################
def compute_entropy(attr, dff, nam):
  dist = {ii:[0,0,0] for ii in attr.unique()}
  for index, row in dff.iterrows():
    dist[row[nam]][0] += 1
    if row["is_attack"] == '1':
      dist[row[nam]][1] += 1 # closed
    else:
      dist[row[nam]][2] += 1 # open

  entropyAfterSplit = 0
  for key, val in dist.items():
    nn, pnatk, pyatk = val
    entropynatk = (pnatk/nn)*math.log(pnatk/nn, 2) if pnatk != 0 else 0
    entropyyatk = (pyatk/nn)*math.log(pyatk/nn, 2) if pyatk != 0 else 0
    entropyAfterSplit += (-entropynatk - entropyyatk)*(pnatk + pyatk)/nn
  # print(dist)
  return entropyAfterSplit


mylist = [(df.MV301, df2.MV301, "MV301"), (df.MV304, df2.MV304, "MV304"), (df.MV302, df2.MV302, "MV302"), (df.MV101, df2.MV101, "MV101"), (df.MV201, df2.MV201, "MV201"), (df.MV303, df2.MV303, "MV303")]

for ungrp, grp, nam in mylist:
  result = compute_entropy(ungrp, df, nam) - compute_entropy(grp, df2, nam)
  if result > 0:
    print("{} grouped is better: entrop_ungrp - entrop_grp = ".format(nam), end = '')
  else:
    print("{} ungrouped is better: entrop_ungrp - entrop_grp = ".format(nam), end = '')
  print(result)
