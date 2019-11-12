import pandas as pd
import numpy as np
from sklearn.cluster import dbscan
import math
# 14996 total rows in the dataset
# handy resource: https://github.com/asnr/sas-to-python

df = pd.read_csv("5402_dataset.csv", dtype=str)
df.nunique().to_csv("nunique_init.csv")

######################################################
# remove the timestamps column because it is useless #
######################################################
df = df.drop(columns="timestamps")

#################################################################################
# convert the "y's" and "n's" in the is_attack column to 2s and 1s respectively #
#################################################################################
df['is_attack'] = df['is_attack'].map({"0":"1", "1":"2", "N":"1", "Y":"2"})

###########################################################
# replace all "almost" values with "full" values in MV#0# #
###########################################################
cc = "1" # 1 => closed
oo = "2" # 2 => open

df = df.replace("ALMOST_CLOSD", "ALMOST_CLOSED") # only MV304 has a typo

# based on the grouping attributes entropy, we found that replacing these leads to lower entropy (higher info gain)
df['MV304'] = df['MV304'].map({"CLOSED":cc, "OPEN":oo, "ALMOST_CLOSED":cc, "SEMI_CLOSED":cc, "ALMOST_OPEN":oo, "SEMI_OPEN":oo})
df['MV301'] = df['MV301'].map({"CLOSED":cc, "OPEN":oo, "ALMOST_CLOSED":cc, "SEMI_CLOSED":cc, "ALMOST_OPEN":oo, "SEMI_OPEN":oo})
df['MV302'] = df['MV302'].map({"CLOSED":cc, "OPEN":oo, "ALMOST_CLOSED":cc, "SEMI_CLOSED":cc, "ALMOST_OPEN":oo, "SEMI_OPEN":oo})
df['MV101'] = df['MV101'].map({"CLOSED":cc, "OPEN":oo, "ALMOST_CLOSED":cc, "SEMI_CLOSED":cc, "ALMOST_OPEN":oo, "SEMI_OPEN":oo})
df['MV201'] = df['MV201'].map({"CLOSED":cc, "OPEN":oo, "ALMOST_CLOSED":cc, "SEMI_CLOSED":cc, "ALMOST_OPEN":oo, "SEMI_OPEN":oo})
df['MV303'] = df['MV303'].map({"CLOSED":cc, "OPEN":oo, "ALMOST_CLOSED":cc, "SEMI_CLOSED":cc, "ALMOST_OPEN":oo, "SEMI_OPEN":oo})
# print(df.MV301.value_counts().sort_index()) # proc freq

###############################################################################
# drop all columns with only one unique value (those columns tell us nothing) #
############################################################################### 
columns_to_drop = [column for column in df if df.nunique()[column] == 1]
df = df.drop(columns=columns_to_drop)

######################################
# replace NAs with most common value #
######################################
# df = df.dropna() # this throws out around 12,000 rows!
# df.to_csv("out2.csv")
df = df.fillna(df.mode().iloc[0]) # https://stackoverflow.com/a/32619781

########################################################################################
# convert 1s and 2s to zeros and ones so weka automatically interprets them as nominal #
########################################################################################
def conv(x):
  if x == "1":
    return "zero"
  if x == "2":
    return "one"
  else:
    return x
df = df.applymap(conv) # i used 1 and 2 originally so that only the categorical attributes are affected at this stage (there are no continuous variables with values of exactly 1 or 2, but there are some with values of 0)

#########################
# remove duplicate rows #
#########################
# duplicated = df[df.duplicated(keep=False)]
# print(duplicated)
df = df.drop_duplicates()

############################################################################
# fix entries which are off by multiples of ten (clearly a recording error #
############################################################################
df_discrete_colnames = [column for column in df if df.nunique()[column] < 3]
df_continuous_colnames = [column for column in df if df.nunique()[column] >= 3]

# some values are excessively large in the dataframe, throwing off the averages
# example: 931.6713 for AIT202 at row 14591
# example: fit301 has values that are orders of magnitude off: 0.0512443, 0.000512443, 0.00000512443
def fix_mags(colnames):
  def magnitude(value): # https://stackoverflow.com/a/52335468
    if value == 0: 
      return 0
    return int(math.floor(math.log10(abs(value))))

  for ind, colname in enumerate(colnames):
    dist = {ii:0 for ii in range(-10, 10)} # dict acting as a distribution of the orders of magnitude
    for index, row in df.iterrows():
      dist[magnitude(float(row[colname]))] += 1
      
    dist_list = sorted(dist.items(), key=lambda x: x[1], reverse=True) # the most common order of magnitude is now listed first
    
    for index, row in df.iterrows():
      mag = magnitude(float(row[colname]))
      if mag != dist_list[0][0]: # if the order of magintude is not the most common order of magnitude, then fix it
        df.loc[index, colname] = float(row[colname])*10**(dist_list[0][0]-mag)
    print("Fixed orders of magnitude for {} (Col {}/{})".format(colname, ind+1, len(colnames)))

fix_mags(df_continuous_colnames)

###############################################################################
# create output files that we will use outside of Python (such as Weka and R) #
###############################################################################
df_discrete = df[df_discrete_colnames] # these are already interpreted as strings
df_continuous = df[df_continuous_colnames].astype(float)
df_discrete.to_csv("df_discrete.csv", index=False)
df_continuous.to_csv("df_continuous.csv", index=False)
df_continuous.corr(method="pearson").to_csv("pearson_correlations.csv")
df_continuous.corr(method="spearman").to_csv("spearman_correlations.csv")
df.to_csv("USE THIS DATASET FOR PCA, CHI SQUARE, AND APRIORI.csv")

########################################################################################
# remove attributes determined unecessary by apriori, pca, chi square, and correlation #
########################################################################################
# based on the results from Weka and R, we will then decide which attributes to drop here
# drop MV301 and MV302 (p values from chi square were nearly zero when tested for indep against MV303)
df = df.drop(columns=["MV301", "MV303", "AIT203", "FIT501", "FIT502", "FIT503", "P401"])

###################
# remove outliers #
###################
df_discrete_colnames = [column for column in df if df.nunique()[column] < 3]
df_continuous_colnames = [column for column in df if df.nunique()[column] >= 3]
df_discrete = df[df_discrete_colnames] # these are already interpreted as strings
df_continuous = df[df_continuous_colnames].astype(float)
# from scipy import stats
# df_continuous = df_continuous[(np.abs(stats.zscore(df_continuous)) < 3).all(axis=1)] # throw out all rows where at least one continuous attribute is not within three standard deviations of the mean
df_continuous_normalized = df_continuous.apply(lambda x: x/x.max(), axis=1) # divide each column by its maximum value

print("Starting dbscan...")
cores, labels = dbscan(df_continuous_normalized, eps = 0.5, min_samples = 150)
df_continuous["classification"] = list(labels[:]) # add the dbscan classification column to the dataset
df_continuous = df_continuous.loc[df_continuous["classification"] != -1].drop(columns="classification") # eliminate all the noise points determined by dbscan
df = df_discrete.join(df_continuous, how="inner") # put the two back together, throwing out all rows in df_discrete that don't have a corresponding row in df_continuous

#####################################
# output the final, cleaned dataset #
#####################################
df.to_csv("cleaned_df.csv")