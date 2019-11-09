import pandas as pd
import numpy as np
from apyori import apriori
from sklearn.cluster import dbscan
# 14996 total rows in the dataset
# handy resource: https://github.com/asnr/sas-to-python

df = pd.read_csv("5402_dataset.csv", dtype=str)

print(df.nunique()) # the number of unique values in each column
df.nunique().to_csv("nunique_init.csv")

######################################################
# remove the timestamps column because it is useless #
######################################################
df = df.drop(columns="timestamps")

###########################################################
# replace all "almost" values with "full" values in MV#0# #
###########################################################
cc = "0" # 0 => closed
oo = "1" # 1 => open
# only MV304 has a typo
df['MV304'] = df['MV304'].map({"CLOSED":cc, "OPEN":oo, "ALMOST_CLOSD":cc, "ALMOST_CLOSED":cc, "SEMI_CLOSED":cc, "ALMOST_OPEN":oo, "SEMI_OPEN":oo})
df['MV301'] = df['MV301'].map({"CLOSED":cc, "OPEN":oo, "ALMOST_CLOSED":cc, "SEMI_CLOSED":cc, "ALMOST_OPEN":oo, "SEMI_OPEN":oo})
df['MV302'] = df['MV302'].map({"CLOSED":cc, "OPEN":oo, "ALMOST_CLOSED":cc, "SEMI_CLOSED":cc, "ALMOST_OPEN":oo, "SEMI_OPEN":oo})
df['MV101'] = df['MV101'].map({"CLOSED":cc, "OPEN":oo, "ALMOST_CLOSED":cc, "SEMI_CLOSED":cc, "ALMOST_OPEN":oo, "SEMI_OPEN":oo})
df['MV201'] = df['MV201'].map({"CLOSED":cc, "OPEN":oo, "ALMOST_CLOSED":cc, "SEMI_CLOSED":cc, "ALMOST_OPEN":oo, "SEMI_OPEN":oo})
df['MV303'] = df['MV303'].map({"CLOSED":cc, "OPEN":oo, "ALMOST_CLOSED":cc, "SEMI_CLOSED":cc, "ALMOST_OPEN":oo, "SEMI_OPEN":oo})
# print(df.MV301.value_counts().sort_index()) # proc freq

#################################################################################
# convert the "y's" and "n's" in the is_attack column to 1s and 0s respectively #
#################################################################################
df['is_attack'] = df['is_attack'].map({"0":"0", "1":"1", "N":"0", "Y":"1"})

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

#########################
# remove duplicate rows #
#########################
# duplicated = df[df.duplicated(keep=False)]
# print(duplicated)
df = df.drop_duplicates()

############################################################################
# fix entries which are off by multiples of ten (clearly a recording error #
############################################################################
df_discrete_colnames = [column for column in df if df.nunique()[column] < 30]
df_continuous_colnames = [column for column in df if df.nunique()[column] >= 30]
# insert code here
# some values are excessively large in the dataframe, throwing off the averages
# example: 931.6713 for AIT202 at row 14591
# example: fit301 has values that are orders of magnitude off: 0.0512443, 0.000512443, 0.00000512443


df.to_csv("USE THIS DATASET FOR PCA, CHI SQUARE, AND APRIORI.csv")

###########################################################
# Principle components analysis to remove more attributes #
###########################################################

##############################################################
# correlation, remove attributes which are highly correlated # # do this for the continuous variables
##############################################################
# could use pearson or spearman or correlation matrix
df_discrete = df[df_discrete_colnames] # these are already interpreted as strings
df_continuous = df[df_continuous_colnames].astype(float)
# print(df_continuous.corr())
df_continuous.corr(method="pearson").to_csv("pearson_correlations.csv")
df_continuous.corr(method="spearman").to_csv("spearman_correlations.csv")

#################################################
# chi square tests of independence for discrete #
#################################################

#######################################
# apriori association rules algorithm # NOT WORKING YET
#######################################
df_discrete = df[df_discrete_colnames] # these are already interpreted as strings
df_continuous = df[df_continuous_colnames].astype(float)

records = df_discrete.values.tolist()

association_rules = apriori(records, min_support=0.0045, min_confidence=0.2, min_lift=3, min_length=2)
association_results = list(association_rules)
print(association_results)

# https://stackabuse.com/association-rule-mining-via-apriori-algorithm-in-python/

########################################################################################
# remove attributes determined unecessary by apriori, pca, chi square, and correlation #
########################################################################################
# drop columns once we decide on what to drop

###################
# remove outliers #
###################
df_discrete_colnames = [column for column in df if df.nunique()[column] < 30]
df_continuous_colnames = [column for column in df if df.nunique()[column] >= 30]
df_discrete = df[df_discrete_colnames] # these are already interpreted as strings
df_continuous = df[df_continuous_colnames].astype(float)
# from scipy import stats
# df_continuous = df_continuous[(np.abs(stats.zscore(df_continuous)) < 3).all(axis=1)] # throw out all rows where at least one continuous attribute is not within three standard deviations of the mean
df_continuous_normalized = df_continuous.apply(lambda x: x/x.max(), axis=1) # divide each column by its maximum value

cores, labels = dbscan(df_continuous_normalized, eps = 0.5, min_samples = 150)
df_continuous["classification"] = list(labels[:]) # add the dbscan classification column to the dataset
df_continuous.to_csv("df_continuous.csv")
df_continuous = df_continuous.loc[df_continuous["classification"] != -1].drop(columns="classification") # eliminate all the noise points determined by dbscan
df = df_discrete.join(df_continuous, how="inner") # put the two back together, throwing out all rows in df_discrete that don't have a corresponding row in df_continuous


# final, cleaned dataset
df.to_csv("cleaned_df.csv")