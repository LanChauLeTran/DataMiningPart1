import pandas as pd
from scipy import stats
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import dbscan
# 14996 total rows in the dataset

df = pd.read_csv("5402_dataset.csv", dtype=str)

print(df.nunique()) # the number of unique values in each column
df.nunique().to_csv("nunique.csv")

######################################################
# remove the timestamps column because it is useless #
######################################################
df = df.drop(columns="timestamps")

#################################################################################
# convert the "y's" and "n's" in the is_attack column to 1s and 0s respectively #
#################################################################################
df['is_attack'] = df['is_attack'].map({"0":"0", "1":"1", "N":"0", "Y":"1"})

###############################################################################
# drop all columns with only one unique value (those columns tell us nothing) #
############################################################################### 
columns_to_drop = [column for column in df if df.nunique()[column] == 1]
df = df.drop(columns=columns_to_drop)

####################################
# replace NAs with most common value
####################################
df = df.dropna()
# df.to_csv("out2.csv")
df = df.fillna(df.mode().iloc[0]) # https://stackoverflow.com/a/32619781

#########################
# remove duplicate rows #
#########################
# duplicated = df[df.duplicated(keep=False)]
# print(duplicated)
df = df.drop_duplicates()

###################
# remove outliers #
###################
df_discrete_colnames = [column for column in df if df.nunique()[column] < 30]
df_continuous_colnames = [column for column in df if df.nunique()[column] >= 30]
df_discrete = df[df_discrete_colnames] # these are already interpreted as strings
df_continuous = df[df_continuous_colnames].astype(float)
df_continuous = df_continuous[(np.abs(stats.zscore(df_continuous)) < 3).all(axis=1)] # throw out all rows where at least one continuous attribute is not within three standard deviations of the mean
# df_continuous = df_continuous.divide(df_continuous.max(), axis=0)
df_continuous_normalized = df_continuous.apply(lambda x: x/x.max(), axis=1)

cores, labels = dbscan(df_continuous_normalized, eps = 0.2, min_samples = 10)
df_continuous["classification"] = list(labels[:]) # add the dbscan classification column to the dataset
df_continuous.to_csv("df_continuous.csv")
df_continuous = df_continuous.loc[df_continuous["classification"] != -1].drop("classification") # eliminate all the noise points determined by dbscan

df = df_discrete.join(df_continuous) # put the two back together
# print(labels)

# kmeans = KMeans(n_clusters=3).fit(df_continuous)
# df_continuous["kmeans"] = kmeans.labels_[:]
# print(kmeans.labels_)



# df_continuous.to_csv("df_continuous.csv")

# may need to do regression analysis to do this???
# some values are excessively large in the dataframe, throwing off the averages
# example: 931.6713 for AIT202 at row 14591
# example: fit301 has values that are orders of magnitude off: 0.0512443, 0.000512443, 0.00000512443

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




df.to_csv("cleaned_df.csv")