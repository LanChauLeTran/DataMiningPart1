import pandas as pd
import numpy as np
# 14996 total rows in the dataset
# handy resource: https://github.com/asnr/sas-to-python

df = pd.read_csv("5402_dataset.csv", dtype=str)
df.nunique().to_csv("nunique_init.csv")


# df['AIT201'] = df['']



for count,float(value) in enumerate(df['AIT201']):
	if value<2:
		df['AIT201'][count]=value*100
	elif value>200:
		df['AIT201'][count]=(value/100)
	elif value<200 and value>2:
		df['AIT201'][count]=value
	else:
		df['AIT201'][count]='NA'
# print(df['AIT201'])
print min(df['AIT201'])
print max(df['AIT201'])

for count,float(value) in enumerate(df['FIT601']):
	if value<.00001:
		df['FIT601'][count]=value*100000
	elif value<.0001:
		df['FIT601'][count]=value*10000
	elif value<.001:
		df['FIT601'][count]=value*1000
	elif value<.01:
		df['FIT601'][count]=value*100
	elif value<.1:
		df['FIT601'][count]=value*10
	elif value<.1 and value>10:
		df['FIT601'][count]=value
	else:
		df['FIT601'][count]='NA'
# print(df['FIT601'])
print min(df['FIT601'])
print max(df['FIT601'])

for count,float(value) in enumerate(df['PIT503']):
	if value<2:
		df['PIT503'][count]=value*100
	elif value>200:
		df['PIT503'][count]=(value/100)
	elif value<200 and value>2:
		df['PIT503'][count]=value
	else:
		df['PIT503'][count]='NA'
print min(df['PIT503'])
print max(df['PIT503'])

for count,float(value) in enumerate(df['PIT502']):
	if value<1:
		df['PIT502'][count]=value*10000
	elif value<30:
		df['PIT502'][count]=value*100
	elif value>800:
		df['PIT502'][count]=(value/100)
	elif value<30 and value>800:
		df['PIT502'][count]=value
	else:
		df['PIT502'][count]='NA'
print min(df['PIT502'])
print max(df['PIT502'])


# ######################################################
# # remove the timestamps column because it is useless #
# ######################################################
# df = df.drop(columns="timestamps")

# ###########################################################
# # replace all "almost" values with "full" values in MV#0# #
# ###########################################################
# cc = "1" # 1 => closed
# oo = "2" # 2 => open
# # only MV304 has a typo
# df['MV304'] = df['MV304'].map({"CLOSED":cc, "OPEN":oo, "ALMOST_CLOSD":cc, "ALMOST_CLOSED":cc, "SEMI_CLOSED":cc, "ALMOST_OPEN":oo, "SEMI_OPEN":oo})
# df['MV301'] = df['MV301'].map({"CLOSED":cc, "OPEN":oo, "ALMOST_CLOSED":cc, "SEMI_CLOSED":cc, "ALMOST_OPEN":oo, "SEMI_OPEN":oo})
# df['MV302'] = df['MV302'].map({"CLOSED":cc, "OPEN":oo, "ALMOST_CLOSED":cc, "SEMI_CLOSED":cc, "ALMOST_OPEN":oo, "SEMI_OPEN":oo})
# df['MV101'] = df['MV101'].map({"CLOSED":cc, "OPEN":oo, "ALMOST_CLOSED":cc, "SEMI_CLOSED":cc, "ALMOST_OPEN":oo, "SEMI_OPEN":oo})
# df['MV201'] = df['MV201'].map({"CLOSED":cc, "OPEN":oo, "ALMOST_CLOSED":cc, "SEMI_CLOSED":cc, "ALMOST_OPEN":oo, "SEMI_OPEN":oo})
# df['MV303'] = df['MV303'].map({"CLOSED":cc, "OPEN":oo, "ALMOST_CLOSED":cc, "SEMI_CLOSED":cc, "ALMOST_OPEN":oo, "SEMI_OPEN":oo})
# # print(df.MV301.value_counts().sort_index()) # proc freq

# #################################################################################
# # convert the "y's" and "n's" in the is_attack column to 2s and 1s respectively #
# #################################################################################
# df['is_attack'] = df['is_attack'].map({"0":"1", "1":"2", "N":"1", "Y":"2"})

# ###############################################################################
# # drop all columns with only one unique value (those columns tell us nothing) #
# ############################################################################### 
# columns_to_drop = [column for column in df if df.nunique()[column] == 1]
# df = df.drop(columns=columns_to_drop)

# ########################################################################################
# # convert 1s and 2s to zeros and ones so weka automatically interprets them as nominal #
# ########################################################################################
# def conv(x):
#   if x == "1":
#     return "zero"
#   if x == "2":
#     return "one"
#   else:
#     return x
# df = df.applymap(conv) # i used 1 and 2 originally so that only the categorical attributes are affected at this stage (there are no continuous variables with values of exactly 1 or 2, but there are some with values of 0)

# ######################################
# # replace NAs with most common value #
# ######################################
# # df = df.dropna() # this throws out around 12,000 rows!
# # df.to_csv("out2.csv")
# df = df.fillna(df.mode().iloc[0]) # https://stackoverflow.com/a/32619781

# #########################
# # remove duplicate rows #
# #########################
# # duplicated = df[df.duplicated(keep=False)]
# # print(duplicated)
# df = df.drop_duplicates()

# ############################################################################
# # fix entries which are off by multiples of ten (clearly a recording error #
# ############################################################################
# df_discrete_colnames = [column for column in df if df.nunique()[column] < 3]
# df_continuous_colnames = [column for column in df if df.nunique()[column] >= 3]
# # insert code here
# # some values are excessively large in the dataframe, throwing off the averages
# # example: 931.6713 for AIT202 at row 14591
# # example: fit301 has values that are orders of magnitude off: 0.0512443, 0.000512443, 0.00000512443

# ###############################################################################
# # create output files that we will use outside of Python (such as Weka and R) #
# ###############################################################################
# df_discrete = df[df_discrete_colnames] # these are already interpreted as strings
# df_continuous = df[df_continuous_colnames].astype(float)
# df_discrete.to_csv("df_discrete.csv", index=False)
# df_continuous.corr(method="pearson").to_csv("pearson_correlations.csv")
# df_continuous.corr(method="spearman").to_csv("spearman_correlations.csv")
# df.to_csv("USE THIS DATASET FOR PCA, CHI SQUARE, AND APRIORI.csv")

# ########################################################################################
# # remove attributes determined unecessary by apriori, pca, chi square, and correlation #
# ########################################################################################
# # based on the results from Weka and R, we will then decide which attributes to drop here

# ###################
# # remove outliers #
# ###################
# df_discrete_colnames = [column for column in df if df.nunique()[column] < 3]
# df_continuous_colnames = [column for column in df if df.nunique()[column] >= 3]
# df_discrete = df[df_discrete_colnames] # these are already interpreted as strings
# df_continuous = df[df_continuous_colnames].astype(float)
# # from scipy import stats
# # df_continuous = df_continuous[(np.abs(stats.zscore(df_continuous)) < 3).all(axis=1)] # throw out all rows where at least one continuous attribute is not within three standard deviations of the mean
# df_continuous_normalized = df_continuous.apply(lambda x: x/x.max(), axis=1) # divide each column by its maximum value

# cores, labels = dbscan(df_continuous_normalized, eps = 0.5, min_samples = 150)
# df_continuous["classification"] = list(labels[:]) # add the dbscan classification column to the dataset
# df_continuous.to_csv("df_continuous.csv")
# df_continuous = df_continuous.loc[df_continuous["classification"] != -1].drop(columns="classification") # eliminate all the noise points determined by dbscan
# df = df_discrete.join(df_continuous, how="inner") # put the two back together, throwing out all rows in df_discrete that don't have a corresponding row in df_continuous

# #####################################
# # output the final, cleaned dataset #
# #####################################
df.to_csv("cleaned_df.csv")