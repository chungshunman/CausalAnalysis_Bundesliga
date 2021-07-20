"""
Microeconometrics: Self Study.

Group Assignment:

Alec Eisenkolb, Chung-Shun Man, Nicolas Greber, Tim Hug

Spring Semester 2021.

University of St. Gallen.
"""

# import modules
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# if the below line to set the working directory does not work, uncomment the
# following line
# __file__ = "self_study.py"

# set working directory to current directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# import own functions
import self_study_functions as ssf

# set the seep for replicability
np.random.seed(270421)

# define name of data
DATA = 'data_group_1.csv'

# load in the data
data = pd.read_csv(DATA)


#%%

# Recode the outcome and treatment variable

# outcome variable: 'FullResult' -> 'HomeClasses':
data['HomeClasses'] = data['FullResult'].apply(ssf.outcome_class)

# treatment: 'HomeValue' & 'AwayValue' -> 'HomeHigherVal':
# dummy variable with 1 if the home team has higher value
data['HomeHigherVal'] = (data['HomeValue'] > data['AwayValue']).astype(int)

# Rescale selected coefficients
scale1000 = ['Attendance', 'Capacity']
scaleMio = ['HomeValue', 'AwayValue', 'AvgValueHome', 'MedianValueHome',
            'AvgValueAway', 'MedianValueAway', 'TVHome', 'TVAway']

data[scale1000] = data[scale1000] / 1_000
data[scaleMio] = data[scaleMio] / 1_000_000


#%%

# Data analysis ('Raw Data')

# descriptive statistics
ssf.df_summary_num(data)
ssf.df_summary_char(data)

# Check distribution of binary variable
treatment_table = ssf.binary_share(data, "HomeHigherVal")
# export table to latex
treatment_table.to_latex()

# Check mean values of "HomeHigherVal" for Match Outcomes
classes_table = ssf.mean_by(data, "HomeHigherVal", "HomeClasses")
# export to latex
classes_table.to_latex()

# check outcome variable using a barplot of the classes
ssf.barplot(data, 'HomeClasses', ['Loss', 'Draw', 'Win'], 'documentation/img/')

# check imbalance
total_df, diff_df = ssf.home_away_check(data, 'Season', 'HomeTeam', 'AwayTeam')

# plot single histogram
var_names = ['TransportTime', 'Attendance', 'Capacity']
for name in var_names:
    ssf.histogram(data, name)
    
# line plot of change in market value for each team
ssf.time_name(data, 'Season', 'HomeTeam', 'HomeValue', missing_allowed=15)

# Preparations for dual histogram
names = ['Home', 'Away']
# these names appear first as variables. i.e. HHIHome
var_names_first = ['HHI', 'AvgValue', 'MedianValue', 'TV']
# these names appear first as variables. i.e. HomeScore
var_names_second = ['Score', 'Shots', 'ShotsTarget', 'Fouls', 'Corners',
                  'Yellows', 'Value', 'GDP', 'Unempl']

# plot dual histograms
ssf.dual_histogram(data, var_names_first, names, first_name=True)

ssf.dual_histogram(data, var_names_second, names, first_name=False)

#%%

# Differenced variables (Home - Away)

# define variables in scope
names = ['Home', 'Away']
var_first = ['GDP', 'Unempl']
var_second = ['dHHI', 'AvgValue', 'TV', 'AvgAge', 'StdvAge', 'AvgHeight',
              'StdvHeight']

# iterate over the variables and compute differences
data = ssf.var_diff(data, names, var_first, var_second)


#%%

# Logistic regression

# define covariates used
outcome = 'HomeClasses'
treatment = 'HomeHigherVal'
covariates = ['Attendance', 'HomeShots', 'AwayShots', 'HomeFouls', 'AwayFouls', 
              'HomeCorners', 'AwayCorners', 'HomeYellows', 'AwayYellows', 
              'HomeReds', 'AwayReds', 'AvgAgeHome', 'StdvAgeHome', 
              'AvgAgeAway', 'StdvAgeAway', 'BothHome', 'BothAway', 
              'AvgHeightHome', 'StdvHeightHome', 'AvgHeightAway', 
              'StdvHeightAway', 'HomeChampion', 'AwayChampion', 
              'HomeRelegated', 'AwayRrelegated', 'DiffGDP', 
              'DiffUnempl', 'DiffTV']

# Encode the seasons variable into dummy variables
data = ssf.oneHotEncoding(data, 'Season')

# collect the names of the season dummies into a list
season = []
for s in np.sort(data['Season'].unique())[:-1]:
    season.append('Season_' + s)
covWithTime = covariates + season

X_name = np.insert(covWithTime, 0, treatment)

# transform data frame into two arrays corresponding to the observations of 
# covariates and outcomes
y = np.array(data[outcome])
X = np.array(data[X_name])

# get number of classes
n = len(np.unique(y))

# fit logistic regression (GD)
results = ssf.fitLogistic(X, y, n, display=False)
coefficients = results[:-(n - 1)]
thresholds = results[-(n - 1):]
# get value of objective function
print(ssf.objMargin(results, X, y, n))

# fit logistic regression using scipy optimizer (L-BFGS-B)
resultsBFGS = ssf.fitLogistic(X, y, n, optim='L-BFGS-B', display=False)
coefficientsBFGS = resultsBFGS[:-(n - 1)]
thresholdsBFGS = resultsBFGS[-(n - 1):]
# get value of objective function
print(ssf.objMargin(resultsBFGS, X, y, n))

labels = ['0 point', '1 point', '3 points']

# calculate mean marginal effects
mean_margeffect = ssf.mean_margeff(X, coefficients, thresholds)

mean_margeffectBFGS = ssf.mean_margeff(X, coefficientsBFGS, thresholdsBFGS)

# expected additional points when team is more expensive
mme_ologit = np.dot(mean_margeffect, np.array([0, 1, 3]))

mme_ologitBFGS = np.dot(mean_margeffectBFGS, np.array([0, 1, 3]))

# probabilities of the outcome classes
probs = ssf.logistic_probs(X, coefficients, thresholds)

probsBFGS = ssf.logistic_probs(X, coefficientsBFGS, thresholdsBFGS)

# compute the estimated classes for each observation
estimates_ologit = np.argmax(probs, axis=1)

estimates_ologitBFGS = np.argmax(probsBFGS, axis = 1)

#%%

# Ordered random forest

# Obtain the outcome and covariates as pd.Series and pd.DataFrame respectively
y2 = (pd.Series(y, name='HomeClasses')+1).copy()
X2 = data[np.insert(covWithTime, 0, treatment)].copy()

np.random.seed(1234)
# Get the ordered random forest models and probability for each class
output = ssf.ordered_forest(X2, y2, labels)

# Get the mean marginal effects of all the covariates
mean_margeffect_orf = ssf.marginal_eff(output, X2, labels, mean_mf=True)

# expected additional points when team is more expensive
mme_orf = np.dot(mean_margeffect_orf.loc[treatment], np.array([0, 1, 3]))

# compute the estimated classes for each observation
estimates_ORF = np.argmax(np.array(output["probs"]), axis=1)


#%%

# Compute summary statistics of only covariates / treatment / outcome
# Define variables
variables_summary = [outcome] + [treatment] + covWithTime 

# define seperate DataFrame
data_summarystat = data[variables_summary].copy()

# compute table of summary statistics
summary_table = ssf.df_summary_num(data_summarystat)
# compute LATEX code for this table
summary_table.to_latex()

# Summary
summary=pd.DataFrame(columns=['Mean marginal effect'],
                     index=['ologit_Grad', 'ologit_BFGS', 'orf'])
summary.loc['ologit_Grad'] = [mme_ologit]
summary.loc['ologit_BFGS'] = [mme_ologitBFGS]
summary.loc['orf'] = [mme_orf]

print('-'*80, 'If the home team is more expensive than the away team',
      'Number of points to expect is:', ' '*80,
      summary, '-'*80, sep='\n')


#%%

# Estimating prediction performance

X_name = np.insert(covWithTime, 0, treatment)

# set seed again
np.random.seed(270421)

# Train and test split
dataTrain, dataTest = ssf.trainTestSplit(data, 0.8)

# define outcome and covariates
ytrain_np = np.array(dataTrain[outcome]).astype(float)
Xtrain_np = np.array(dataTrain[X_name]).astype(float)
ytest_np = np.array(dataTest[outcome]).astype(float)
Xtest_np = np.array(dataTest[X_name]).astype(float)

ytrain_pd = (pd.Series(dataTrain[outcome], name='HomeClasses')+1)
Xtrain_pd = dataTrain[np.insert(covWithTime, 0, treatment)].copy()
ytest_pd = (pd.Series(dataTest[outcome], name='HomeClasses')+1)
Xtest_pd = dataTest[np.insert(covWithTime, 0, treatment)].copy()

# LOGISTIC REGRESSION - Gradient Descent #
# get number of classes
ntrain = len(np.unique(ytrain_np))

# fit logistic regression
results_log = ssf.fitLogistic(Xtrain_np, ytrain_np, ntrain, display=False)
coef_log = results_log[:-2]
thresh_log = results_log[-2:]

# predict in the test sample
probs_log = ssf.logistic_probs(Xtest_np, coef_log, thresh_log)

# What is the performance?
print('-'*80, 
      'Ordered logit using Gradient Descent: The predictive performance is: ',
      ssf.pred_performance(ytest_np, probs_log, 
                           "Logistic Regression using Gradient Descent"), 
      '-'*80, sep='\n')

# LOGISTIC REGRESSION - BFGS #
# fit logistic regression
results_logBFGS = ssf.fitLogistic(Xtrain_np, ytrain_np, ntrain, optim='BFGS', 
                                  display=False)
coef_logBFGS = results_logBFGS[:-2]
thresh_logBFGS = results_logBFGS[-2:]

# predict in the test sample
probs_log = ssf.logistic_probs(Xtest_np, coef_logBFGS, thresh_logBFGS)

# What is the performance?
print('-'*80, 
      'Ordered logit using Global Optimizer: The predictive performance is: ',
      ssf.pred_performance(ytest_np, probs_log, 
                           "Logistic Regression using Global Optimizer"), 
      '-'*80, sep='\n')

# ORF #
# estimate the forest with training set
output_orf = ssf.ordered_forest(Xtrain_pd, ytrain_pd, labels)

# estimate probabilities in test set
probs_orf = ssf.predict_orf(output_orf, Xtest_pd, labels)

# What is the performance?
print('-'*80, 
      'Ordered random forest: The predictive performance is: ',
      ssf.pred_performance(ytest_np, np.array(probs_orf), "ORF"), 
      '-'*80, sep='\n')

# Computes the crosstable of the predicted class for each observation 
# in the whole sample
data_pred = pd.DataFrame(
    data={"TrueValues": data["HomeClasses"], "Logit": estimates_ologit, 
          "LogitBFGS": estimates_ologitBFGS, "ORF": estimates_ORF})

# calculate the crosstables of all three estimators. 
ssf.crosstable(data_pred, "Logit", "TrueValues")
ssf.crosstable(data_pred, "LogitBFGS", "TrueValues")
ssf.crosstable(data_pred, "ORF", "TrueValues")

#%%

# Robustness Test 1

# control for the outliers: Bayern Munich and Dortmund
dataRobust_1 = data.copy()
# Create a dummy variable that =1 if an observation belongs to an outlier
dataRobust_1['outlier'] = 0
dataRobust_1.loc[(dataRobust_1['HomeTeam'] == 'Bayern Munich')
                 | (dataRobust_1['HomeTeam'] == 'Dortmund')
                 | (dataRobust_1['AwayTeam'] == 'Bayern Munich')
                 | (dataRobust_1['AwayTeam'] == 'Dortmund'),
                 'outlier'] = 1
# Remove the outliers
dataRobust_1 = dataRobust_1.loc[dataRobust_1['outlier'] == 0]
dataRobust_1.reset_index(drop=True, inplace=True)

# Recalculate the mean marginal effects with the two estimators
# Logit Model gets computed using the global optimizer
logit_r1, orf_r1=ssf.logit_orf_marginal_eff(dataRobust_1, outcome, X_name, 
                           labels, display=False)

# print the results after outlier removal
robust_result = ssf.outlier_comp(logit_r1, orf_r1, 
                                 treatment, labels, removal=True)
# print the results before outlier removal
robust_result_bf = ssf.outlier_comp(mean_margeffectBFGS, mean_margeffect_orf,
                              treatment, labels, removal=False)

# plot both results 
ssf.oned_plot(robust_result.iloc[0,0:3]
          ,robust_result_bf.iloc[0,0:3], 'Ordered logit model')
ssf.oned_plot(robust_result.iloc[1,0:3]
          ,robust_result_bf.iloc[1,0:3], 'Ordered forest regression')
#%%

# Robustness Test 2

# control for the time component
dataRobust_2 = data.copy()

X_name = np.insert(covWithTime, 0, treatment)

# collect the names of the season dummies into a list
seasons = []
for s in np.sort(data['Season'].unique())[:-1]:
    seasons.append('Season_' + s)
covWithTime = covariates + seasons

# Create empty dataframes to save the average treatment effects(ATEs)
ATEs_logit = pd.DataFrame(columns=labels)
ATEs_orf = pd.DataFrame(columns=labels)

# Iterate through each season to find out ATE
for season in seasons:
    # used to reset the dataframe later on
    origin = dataRobust_2.copy()
    dataRobust_2 = dataRobust_2.loc[dataRobust_2[season] == 1].copy()
    dataRobust_2.reset_index(drop=True, inplace=True)
    # calculate the mean marginal effects with the two estimators
    logit_r, orf_r = ssf.logit_orf_marginal_eff(dataRobust_2, outcome,
                                            X_name, labels, display=False)
    # get only the treatment ATE
    ATEs_logit.loc[season] = logit_r
    ATEs_orf.loc[season] = orf_r.loc['HomeHigherVal'].values

    # reset the dataframe
    dataRobust_2 = origin

# reset index labels of the ATE output
season_n = list(np.sort(data['Season'].unique())); season_n.remove('2021')
ATEs_logit.index=season_n; ATEs_orf.index=season_n

# plot ATEs of ordered logit over the seasons
sns.lineplot(data=ATEs_logit); plt.xlabel('Season'); plt.ylabel('ATE')
plt.title('ATEs of ordered logit over the seasons')
plt.show()

# plot ATEs of ordered forest over the seasons
sns.lineplot(data=ATEs_orf); plt.xlabel('Season'); plt.ylabel('ATE')
plt.title('ATEs of ordered forest over the seasons')
plt.show()



# --------------------------------------------------------------------------- #
#
# END
#
# --------------------------------------------------------------------------- #
