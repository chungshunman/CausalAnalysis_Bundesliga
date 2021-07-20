"""
Microeconometrics: Self Study Functions.

Group Assignment:
    
Alec Eisenkolb, Chung-Shun Man, Nicolas Greber, Tim Hug

Spring Semester 2021.

University of St. Gallen.
"""

# import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from scipy import optimize


# --------------------------------------------------------------------------- #
#
# Recoding
#
# --------------------------------------------------------------------------- #
    
# define function to transform the outcome variables into classes
def outcome_class(string):
    """
    This function will recode the string values for the variable 'FullResult' 
    into different classes that to 2 for a Home win ('H'), 1 to a Draw ('D') 
    and 0 to an Away win ('A'). Thus, the classes are considered from the 
    perspective of the Home team for each match.

    Parameters
    ----------
    string : String.
        String value of the 'FullResult' variable, i.e. corresponding to 
        either 'H', 'D' or 'A'.

    Returns
    -------
    The function returns the corresponding points for each category of match
    outcome. 

    """
    # if match outcome is 'H', return class 2
    if string == 'H': return 2
    
    # if match outcome is 'D', return class 1
    if string == 'D': return 1
    
    # if match outcome is 'A', return class 0
    if string == 'A': return 0


# computes the difference of two variables
def var_diff(data, names, var_first, var_second):
    '''
    Iterate over the variables to compute the differences

    Parameters
    ----------
    data : TYPE: pd.DataFrame
        DESCRIPTION. Data on which the covariates are based
    names : TYPE: list of strings
        DESCRIPTION. Names of the subjects i.e. Home and Away
    var_first : TYPE: list of strings
        DESCRIPTION. The variable names that appear before the subject name
    var_second : TYPE: list of strings
        DESCRIPTION. The variable names that appear after the subject name

    Returns
    -------
    data : TYPE: pd.DataFrame
        DESCRIPTION.

    '''
    # if the variable name appears before the subject name i.e. Home
    for var in var_first:
        code = """data['Diff{}']=data.apply(lambda row: row['{}{}'] - 
        row['{}{}'], axis=1)"""
        exec(code.format(var, names[0], var, names[1], var))
        
    # if the variable name appears after the subject name i.e. Home
    for var in var_second:
        code = """data['Diff{}']=data.apply(lambda row: row['{}{}'] - 
        row['{}{}'], axis=1)"""
        exec(code.format(var, var, names[0], var, names[1]))
    return data


# function that encodes a categorical variable into dummy variables
def oneHotEncoding(data, var):
    """
    Encodes a categorical variable into dummy vectors

    Parameters
    ----------
    data : TYPE: pd.DataFrame
        DESCRIPTION. corresponds to the underlying data
    var : TYPE: string
        DESCRIPTION. variable name of a categorical thath should be encoded 
        into dummies

    Returns
    -------
    data : TYPE: pd.DataFrame
        DESCRIPTION. dataframe including the corresponding dummy vectors.

    """
    # get categories of cateogrical variable
    uniqueVal = np.sort(data[var].unique())
    # iterate over categories
    for cat in uniqueVal:
        # define label
        label = var + '_' + cat
        # initialize new column consisting of zeros
        data.loc[:, label] = 0
        # encode the respective values to one
        data.loc[data[var] == cat, label] = 1
    # return data frame
    return data



# --------------------------------------------------------------------------- #
#
# Data Analysis
#
# --------------------------------------------------------------------------- #

# compute a table of the time trends of market values
def time_name(data, season_name, team_name, mean_name, missing_allowed=5):
    '''
    This function calculates the mean value of every team in every season. 
    It generates a table with team and season dimension,

    Parameters
    ----------
    data : pd.DataFrame
        DESCRIPTION. The dataframe
    season_name : string
        DESCRIPTION. The time dimension of the data
    team_name : string
        DESCRIPTION. The cross-sectional dimension of the data
    mean_name : string
        DESCRIPTION. The mean values that are of interest
    missing_allowed : number, optional
        DESCRIPTION. The default is 5. It is the number of missing mean values 
        allowed in the function.

    Returns
    -------
    df_show

    '''
    # Time: remove the underline
    data[season_name] = data[season_name].apply(lambda x: x.replace("_", ""))
    # Time: sort the seasons in ascending order
    data = data.sort_values(by=season_name)
    
    # create lists of names and time
    teams = list(data[team_name].unique())
    seasons = list(data[season_name].unique())
    
    # Get the means of every team in every time period
    means = []
    for team in teams:
        mean_v = []
        for period in seasons:
            mean_v.append(data.loc[(data[team_name]==team) & 
                                   (data[season_name]==period)]
                          [mean_name].mean())
        means.append(mean_v)
      
    # Create a dataframe with cross-sectional and time dimensions
    df = pd.DataFrame(means, index=teams, columns= seasons).T
    
    # Show the subset of dataset with few NaN values
    df_show = pd.DataFrame()
    for col in df.columns:
        if sum(df[col].isna()) <= missing_allowed:
            df_show[col] = df[col]
     
    # Use seaborn to show the time on x axis and mean values on the y axis.
    plt.figure(figsize=(12,5))
    sns.set_style("darkgrid")
    ax=sns.lineplot(data=df_show, legend=True, dashes=False)
    ax.set(xlabel=season_name, ylabel= mean_name) 
    plt.legend(bbox_to_anchor=(0, -0.35, 1, 1), loc=8, 
               ncol=5, mode="expand", borderaxespad=0, fontsize=6)
    # alternative placement of the legend:
    # plt.legend(bbox_to_anchor=(1.05, 0, 1, 1), loc=2, 
    #            ncol=2, fontsize=8)
    plt.title("Evolution of Squad Market Value")
    # Return the DataFrame
    return df


# function that generates a cross table of two variables
def crosstable(dataset, var1, var2):
    """
    crosstable for two variables.
    parameters:
    ----------
    dataset: DataFrame pandas
    var1: string of variable name
    var2: string of variable name
    
    Returns:
    ----------
    Crosstable
    """
    
    # get all possible values of both variables, order them
    var1_set = sorted(list(set(dataset[var1])))
    var2_set = sorted(list(set(dataset[var2])))
    # predefine dataframe for result
    crosstab = pd.DataFrame(index = var1_set, columns= var2_set)
    # name the two axis
    crosstab = crosstab.rename_axis(str(var1), axis=0)
    crosstab = crosstab.rename_axis(str(var2), axis=1)
    # i and j are used to fill in row and column. 
    # for all values of this variable do:
    for el1 in var1_set:
        # take the subset
        subset1 = dataset[dataset[var1]== el1]
        # for all values of the other variable
        for el2 in var2_set:
            # again take subset of those
            subset2 = subset1[subset1[var2] == el2]
            # how many observations are there?
            obs = len(subset2)
            # fill in predefined (results) dataframe
            crosstab.loc[el1,el2] = obs
    # return the resulting dataframe / crosstable
    return crosstab


# function to compute the share of observations for a binary variable
def binary_share(df, variables):
    """
    Function that computes a table of statistics of chosen binary variable,
    as defined by the 'variables' input. 

    Parameters
    ----------
    df : DataFrame
        DataFrame of the variables in consideration for the computation of 
        statistics.
    
    variables : String
        A string input defining the name of the binary variable
        of which the function shall compute the shares (mean) and number of
        observations.

    Returns
    -------
    This function prints a table of statistics (shares and observations)
    relative to the defined binary variable in the DataFrame.

    """
    # Create a sorted list of all unique values for the variable
    binary = sorted(list(df[variables].unique()))
    # Create an empty DataFrame of the given dimensions
    table = pd.DataFrame(index=range(len(binary)), 
                         columns=(variables, ) + ('Share', ) + ('Obs', ))
    # Set the index = 0 
    idx = 0
    # Start a for-loop over all values of the binary variable
    for val in binary: 
        # In first column, insert value of binary variable
        table.loc[idx, variables] = val
        # In second column, insert the share of binary = val
        table.loc[idx, 'Share'] = (len(df.loc[(df[variables] == val), :]) / 
                                   len(df.loc[df[variables], :]))
        # In third column, insert the number of observations for val
        table.loc[idx, 'Obs'] = len(df.loc[(df[variables] == val), :])
        # Add one to the index to iterate over the next value for binary var.
        idx = idx + 1
    # Set index of DataFrame to the values of binary var.
    table = table.set_index(variables)
    # Print the resulting Table
    print('Share of Sample w.r.t. ' + str(variables) + ':', '-' * 45,
          table, '-' * 45, '\n\n', sep='\n')
    # return DataFrame
    return table
   
     
# function for the mean of a variable over categories over another variable
def mean_by(df, main_var, byvar):
    """
    Function which will compute the mean of the main variable (main_var) with 
    respect to the grouping variable (byvar). 

    Parameters
    ----------
    df : DataFrame
        DataFrame which contains all the relevant variables and observations.
    main_var : String
        Main Variable, of which means shall be computed.
    byvar : String
        Grouping variable, which will be used to sub-sample the main variable.

    Returns
    -------
    This function will compute a table of means and observations of the main
    variable over the various groups of the grouping variable.

    """
    # Compute the unique values of the grouping variable, 'byvar'
    unique_values = sorted(list(df[byvar].unique()))
    # Create an empty DataFrame of given dimensions
    table = pd.DataFrame(index=range(len(unique_values)),
                         columns=(byvar, ) + (main_var, ) + ('Obs', ))
    # Setting the index equal to zero
    idx = 0
    # Initiating a for-loop for all unique values of the grouping variable
    for value in unique_values:
        # Placing the grouping variable value in the first column
        table.loc[idx, byvar] = value 
        # Placing the mean of the main variable, w.r.t. the grouping variable
        # into the second column
        table.loc[idx, main_var] = round(np.mean(df.loc[(df[byvar] == value), 
                                                         main_var]), 3)
        # Placing the number of observations into the third column
        table.loc[idx, 'Obs'] = len(df.loc[(df[byvar] == value), :])
        # adding one value to the index
        idx = idx + 1
    # Setting the grouping variable column as the index of the DataFrame
    table = table.set_index(byvar)
    # Printing the DataFrame of mean values
    print('Mean Values of ' + str(main_var) + ' with respect to ' + str(byvar) 
          + ':', '-' * 50, table, '-' * 50, '\n\n', sep='\n')
    # Return DataFrame
    return table
    

# creates a table with summary statistics of the numerical variables
def df_summary_num(dataframe):
    """
    Returns summary including mean, variance, standard deviation,
    maximum and minimum value of each variable. Additionally, the number 
    of missing and unique values, number of observations and the 
    variable names.

    Parameters
    ----------
    - dataframe : pandas dataframe

    Returns
    -------
    mean, var, std, max, min, num. of missing values, num. of unique obs.
    for all variabes in the dataframe.

    """   
    #  only pick variables with floats and integers
    dataframe.select_dtypes(include=['float64', 'int64'])
    # calculate the 
    mean = dataframe.mean()
    var = dataframe.var()
    std = dataframe.std()
    maximum = dataframe.max()
    minimum = dataframe.min()
    null = dataframe.isnull().sum()
    unique = dataframe.nunique()
    nobs = dataframe.count()
    # merges the whole list in a dataframe
    result = pd.concat([mean, var, std, maximum, minimum, null, unique, nobs],
                       axis=1, join = 'inner')
    # give names to columns
    COLUMN_NAMES = ['MEAN', 'VAR', 'STD', 'MAX', 'MIN', 'MISSING', 'UNIQUE', 
                    'COUNT']
    result.columns = COLUMN_NAMES
    #return the table, s.t. everything is shown
    with pd.option_context('display.max_rows', None, 'display.max_columns', 
                           None):
        print(round(result,1))
    # Return the DataFrame
    return round(result, 1)


# creates a table with summary statistics of the string variables
def df_summary_char(dataframe):
    """
    
    Parameters
    ----------
    - dataframe : pandas DataFrame

    Returns
    -------
    Overview 

    """
    # only pick variables other than the numeric ones
    variables = list(dataframe.select_dtypes(exclude=['int64', 'float'])
                     .columns)
    # for all the variables in this set
    for var in variables:
        # how many unique observations are there?
        number = len(set(dataframe[var]))
        # if more than 40, print the number of different observations
        if number > 40:
            print("The variable {} has {} different objects."
                  .format(var, number), '\n\n')
        # else print all the unique values incl. number of observations
        else:
            result = dataframe[var].value_counts()
            result.columns = ["Observations", "Counts"]
            print(var , '-' * 30, result, '-' * 30, '\n\n', sep='\n')


# perform balancing checks considering home and away team
def home_away_check(data, season_name, home_name, away_name):
    '''
    It checks whether the numbers of home games and away games of every team
    are the same over the season

    Parameters
    ----------
    data : TYPE pd.DataFrame
        DESCRIPTION. The dataset on which the variables are based
    season_name : TYPE string
        DESCRIPTION. The seasons in which the home and away teams play
    home_name : TYPE string
        DESCRIPTION. The name of the home team
    away_name : TYPE string
        DESCRIPTION. The name of the away team

    Returns
    -------
    total_df : TYPE pd.DataFrame
        DESCRIPTION. The number of home games on the left and away games on
        the right in each cell of the dataframe

    '''
    # Find out the unique taem names
    teams=list(data[home_name].unique())
    # Find out the unique season names in ascending order
    seasons=np.sort(list(data[season_name].unique()))
    
    total_df=[] # This stores the total number of games
    diff_df=[] # This stores the difference in the number of games
    for period in seasons:
        total=[] # This stores the total number of games every season
        diff=[] # This stores the difference in number of games every season
        for team in teams:
            # The number of home games for each team in each season
            right=len(data.loc[(data[home_name]==team) 
                               & (data[season_name]==period)])
            # The number of away games for each team in each season
            left=len(data.loc[(data[away_name]==team) 
                              & (data[season_name]==period)])
            diff.append(right-left)
            total.append([right, left])
        diff_df.append(diff)
        total_df.append(total)
    
    # Turn them into dataframes for better illusttration
    diff_df=pd.DataFrame(diff_df, index=seasons, columns=teams)
    total_df=pd.DataFrame(total_df, index=seasons, columns=teams)
    
    # Output the  result
    print('-'*80, 'Number of Home matches and number of Away matches',
          '-'*80, total_df,
          '-'*80, 'Number of Home matches minus number of Away matches',
          '-'*80, diff_df,
          '-'*80, sep='\n')
    return total_df, diff_df



# --------------------------------------------------------------------------- #
#
# Plotting
#
# --------------------------------------------------------------------------- #

# function to create barplots
def barplot(data, varName, label, path, display=True):
    """
    A function that generates a barplot and a correponing probability table.

    Parameters
    ----------
    data : TYPE: pd.DataFrame
        DESCRIPTION. corresponds to the underlying data
    varName : TYPE: string
        DESCRIPTION. corresponds to the name of the variable that should be 
        plotted
    label : TYPE: np.array
        DESCRIPTION. corresponds to the label of unique values
    path : TYPE: string
        DESCRIPTION. describes the path where the plot should be stored
    display : TYPE, optional: boolean
        DESCRIPTION. The default is True. toggles if probability table should 
        be displayed.

    Returns
    -------
    table : TYPE: pd.DataFrame
        DESCRIPTION. probability table corresponding to the barplot

    """
    # get number of unique values in the variable vector
    uniqueVal = np.sort(data[varName].unique())
    # get counts
    bars = data[varName].value_counts().sort_index()
    # define positions
    y_pos = np.arange(len(uniqueVal))
    # create bar plot
    plt.bar(y_pos, bars, color='grey')
    plt.xticks(y_pos, label)
    plt.title('Barplot of ' + varName)
    #compute probabilities
    table = pd.DataFrame(data=None, index=bars.index, 
                         columns=['Frequency', 'Probability'])
    table.iloc[:, 0] = bars
    for i in range(len(bars)):
        table.iloc[i, 1] = bars.iloc[i] / bars.sum()
    # check if table should be displayed
    if display:
        print(round(table, 2))
    # save plot
    plt.savefig(path + 'histogram_of_' + varName + '.png')
    # show plot
    plt.show()
    # return table anyways
    return table


# function to plot histograms
def histogram(data, var_name, bins=10):
    '''
    It generates a histogram with a pre-determined setting.

    Parameters
    ----------
    data : TYPE: pd.DataFrame
        DESCRIPTION. It is the dataset on which the variable is based
    var_name : TYPE: string
        DESCRIPTION. It is the name of the variable of interest
    bins : TYPE, optional
        DESCRIPTION. The default is 10.

    Returns
    -------
    None.

    '''
    # It generates a histogram with a pre-determined setting.
    plt.hist(data[[var_name]], bins=11, rwidth=0.9, color='grey')
    plt.xlabel(var_name)
    plt.ylabel('Count')
    plt.show()
    


# function to plot histograms while overlaying two variables
def dual_histogram(data, var_names, names, first_name=True, bins=10):
    '''
    It generates two histograms in one graph. They should share the same 
    scales and are thus comparable

    Parameters
    ----------
    data : TYPE: pd.DataFrame
        DESCRIPTION. The dataset on which the variables are based.
    var_names : TYPE: string
        DESCRIPTION. The data we are interested in.
    names : TYPE: string
        DESCRIPTION. It usually contains only two strings. i.e. 'Home', 'Away'
    first_name : TYPE, optional 
        DESCRIPTION. The default is True. the value is before the name 
    bins : TYPE, optional
        DESCRIPTION. The default is 10.

    Returns
    -------
    None.

    '''
    # if the value is put before the name of either home or away.
    if first_name:
        for i in var_names:
            n_bins=[]
            for j in names:
                string=i+j
                maxi=max(data[string])
                mini=min(data[string])
                n_bins.append(np.linspace(mini, maxi, bins))
                plt.hist(data[string], n_bins[0], alpha=0.5, label=str(string))
            plt.legend(loc='upper right')
            plt.xlabel(i)
            plt.ylabel('Count')
            plt.show()
    # if the value is put after the name of either home or away.
    else:
        for i in var_names:
            n_bins=[]
            for j in names:
                string=j+i
                maxi=max(data[string])
                mini=min(data[string])
                n_bins.append(np.linspace(mini, maxi, bins))
                plt.hist(data[string], n_bins[0], alpha=0.5, label=str(string))
            plt.legend(loc='upper right')
            plt.xlabel(i)
            plt.ylabel('Count')
            plt.show()


            
# --------------------------------------------------------------------------- #
#
# Logistic Regression
# Inspired by the mord package
#
# --------------------------------------------------------------------------- #


# Define logistic cumulative distribution function
def cdf(z):
    """
    Computes the probabilites of a logistic distribution

    Parameters
    ----------
    z : TYPE: np.array
        DESCRIPTION. array with bounds to compute probabilities

    Returns
    -------
    cdf : TYPE: np.array
        DESCRIPTION. array with probabilities

    """
    # check for stability
    idx = z > 0
    # initialize empty output vector
    cdf = np.zeros_like(z)
    # compute probabilities
    cdf[idx] = 1 / (1 + np.exp(-z[idx]))
    cdf[~idx] = np.exp(z[~idx]) / (1 + np.exp(z[~idx]))
    # return cdf
    return cdf

# Define Logistic Loss
def logLoss(z):
    """
    Computes the logistic loss

    Parameters
    ----------
    z : TYPE: np.array
        DESCRIPTION. array with values from which the logistic loss should be
        computed

    Returns
    -------
    logLoss: TYPE: np.array
        DESCRIPTION. array consisting of logistic losses

    """
    # check for stability
    idx = z > 0
    # initialize empty output vector
    logLoss = np.zeros_like(z)
    # compute logLoss
    logLoss[idx] = np.log(1 + np.exp(-z[idx]))
    logLoss[~idx] = (-z[~idx] + np.log(1 + np.exp(z[~idx])))
    # return logLoss
    return logLoss


# Define objective function
def objMargin(w, X, y, n):
    """
    Objective function which should be minimised.

    Parameters
    ----------
    w : TYPE: np.array
        DESCRIPTION. weight vector including values for the thresholds
    X : TYPE: np.array
        DESCRIPTION. array of observations
    y : TYPE: np.array
        DESCRIPTION. column vector consisting of observed outcomes
    n : TYPE: int
        DESCRIPTION. corresponds to the number of classes

    Returns
    -------
    TYPE: float
        DESCRIPTION. sum of the errors

    """
    # define computation matrices
    L = np.zeros((n - 1, n - 1))
    L[np.tril_indices(n - 1)] = 1.
    S = np.sign(np.arange(n - 1)[:, None] - y + 0.5)
    # extract coefficients from w-vector
    coef = w[:X.shape[1]]
    a = w[X.shape[1]:]
    # compute thresholds
    alpha = np.dot(L, a)
    # compute error
    error = logLoss(S * (alpha[:, None] - np.dot(X, coef)))
    # return sum of errors
    return np.sum(error)


# Define gradient function
def gradMargin(w, X, y, n):
    """
    Gradient of the objective function

    Parameters
    ----------
    w : TYPE: np.array
        DESCRIPTION. weight vector including values for the thresholds
    X : TYPE: np.array
        DESCRIPTION. array of observations
    y : TYPE: np.array
        DESCRIPTION. column vector consisting of observed outcomes
    n : TYPE: int
        DESCRIPTION. corresponds to the number of classes

    Returns
    -------
    grad_coef : TYPE: np.array
        DESCRIPTION. array with gradient values regarding coefficients
    grad_alpha : TYPE: np.array
        DESCRIPTION. array with gradient values regarding thresholds

    """
    # define computation matrices
    L = np.zeros((n - 1, n - 1))
    L[np.tril_indices(n - 1)] = 1.
    S = np.sign(np.arange(n - 1)[:, None] - y + 0.5)
    # extract coefficients from w-vector
    coef = w[:X.shape[1]]
    a = w[X.shape[1]:]
    # compute thresholds
    alpha = np.dot(L, a)
    # compute probabilities
    sigma = S * cdf(-S * (alpha[:, None] - np.dot(X, coef)))
    # compute gradients
    grad_coef = np.dot(X.T, sigma.sum(0))
    grad_a = -sigma.sum(1)
    grad_alpha = np.dot(L.T, grad_a)
    # return gradients for coefficients and thresholds
    return np.concatenate((grad_coef, grad_alpha), axis=0)


# Fit the logistic regression using gradient descent
def fitLogistic(X, y, n, optim='grad', lR=1e-8, numIter=10_000, 
                display=True):
    """
    Fits a logistic regression using a gradient descent algorithm

    Parameters
    ----------
    X : TYPE: np.array
        DESCRIPTION. array of observations
    y : TYPE: np.array
        DESCRIPTION. column vector consisting of observed outcomes
    n : TYPE: int
        DESCRIPTION. corresponds to the number of classes
    optim : TYPE, optional: string
        DESCRIPTION: The default is grad. Toggles the optimiziation algorithm.
        Available is 'grad' or 'L-BFGS-B'
    lR : TYPE, optional: float
        DESCRIPTION. The default is 10 ** (-8). Corresponds to the learning 
        rate
    numIter : TYPE, optional: int
        DESCRIPTION. The default is 10_000. Corresponds to the number of 
        iterations.
    display : TYPE, optional: boolean
        DESCRIPTION. The default is True. Toggles if coefficients should be 
        displayed.

    Returns
    -------
    w : TYPE: np.array
        DESCRIPTION. corresponds to an array consisting of optimal coefficients
        and thresholds.

    """
    # define computation matrices
    L = np.zeros((n - 1, n - 1))
    L[np.tril_indices(n - 1)] = 1.
    # initialize weights and thresholds
    w = np.zeros(X.shape[1] + (n - 1)) + .5
    w[X.shape[1]:] = np.arange(n - 1)
    # check for optim algoritthm
    if optim == 'grad':
        # iterate numIter times
        for i in range(numIter):
            # compute gradient
            g = gradMargin(w, X, y, n)
            # update alphas & betas
            w[:X.shape[1]] -= lR * g[:X.shape[1]]
            w[X.shape[1]:] -= lR * g[X.shape[1]:]
    elif optim == 'L-BFGS-B':
        # run scipy optimizer
        result = optimize.minimize(objMargin, w, method='L-BFGS-B', 
                                   jac=gradMargin,
                                   options={'maxiter' : numIter},
                                   args=(X, y, n))
        # assign result
        w = result.x
    # compute final thresholds
    w[X.shape[1]:] = np.dot(L, w[X.shape[1]:])
    # check if results should be displayed
    if display:
        print(w)
    return w


# --------------------------------------------------------------------------- #
#
# Marginal Effects - Logistic Regression
# inspired by PC session 3 code, Microeconometrics 2021
#
# --------------------------------------------------------------------------- #

# computes the probabilites using the logit model
def logistic_probs(X, betas, threshold):
    """
    calculates the probabilities of each class given estimated coefficients, 
    covariates and estimated thresholds

    Parameters
    ----------
    X : np.ndarray
        exogenous variables, first one is the one of interest
    betas : np.ndarray
        coefficients  from the logit estimation
    threshold : np.ndarray
        thresholds form the logit estimation

    Returns
    -------
    estimated probabilities for each observation for each , given outputs 
    from logistic regression

    """
    # how many observations
    num_y = len(X)
    # how many categories
    number_categories = len(threshold) + 1
    # create container for results
    probability = np.empty([num_y, number_categories])
    # now for all outcome categories calculate the probs
    for category in range(number_categories):
        # first category (special)
        if category == 0:
            probability[:, category] = (
                1 / (1 + np.exp(np.dot(X, betas) - threshold[category])))
        # for all inbetween
        elif category in range(1, number_categories-1):
            probability[:, category] = (1 / (1 + np.exp(
                np.dot(X, betas) - threshold[category]))) - (
                1 / (1 + np.exp(np.dot(X, betas) - threshold[category-1])))
        # last category (special)
        else:
            probability[:, category] = 1 - \
                (1 / (1 + np.exp(np.dot(X, betas) - threshold[category-1])))
    return(probability)


# compute mean marginal effects
def mean_margeff(X, betas, threshold):
    """
    calculates the mean marginal effect after logistic regression

    Parameters
    ----------
    X : np.ndarray
        exogenous variables, first one is the one of interest
    betas : np.ndarray
        coefficients  from the logit estimation
    threshold : np.ndarray
        thresholds form the logit estimation

    Returns
    -------
    estimated mean marginal effect on the first variable in X.

    """
    # how many categories in outcome variable
    number_categories = len(threshold) + 1
    # how many observations
    num_y = len(X)
    # containter for the probabilities, 3 dim
    # (number observations, number cat, 2)
    probability = np.empty([num_y, number_categories, 2])
    # do for treatment 0,1
    for treat in [0, 1]:
        # set first entry (treament) = 0,1
        treat_v = np.repeat(treat, num_y)
        covariates = np.c_[treat_v, X[:,1:]]
        # now for all outcome categories calculate the probs given treat fix,
        # fill into container
        for category in range(number_categories):
            # first category (special)
            if category == 0:
                probability[:, category, treat] = (
                    1 / (1 + np.exp(np.dot(covariates, betas) - threshold[category])))
            # for all inbetween
            elif category in range(1, number_categories-1):
                probability[:, category, treat] = (1 / (1 + np.exp(
                    np.dot(covariates, betas) - threshold[category]))) - (
                    1 / (1 + np.exp(np.dot(covariates, betas) - threshold[category-1])))
            # last category (special)
            else:
                probability[:, category, treat] = 1 - \
                    (1 /
                     (1 + np.exp(np.dot(covariates, betas) - threshold[category-1])))
    # calculate the differences in probabilities between treat and not treat
    eff = probability[:, :, 1] - probability[:, :, 0]
    # mean marginal effect (take mean over all observations)
    eff = eff.mean(0)
    return(eff)


# --------------------------------------------------------------------------- #
#
# Ordered Forest
#
# --------------------------------------------------------------------------- #

# trains an ordered forest
def ordered_forest(exog, outcome, labels, n_tree=1000, min_n=5, s_cov=0.3):
    '''
    It calculates the probability of each class of outcome based on random
    forest prediction. All the orf functions are inspired by Lechner, M, 
    and Okasa, G. (2020) Ordered forest, Random Forest Estimation of the 
    Ordered Choice Model.


    Parameters
    ----------
    exog : TYPE: pd.DataFrame
        DESCRIPTION. exogeneous variables.
    outcome : TYPE: pd.Series
        DESCRIPTION. outcome variable.
    labels : TYPE: a list of strings
        DESCRIPTION. the labels of classes of the outcome variable.
    n_tree : TYPE, optional
        DESCRIPTION. The default is 1000. The number of trees in the forest.
    min_n : TYPE, optional
        DESCRIPTION. The default is 5. The minimum leaf size in the forest.
    s_cov : TYPE, optional
        DESCRIPTION. The default is 0.3. Share of covariates in the forest.

    Returns
    -------
    output : TYPE: dictionary
        DESCRIPTION. it includes both ordered random forest models and 
        class probabilites.

    '''    
    # get the number of classes
    n_classes = (outcome.unique()).max()
    
    # turn the outcome variable into multiple dummy variables
    outcome_dum=pd.get_dummies(outcome)
    forests=[] # save the random forest models
    probs=[] # save the probabilities
    
    # loop through classes from 1 to number of classes -1
    for class_index in range(1, (n_classes)):
        # get the binary outcome 
        outcome_orf=outcome_dum.loc[:,1:class_index].sum(axis=1)
        # build a binary random forest model
        orf_model = RandomForestRegressor(n_estimators=n_tree,
                                          min_samples_leaf=min_n,
                                          max_features=s_cov,
                                          oob_score=True)
        # fit the outcome variable and covariates into the model
        orf_model.fit(X=exog, y=outcome_orf)
        prob=orf_model.oob_prediction_
        
        forests.append(orf_model) # save the forest
        probs.append(prob) # save the probability
    
    # turn the forests into a dictionary for easy extraction
    forests=dict(zip(np.linspace(1, n_classes-1, n_classes-1), forests))
    
    # copy the probabilites and insert an array of 1 into the last place
    probs_1= probs.copy()
    probs_1.insert((n_classes-1), np.array([1]*exog.shape[0]))
    
    # copy the probabilites and insert an array of 0 into the first place
    probs_0= probs.copy()
    probs_0.insert(0, np.array([0]*exog.shape[0]))
    
    # get the difference between the probabilities
    class_prob = np.array(probs_1) - np.array(probs_0)
    class_prob = pd.DataFrame(class_prob).T
    
    # turn negative probability into zero
    class_prob[class_prob < 0]=0
    for i in range(0, n_classes):
        # normalize the probabilities of each row so they sum up to one
        class_prob.iloc[:,i] = class_prob.iloc[:,i]/class_prob.sum(axis=1)
    class_prob.columns = labels
    
    # the output combines both forest models and class probabilities
    output={'forests': forests, 'probs': class_prob}
    
    return output


# predicts the classes using an ordered forest
def predict_orf(forests, exog, labels):
    '''
    It predicts the probability of each class for the out-of-sample
    observations.

    Parameters
    ----------
    forests : TYPE: dictionary
        DESCRIPTION. The forest models that are used to predict outcome.
        The number of forest is the number of outcome classes minus 1
    exog : TYPE: pd.DataFrame
        DESCRIPTION. A matrix of exogenous variables.
    labels : TYPE: a list of strings
        DESCRIPTION. The labels of the classes of the outcome variable

    Returns
    -------
    class_prob : TYPE pd.DataFrame
        DESCRIPTION.

    '''
    # get the number of classes
    n_classes=len(labels)
    # get the forest models from the dictionary
    forests= forests['forests']
    # create a list to save the probabilities 
    probs=[]
    
    for index in range(1, n_classes):
        # save the probability
        probs.append(forests[index].predict(exog))
    
    # copy the probabilites and insert an array of 1 into the last place
    probs_1= probs.copy()
    probs_1.insert((n_classes-1), np.array([1]*exog.shape[0]))
    
    # copy the probabilites and insert an array of 0 into the first place
    probs_0= probs.copy()
    probs_0.insert(0, np.array([0]*exog.shape[0]))
    
    # get the difference between the probabilities
    class_prob = np.array(probs_1) - np.array(probs_0)
    class_prob = pd.DataFrame(class_prob).T
    
    # turn negative probability into zero
    class_prob[class_prob<0]=0
    for i in range(0, n_classes):
        # normalize the probabilities of each row so they sum up to one
        class_prob.iloc[:,i] = class_prob.iloc[:,i]/class_prob.sum(axis=1)
    class_prob.columns = labels
    
    # return the class probabilities
    return class_prob



# --------------------------------------------------------------------------- #
#
# Marginal Effects - Ordered Forest
#
# --------------------------------------------------------------------------- #

# compute marginal effects for the ordered forest
def marginal_eff(forests, exog, labels, mean_mf=True, display=True):
    '''
    It calculates either the mean marginal effects of covariates on the
    outcome or the marginal effects at the covariate means on the outcome. 

    Parameters
    ----------
    forests : TYPE: dictionary
        DESCRIPTION. It has both ordered random forest models and class
        probabilities.
    exog : TYPE: pd.DataFrame
        DESCRIPTION. It contains the observed values of all covariates.
    labels : TYPE: list of strings
        DESCRIPTION. They are the labels of the outcome classes
    mean_mf : TYPE, optional
        DESCRIPTION. True means the mean marginal effects. 
        False means the marginal effects at the means.
        The default is True.

    Returns
    -------
    marginal_eff : TYPE: pd.DataFrame
        DESCRIPTION. It is the marginal effects of the covariates on 
        the outcome

    '''
    # if mean marginal effects are chosen
    if mean_mf:
        marginal_eff = pd.DataFrame(columns=labels)
        for name in exog.columns:
            # select the dummy varaible
            if exog[name].nunique() == 2:
                # save the original observed values of the dummy variable
                dummy = exog[name].copy()
                # turn all values into one and predict the outcome
                exog[name] = 1
                prob_1 = predict_orf(forests, exog, labels)
                
                # turn all values into zero and predict the outcome
                exog[name] = 0
                prob_0 = predict_orf(forests, exog, labels)
                
                # calculate the mean difference of the probabilities
                marginal_eff.loc[name,:] = (prob_1.values - 
                                            prob_0.values).mean(axis=0)
                
                # reset the dummy variable
                exog[name] = dummy
                
            # select any non-dummy varaible, including discrete or continuous 
            else:
                # weight given to the standard deviation
                h_std = 0.1
                
                # save the original values of the varaible
                original = exog[name].copy()
                x_max = original.max() # the maximum value of the variable
                x_min = original.min() # the minimum value of the variable
                x_std = original.std() # the standard deviation of the variable
                
                # shift up the values by the weighted standard deviation
                x_up = original + (x_std * h_std)
                for i in range(0, len(x_up)):
                    # keep the new maximum value within the old maximum bound
                    if x_up[i] >= x_max:
                        x_up[i] = x_max 
                    # increase the values below the old minimum bound
                    elif x_up[i] <= x_min:
                        x_up[i] = x_min + (x_std * h_std)
                
                exog[name] = x_up.copy()
                # calculate upper probabilites with an increased covariate
                prob_1 = predict_orf(forests, exog, labels)
                
                # shift down the values by the weighted standard deviation
                x_down = original - (x_std * h_std)
                for i in range(0, len(x_down)):
                    # keep the new minimum value within the old minimum bound
                    if x_down[i] <= x_min:
                        x_down[i] = x_min
                    # decrease the values above the old maximum bound
                    elif x_down[i] >= x_max:
                        x_down[i] = x_max - (x_std * h_std)
                        
                exog[name] = x_down.copy()
                # calculate lower probabilites with an decreased covariate
                prob_0 = predict_orf(forests, exog, labels)
                
                # compute the marginal effects by the diffrerence between
                # the upper and lower probabilities
                marginal_eff.loc[name,: ] = (prob_1.values - 
                                             prob_0.values).mean(axis=0)
                # reset the covariate
                exog[name] = original.copy()
        if display:
            print('-'*80, 'Ordered forest regression:', ' '*80,
                  'The mean marginal effects of the covariates are', 
                  marginal_eff, '-'*80, sep='\n')
        
    # if marginal effects at means are chosen    
    else:
        marginal_eff = pd.DataFrame(columns=labels)
        for name in exog.columns:
            # select the dummy varaible
            if exog[name].nunique() == 2:
                # save the original observed values of the dummy variable
                dummy = exog[name].copy()
                
                # turn all values into one and predict the outcome
                exog[name] = 1
                # compute the means of the covariates
                exog_mean = pd.DataFrame(exog.mean(axis=0)).T
                prob_1 = predict_orf(forests, exog_mean, labels)
                
                 # turn all values into zero and predict the outcome
                exog[name] = 0
                exog_mean = pd.DataFrame(exog.mean(axis=0)).T
                # compute the means of the covariates
                prob_0 = predict_orf(forests, exog_mean, labels)
                
                # compute the marginal effects at the means
                marginal_eff.loc[name,:] = (prob_1.values - prob_0.values)
                # reset the dummy variable
                exog[name] = dummy
                
            # select any non-dummy varaible, including discrete or continuous 
            else:
                # weight given to the standard deviation
                h_std = 0.1
                # save the original values of the varaible
                original = exog[name].copy()
                x_max = original.max() # the maximum value of the variable
                x_min = original.min() # the minimum value of the variable
                x_std = original.std() # the standard deviation of the variable
                
                # shift up the values by the weighted standard deviation
                x_up = original + (x_std * h_std)
                for i in range(0, len(x_up)):
                    # keep the new maximum value within the old maximum bound
                    if x_up[i] >= x_max:
                        x_up[i] = x_max
                    # increase the values below the old minimum bound
                    elif x_up[i] <= x_min:
                        x_up[i] = x_min + (x_std * h_std)
                
                exog[name] = x_up.copy()
                # get the means of the covariates
                exog_mean = pd.DataFrame(exog.mean(axis=0)).T 
                # calculate upper probabilites with an increased covariate
                prob_1 = predict_orf(forests, exog_mean, labels)
                
                # shift down the values by the weighted standard deviation
                x_down = original - (x_std * h_std)
                for i in range(0, len(x_down)):
                    # keep the new minimum value within the old minimum bound
                    if x_down[i] <= x_min:
                        x_down[i] = x_min
                    # decrease the values above the old maximum bound
                    elif x_down[i] >= x_max:
                        x_down[i] = x_max - (x_std * h_std)
                        
                exog[name] = x_down.copy()
                # get the means of the covariates
                exog_mean = pd.DataFrame(exog.mean(axis=0)).T
                # calculate upper probabilites with an increased covariate
                prob_0 = predict_orf(forests, exog_mean, labels)
                
                # compute the marginal effects at the covariate means
                marginal_eff.loc[name,: ] = prob_1.values - prob_0.values
                
                # reset the values of the covariates
                exog[name] = original.copy()
        if display:
            print('-'*80, 'Ordered forest regression:', ' '*80,
                  'The marginal effects at the covariate means are',
                  marginal_eff, '-'*80, sep='\n')
    return marginal_eff


# --------------------------------------------------------------------------- #
#
# Train-Test-Split
#
# --------------------------------------------------------------------------- #

# function that splits the dataset into a training and test set
# splits the data according to the season
def trainTestSplit(data, share):
    """
    performs a split into train and test sample. the split is performed such
    that the share of observations for each season is kept, so it can be taken
    care of time effects.

    Parameters
    ----------
    data : TYPE: pd.DataFrame
        DESCRIPTION. corresponds to underlying data
    share : TYPE: float
        DESCRIPTION. corresponds to the share for the training sample

    Returns
    -------
    df_train : TYPE: pd.DataFrame
        DESCRIPTION. training sample
    df_test : TYPE: pd.DataFrame
        DESCRIPTION. test sample

    """
    # initialize empty dataframes
    df_train = pd.DataFrame(columns=data.columns)
    df_test = pd.DataFrame(columns=data.columns)
    # iterate over different seasons
    for s in data['Season'].unique():
        # subset data
        aux_df = data.loc[data.Season == s, :]
        # get number of observations
        nObs = len(aux_df)
        # get indeces for test and train sample
        idx_train = np.random.choice(aux_df.index, size=round(share * nObs),
                                     replace=False)
        # append train data
        df_train = df_train.append(aux_df.loc[idx_train, :])
        df_test = df_test.append(aux_df.loc[~aux_df.index.isin(idx_train), :])
    # return both datasets
    return df_train, df_test



# --------------------------------------------------------------------------- #
#
# Performance Measurements (Predictions)
#
# --------------------------------------------------------------------------- #
          
# computes the mse
def mse(outcome, probabilities):
    """
    calculates the mean squared error of predictions in a set

    Parameters
    ----------
    - output: true outcomes
    - estimated_output: predicted values for the outcomes

    prints
    -------
    Mean squared error

    """
    # matrix with dummies (n x k). n observations, k categories.
    n_var = len(set(outcome))
    set_var = list(set(outcome))
    n = len(outcome)
    # containter for the dummies
    dummy_outcome = np.empty([n, n_var])
    # for all observations
    for i in range(n):
        # for all categories
        for j in range(n_var):
            if outcome[i] == set_var[j]:
                dummy_outcome[i,j] = 1
            else:
                dummy_outcome[i,j] = 0                
    # calculate the MSE
    mse = np.sum((probabilities - dummy_outcome)**2) / len(outcome)
    # return the mean squared error
    return(round(mse, 3))


# computes an alternative version for the mse
def alternative_mse(outcome, probabilities):
    """
    calculates alternative MSE

    Parameters
    ----------
    outcome : pd Series with the true outcomes
    probabilities : matrix with the probabilities

    Returns
    -------
    alternative MSE

    """
    # get the classes from the outcome
    classes = np.array(list(set(outcome)))
    # calculate intermediary result
    probs = probabilities @ classes
    # calcualte alternative mse
    alt_mse = np.mean((outcome - probs) ** 2)
    return(round(alt_mse,3))


# computes the classification accuracy
def ca(outcome, estimated_outcome):
    """
    calculate the classification accuracy (percentage of correctly predicted 
    outcomes)

    Parameters
    ----------
    - output: true outcomes
    - estimated_output: predicted values for the outcomes

    prints
    -------
    classification accuracy

    """
    ca = np.mean(outcome == estimated_outcome)
    return(ca)


# wrapper function to produce the performance measures
def pred_performance(outcome, probabilities, name):
    """
    performance table

    Parameters
    ----------
    outcome : np.ndarray
    probabilities : np.ndarray

    Returns
    -------
    performance table with MSE, alternative MSE, and accuracy

    """
    # calculate each measure with its function
    mse1 = mse(outcome, probabilities)
    alt_mse = alternative_mse(outcome, probabilities)
    # maps probabilities to vector with category with highest probabilities
    estimated_outcome = np.argmax(probabilities, axis=1)
    ca1 = ca(outcome, estimated_outcome)
    
    # get all together
    results = pd.DataFrame({'MSE': mse1, 'MSE 2': alt_mse, 'CA': ca1}, 
                           index=['value'])
    # print result
    return(results)
    print("Prediction Performance of {}".format(name), '-' * 40,
        results, '-' * 40, '\n\n', sep='\n')


# --------------------------------------------------------------------------- #
#
# Robustness Checks
#
# --------------------------------------------------------------------------- #

def outlier_comp(log_mar, orf_mar, treatment, labels, removal=True):
    '''
    This function can compare the marginal effects and extra points to expect 
    before and after the outlier removal.

    Parameters
    ----------
    log_mar : TYPE np.array
        DESCRIPTION. Ordered logistic marginal effects
    orf_mar : TYPE np.array
        DESCRIPTION. Ordered forest marginal effects
    treatment : TYPE string
        DESCRIPTION. The treatment name
    labels : TYPE a list of strings
        DESCRIPTION. The labels of outcome classes
    removal : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    robust_result : TYPE
        DESCRIPTION.

    '''
    # Build a dataframe to save the estimates
    robust_result=pd.DataFrame(columns=labels)
    # Ordered logit marginal effects
    robust_result.loc['ologit_BFGS']=log_mar 
    # Ordered forest marginal effects
    robust_result.loc['orf']=np.array(orf_mar.loc[treatment])
    # Number of additional points to expect
    robust_result['extra_points']=[np.dot(robust_result.loc['ologit_BFGS'],
                                          np.array([0,1,3])),
                                   np.dot(robust_result.loc['orf'],
                                          np.array([0,1,3]))]
    if removal:
    # print the results after outlier removal
        print('-'*80, 'After removal of outliers:', ' '*80,
              'The mean marginal effects and extra points of the treatment are',
              robust_result, '-'*80
              , sep='\n')
    # print the results before outlier removal
    else: 
        print('-'*80, 'Before removal of outliers:', ' '*80,
              'The mean marginal effects and extra points of the treatment are',
              robust_result, '-'*80
              , sep='\n')
    return robust_result

# it shows the changes in the marginal effects before and after outlier 
# removal
def oned_plot(x_at, x_bf, model_name):
    '''
    It shows the changes in the marginal effects before and after 
    outlier removal

    Parameters
    ----------
    x_at : TYPE np.array
        DESCRIPTION. It contains the marginal effect estimates after 
                     outlier removal
    x_bf : TYPE np.array
        DESCRIPTION. It contains the marginal effect estimates before
                     outlier removal
    model_name : TYPE string
        DESCRIPTION. It is the name of the model

    Returns
    -------
    None.

    '''
    # plot the marginal effect after eliminating the outliers
    plt.plot(x_at, np.zeros(len(x_at)),linestyle='--', marker='o', color='b',
             label='After outlier removal')
    # plot the marginal effect before eliminating the outliers
    plt.plot(x_bf, np.ones(len(x_bf)),linestyle='--', marker='o', color='g',
             label='Before outlier removal')
    plt.legend()
    plt.title(model_name)
    plt.yticks([]) # hide the y axis text
    plt.xlabel('Marginal Effects on 0, 1, and 3 points (from left to right)')
    plt.ylim([2,-1]) # broaden the graph
    plt.show()

# wrapper function mainly used for the robustness checks
def logit_orf_marginal_eff(data, outcome, X_name, labels, display=True):
    '''
    It is a shortcut to produce the marginal effects of ordered logit (using 
    a global optimiziation procedure) and ordered random forest regression 
    results and is used mainly for robustness tests.

    Parameters
    ----------
    data : TYPE: pd.DataFrame
        DESCRIPTION. It includes all the observations of the data.
    outcome : TYPE: string
        DESCRIPTION. It is the name of the outcome variable
    X_name : TYPE: list of strings
        DESCRIPTION. It includes the names of the covariates
    display : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    logit_result : TYPE: pd.DataFrame
        DESCRIPTION. The mean marginal effects estimated by a logit model.
    orf_result : TYPE
        DESCRIPTION. The mean marginal effects estimated by a orf model.

    '''
    # Logistic regression
    y = np.array(data[outcome]) # outcome variable in array
    X = np.array(data[X_name]) # covariates in array
    n = len(np.unique(y)) # length of the outcome variable
    results = fitLogistic(X, y, n, display=False, optim='L-BFGS-B') # logit
    coefficients= results[:-(n - 1)] # save the coefficients
    thresholds = results[-(n - 1):] # save the thresholds
    # the logit mean marginal effect output
    logit_result = mean_margeff(X, coefficients, thresholds)
    # Ordered Random Forest regression (orf)
    y2 = (pd.Series(y, name=outcome) + 1).copy() # outcome variable in pd.Series
    X2 = data[X_name].copy() # covariates in pd.DataFrame
    output = ordered_forest(X2, y2, labels) # fit a orf model
    # the orf output
    orf_result = marginal_eff(output, X2, labels, 
                                mean_mf=True, display=False) 
    if display:
        print('-'*80, 'Robustness test of Mean marginal effects','-'*80, 
              'Ordered logit: ', 
              logit_result, '-'*80, 
              'Ordered random forest:',
              orf_result, '-'*80, sep='\n')
    # return the ordered logit and orf output 
    return logit_result, orf_result



# --------------------------------------------------------------------------- #
#
# END
#
# --------------------------------------------------------------------------- #