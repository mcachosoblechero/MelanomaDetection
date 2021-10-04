
import matplotlib.pyplot as plt

#####################################
# Pre-procesing Methods             #
#####################################
# Obtain sorted list of percentage of missing values


def perc_missing_vals(df):
    missing_vals_train = df.isnull().sum() / df.shape[0] * 100
    print(missing_vals_train.sort_values(ascending=False))


#####################################
# Visualisation Methods             #
#####################################
# Create cummulative bar plots for proportions from 0 to 1
def CumPerc_BarPlot(X, Y1, Y2):
    plt.figure(figsize=(12, 5))
    plt.bar(X, Y1, edgecolor='white')               # Create first set of bars
    plt.bar(X, Y2, bottom=Y1, edgecolor='white')    # Create second set of bars
