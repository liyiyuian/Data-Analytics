import pandas as pd
import statsmodels.api as sm

class Model:
    def __init__(self, name):
        self.name = name
        self.result = []
        self.rsqr = 0.0
        
def stepwise(X, y, mdl=Model("mdl"), 
                       initial_list=[], 
                       threshold_in=0.05, 
                       threshold_out = 0.1, 
                       verbose=True):
    """ Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: 
        list of selected variables 
        R-Squared
    """
    included = list(initial_list)
    i = 1
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print("Iteration {}".format(i))
                i += 1
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() 
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print("Iteration {}".format(i))
                i += 1
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
        print("R-Squared={}\n".format(model.rsquared))
    
    mdl.result = included
    mdl.rsqr = model.rsquared
    
    print('\n'+'===='*20)
    print(mdl.name + " results:")
    print('Selected variables: {}'.format(mdl.result))
    print("R-Squared = {}\n".format(mdl.rsqr))

    return included, model.rsquared