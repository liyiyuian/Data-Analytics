import pandas as pd
import statsmodels.api as sm

class Model:
    def __init__(self, name):
        self.name = name
        self.result = []
        self.rsqr = 0.0
        self.params = []

def stepwise(X, y, mdl=Model("mdl"), 
                       initial_list=[], 
                       threshold_in=0.05, 
                       threshold_out = 0.1):
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
        if best_pval < threshold_in:    # add
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            print("Iteration {}\tAdd  {:30} with p-value {:.6}\tR-Squared={:.6}".format(i,best_feature,best_pval,model.rsquared))
            i += 1

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() 
        if worst_pval > threshold_out:    # remove
            changed=True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            print("Iteration {}\tDrop {:30} with p-value {:.6}\tR-Squared={:.6}".format(i,worst_feature,worst_pval,model.rsquared))
            i += 1
        if not changed:
            break
    
    mdl.result = included
    mdl.rsqr = model.rsquared
    mdl.params = model.params
    
    print('\n'+'===='*20)
    print(mdl.name + " results:")
    print('Selected variables: {}'.format(mdl.result))
    print("R-Squared = {}\n".format(mdl.rsqr))

    return mdl