%% Matlab code to check the stepwise variable selection performance and results for Q7.

W = csvread('weather.csv',1,1);
E = csvread('energy.csv',1,1);

% stepwiselm(W,E,'linear')
[b,se,pval,inmodel,stats,nextstep,history] = stepwisefit(W,E);

n = length(E);
rsq = 1 - (history.rmse.^2/var(E)) .* ((n-1-history.df0)/(n-1));