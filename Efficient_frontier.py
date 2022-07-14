import numpy as np
import pandas as pd
import datetime as dt
from pandas_datareader import data as pdr
import scipy.optimize as sc
import plotly.graph_objects as go
import plotly.io as pio


pio.renderers.default = 'browser'

#Get Data

def get_data (stocks, start_date, end_date):
    
    stock_data = pdr.get_data_yahoo(stocks, start = start_date, end = end_date)
    stock_data = stock_data['Close']
    returns = stock_data.pct_change()
    mean_returns = returns.mean()
    cov_mat = returns.cov()
    
    return mean_returns, cov_mat


def portfolio_performance(weights, mean_returns, cov_mat):
    returns = np.sum(mean_returns*weights)*248
    std= np.sqrt(np.dot(weights.T, np.dot(cov_mat,weights))) *np.sqrt(248)
    return returns, std

def negative_sharpe (weights, mean_returns, cov_mat, rf = 0):
    p_returns, p_std = portfolio_performance(weights, mean_returns, cov_mat)
    return -(p_returns - rf)/p_std

def max_sharpe (mean_returns, cov_mat, rf = 0, constraint_set = (0,1)):
    #Minimize the negative sharpe ratio by changing the weights
    num_assets = len(mean_returns)
    args = (mean_returns, cov_mat, rf)
    constraints = ({'type':'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraint_set
    bounds = tuple(bound for asset in range(num_assets))
    result = sc.minimize(negative_sharpe, num_assets*[1./num_assets], args = args,
                         method = 'SLSQP', bounds= bounds, constraints = constraints)
    
    return result

def port_variance(weights, mean_returns, cov_mat):
    return portfolio_performance(weights, mean_returns, cov_mat)[1]

def port_returns(weights, mean_returns, cov_mat):
    return portfolio_performance(weights, mean_returns, cov_mat)[0]

def min_port_variance(mean_returns, cov_mat, rf = 0, constraint_set = (0,1))   :
    #Minimize portfolio variance by changing weights in the portfolio
    num_assets = len(mean_returns)
    args = (mean_returns, cov_mat)
    constraints = ({'type':'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraint_set
    bounds = tuple(bound for asset in range(num_assets))
    result = sc.minimize(port_variance, num_assets*[1./num_assets], args = args, method = 'SLSQP', bounds= bounds, constraints = constraints)
    return result

def calculated_results (mean_returns, cov_mat, rf = 0, constraint_set = (0,1)):
    #Take inputs and calculate min_sharpe, min_variance and efficient frontier
    
    #Max sharpe ratio portfolio 
    max_sharpe_port = max_sharpe(mean_returns, cov_mat)
    max_sharpe_returns, max_sharpe_std = portfolio_performance(max_sharpe_port['x'], mean_returns, cov_mat)
    
    max_sharpe_alloc = pd.DataFrame(max_sharpe_port['x'],index = mean_returns.index, columns= ['allocation'])
    max_sharpe_alloc.allocation = [round(i*100,0) for i in max_sharpe_alloc.allocation]
    
    #Min variance portfolio
    min_var_port = min_port_variance(mean_returns, cov_mat)
    min_var_returns, min_var_std = portfolio_performance(min_var_port['x'], mean_returns, cov_mat)
    
    min_var_alloc = pd.DataFrame(min_var_port['x'],index = mean_returns.index, columns= ['allocation'])
    min_var_alloc.allocation = [round(i*100,0) for i in min_var_alloc.allocation]
    
    target_returns = np.linspace(min_var_returns, max_sharpe_returns, 20)
    efficient_list = []
    for target in target_returns:
        efficient_list.append(efficient_frontier(mean_returns, cov_mat, target)['fun'])
    
    max_sharpe_returns, max_sharpe_std = round(max_sharpe_returns*100 , 2), round(max_sharpe_std*100 , 2)    
    min_var_returns, min_var_std = round(min_var_returns*100,2), round(min_var_std*100,2)
    
    return max_sharpe_returns, max_sharpe_alloc, max_sharpe_std, min_var_returns , min_var_std , min_var_alloc, efficient_list, target_returns 
    
    
def efficient_frontier(mean_returns, cov_mat, return_target, constraint_set = (0,1)):
    #For each return taget, we want to optimze the portfolio
    num_assets = len(mean_returns)
    args = (mean_returns, cov_mat)
    weights = num_assets*[1./num_assets]
    constraints = ({'type':'eq', 'fun' : lambda x: port_returns(x, mean_returns, cov_mat) - return_target}, 
                    {'type':'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraint_set
    bounds = tuple(bound for asset in range(num_assets))
    efficient_port = sc.minimize(port_variance, weights, args = args, method = 'SLSQP', bounds = bounds, constraints= constraints )
    return efficient_port
    
def efficient_graph(mean_returns, cov_mat, rf = 0, constraint_set = (0,1))  :
    #to plot the efficient frontier 
    max_sharpe_returns, max_sharpe_alloc, max_sharpe_std, min_var_returns , min_var_std , min_var_alloc, efficient_list, target_returns = calculated_results (mean_returns, cov_mat, rf, constraint_set)
    
    #max sharpe ratio
    max_sharpe_ratio = go.Scatter(
        name = 'Max Sharpe Ratio',
        mode = 'markers',
        x = [max_sharpe_std],
        y = [max_sharpe_returns],
        marker = dict(color = 'red' , size = 14 , line = dict(width = 3, color = 'black'))
        )
    #min vols
    min_vol_port = go.Scatter(
        name = 'Min Vol Portfolio',
        mode = 'markers',
        x = [min_var_std],
        y = [min_var_returns],
        marker = dict(color = 'green' , size = 14 , line = dict(width = 3, color = 'black'))
        )
    #ef curve
    ef_curve = go.Scatter(
        name = 'Efficient Frontier',
        mode = 'lines',
        x = [round(ef_std*100,2) for ef_std in efficient_list],
        y = [round(target*100,2) for target in target_returns],
        line = dict(color = 'black', width = 4, dash = 'dashdot')
        )
    
    data = [max_sharpe_ratio, min_vol_port, ef_curve]
    
    layout = go.Layout(
        title = 'Markowitz Optimmization with efficient frontier',
        yaxis = dict(title = 'Annualised returns(%)'),
        xaxis = dict(title = 'Annualised vol'),
        showlegend = True,
        legend = dict(x = 0.75, y = 0, traceorder = 'normal', bgcolor = '#E2E2E2', borderwidth = 2),
        width = 800,
        height = 600
        )
    fig = go.Figure(data = data, layout = layout)
    return fig.show(), max_sharpe_alloc
        
 
    
def portfolio_returns (portfolio_alloc, stocks, start_date, end_date):
    
    stock_data = pdr.get_data_yahoo(stocks, start = start_date, end = end_date)
    stock_data = stock_data['Close']
    returns = np.zeros(len(stocks))
    j = 0
    for i in stocks:
        returns[j] = (stock_data[i][-1]/stock_data[i][0] - 1);
        j += 1
    ind_returns = portfolio_alloc[1]['allocation']*returns
    p_returns = sum(ind_returns)
    n = (end_date - start_date).days
    p_returns = pow((1+p_returns/100), 365/n) - 1
    return p_returns, returns, ind_returns
    
#enter stock tickers
stock_list = ['PG', 'AAPL', 'KO', 'MA', 'DPZ']


#inputs
end_date = dt.datetime.now() - dt.timedelta(1000)
start_date = end_date - dt.timedelta(500)

mean_returns, cov_mat = get_data(stock_list, start_date, end_date)

portfolio_alloc = efficient_graph(mean_returns, cov_mat)

print(portfolio_alloc)


start_date = end_date
end_date = start_date + dt.timedelta(365)
p_returns, returns, ind_returns = portfolio_returns(portfolio_alloc, stock_list, start_date, end_date)

print("portfolio returns over 1 year: " + str(round(p_returns*100,2)) + "%")

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    