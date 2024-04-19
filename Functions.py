import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def stk_ret(df, weekly=True):
    if (weekly==True):
        first_price = df.iloc[-6]['Close']
        last_price = df.iloc[-1]['Close'] 
        stock_ret = (last_price-first_price)/first_price
    else:
        first_price = df.iloc[0]['Close']
        last_price = df.iloc[-1]['Close'] 
        stock_ret = (last_price-first_price)/first_price
    
    return stock_ret

def get_market_cap(ticker):
    try:
        stock = yf.Ticker(ticker)
        market_cap = stock.info.get("marketCap")
        if market_cap:
            return market_cap
        else:
            return "Market cap data not available for this ticker."
    except Exception as e:
        return str(e)
    
def stock_df_create(ticker_list, start_date, end_date):
    dic = {'TKR': ticker_list, 'LAST_PRRICE':"", 'MKT_CAP': "", 'W_RET': "", "M_RET": ""}
    mkt_caps = []
    w_rets = []
    m_rets = []
    last_prices = []

    for i in ticker_list:
        yf_stock_df = yf.download(i, start=start_date, end=end_date) 
        m_w_ret = stk_ret(yf_stock_df)
        m_m_ret = stk_ret(yf_stock_df, weekly=False)
        m_mkt_cap = get_market_cap(i)
        m_l_price = yf_stock_df['Close'][-1]
        
        mkt_caps.append(m_mkt_cap)
        w_rets.append(m_w_ret)
        m_rets.append(m_m_ret)
        last_prices.append(m_l_price)
    
    dic['LAST_PRICE'] = last_prices
    dic['MKT_CAP'] = mkt_caps
    dic['W_RET'] = w_rets
    dic['M_RET'] = m_rets

        
    df = pd.DataFrame(dic)
    
    return df

def get_current_weights(df):
    df["DollarAmount"] = df['Quantity']*df['LastPrice']
    port_val = np.sum(df['DollarAmount'])
    cur_weights = df['DollarAmount']/port_val
    dic = dict(zip(df['TKR'], cur_weights))
    
    return dic

def rebalance_portfolio(cur_weights, new_weights, cur_port, threshold=0.01):
    merged_weights = pd.merge(cur_weights, new_weights, on="TKR")
    
    merged_port = pd.merge(merged_weights, cur_port, on='TKR', how='left')
    
    merged_port['DollarAmount_cur'] = merged_port['Quantity'] * merged_port['LastPrice']
    
    total_value = merged_port['DollarAmount_cur'].sum()
    merged_port['DollarAmount_new'] = total_value * merged_port['Weights_new']
    
    merged_port['DollarAmount_diff'] = merged_port['DollarAmount_new'] - merged_port['DollarAmount_cur']
    
    merged_port['Shares_to_trade'] = merged_port['DollarAmount_diff'] / merged_port['LastPrice']
    
    merged_port['Shares_to_trade'] = merged_port['Shares_to_trade'].apply(lambda x: np.round(x))
    
    trades = merged_port[(merged_port['Shares_to_trade'] != 0) & (np.abs(merged_port['DollarAmount_diff']) > threshold * total_value)]
    
    total_trades = trades.shape[0]
    total_dollars = trades['DollarAmount_diff'].sum()
    
    return trades, total_trades, total_dollars

def additional_cash_investments(df, remaining_cash):
    
    additional_investments = df['uptd_Weights'] * remaining_cash
    additional_shares = np.floor(additional_investments / df['LastPrice'])
    
    result_df = pd.DataFrame({'TKR': df['TKR'], 'Additional_Shares': additional_shares})
    
    return result_df

# PERFORMANCE EVALUATION

# Data

def create_panel_data(tickers, start_date, end_date):
    dfs = []  
    
    for ticker in tickers:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        
        stock_data['Ticker'] = ticker
        
        dfs.append(stock_data)
    
    panel_data = pd.concat(dfs)
    
    return panel_data

def create_portfolio_df(stock_df):
    stock_df['Stock_Value'] = stock_df['Quantity'] * stock_df['Adj Close']
    
    stock_df['Value_USD'] = stock_df['Stock_Value']
    stock_df.loc[stock_df['Currency'] != 'USD', 'Value_USD'] *= stock_df['USDEUR']
    
    portfolio_df = stock_df.groupby('Date')['Value_USD'].sum().reset_index()
    
    portfolio_df['Daily_RET'] = portfolio_df['Value_USD'].pct_change()
    
    return portfolio_df.set_index('Date')

# Metrics

def rets_plot(return_series1, return_series2, return_series3, initial_value=100):
    
    cumulative_returns1 = (1 + return_series1).cumprod() * initial_value
    cumulative_returns2 = (1 + return_series2).cumprod() * initial_value
    cumulative_returns3 = (1 + return_series3).cumprod() * initial_value
    
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_returns1.index, cumulative_returns1, label='Portfolio')
    plt.plot(cumulative_returns2.index, cumulative_returns2, label='Benchmark')
    plt.plot(cumulative_returns3.index, cumulative_returns3, label='SP500')
    plt.xticks(rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Returns')
    plt.title('Performance')
    plt.legend()
    plt.grid(False)
    
    date_formatter = mdates.DateFormatter('%Y-%m-%d')  # Define date format
    plt.gca().xaxis.set_major_formatter(date_formatter) 
    
    plt.show()
    
def plot_portfolio_change(portfolio_values):
    initial_value = portfolio_values.iloc[0]
    percentage_change = ((portfolio_values - initial_value) / initial_value) * 100
    
    plt.figure(figsize=(10, 6))
    plt.plot(percentage_change, color='blue')
    plt.xticks(rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Percentage Change (%)')
    plt.title('Change in Portfolio Value (Relative to Initial)')
    plt.grid(False)

    plt.grid(True)
    plt.show()