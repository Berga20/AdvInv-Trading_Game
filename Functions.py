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

# Plots

def rets_plot_report(return_series1, return_series2, initial_value=100):
    
    cumulative_returns1 = (1 + return_series1).cumprod()
    cumulative_returns2 = (1 + return_series2).cumprod()
    
    
    cumulative_returns1_norm = cumulative_returns1 / cumulative_returns1.iloc[0] * initial_value
    cumulative_returns2_norm = cumulative_returns2 / cumulative_returns2.iloc[0] * initial_value

    
    plt.figure(figsize=(10, 6))
    
    plt.plot(cumulative_returns1_norm.index, cumulative_returns1_norm, color='gold', label='Portfolio')
    plt.plot(cumulative_returns2_norm.index, cumulative_returns2_norm, color='darkred', label='VICEX')
    
    plt.axhline(initial_value, linestyle='--', color='grey', linewidth=0.5)
    
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(False)
    plt.show()

def rets_plot(return_series1, return_series2, return_series3, return_series4, initial_value=100):
    
    cumulative_returns1 = (1 + return_series1).cumprod()
    cumulative_returns2 = (1 + return_series2).cumprod()
    cumulative_returns3 = (1 + return_series3).cumprod()
    cumulative_returns4 = (1 + return_series4).cumprod()
    
    
    cumulative_returns1_norm = cumulative_returns1 / cumulative_returns1.iloc[0] * initial_value
    cumulative_returns2_norm = cumulative_returns2 / cumulative_returns2.iloc[0] * initial_value
    cumulative_returns3_norm = cumulative_returns3 / cumulative_returns3.iloc[0] * initial_value
    cumulative_returns4_norm = cumulative_returns4 / cumulative_returns4.iloc[0] * initial_value
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(cumulative_returns1_norm.index, cumulative_returns1_norm, color='gold', label='Portfolio')
    plt.plot(cumulative_returns2_norm.index, cumulative_returns2_norm, color='darkred', label='VICEX')
    plt.plot(cumulative_returns3_norm.index, cumulative_returns3_norm, color='darkorange', label='SP500')
    plt.plot(cumulative_returns4_norm.index, cumulative_returns4_norm, color='darkblue', label='SWDA')
    
    plt.axhline(initial_value, linestyle='--', color='grey', linewidth=0.5)
    
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(False)
    plt.show()
    
def weekly_returns_bar_chart(return_series1, return_series2):
    df = pd.DataFrame({'Series1': return_series1, 'Series2': return_series2})
    
    start_date = max(df.index.min() for df in [return_series1, return_series2])
    end_date = min(df.index.max() for df in [return_series1, return_series2])
    
    return_series1 = return_series1.loc[start_date:end_date]
    return_series2 = return_series2.loc[start_date:end_date]
    
    weekly_returns = pd.DataFrame({
        'Portfolio': return_series1.resample('W').sum(),
        'VICEX': return_series2.resample('W').sum()
    })
    
    plt.figure(figsize=(10, 6))
    weekly_returns.plot(kind='bar', color=['gold', 'darkred'], alpha=0.8)

    plt.xticks([])
    plt.axhline(0, linestyle='--', color='grey', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
# Metrics

def performance_metrics(portfolio_value, return_series, benchmark_series=None, risk_free_rate=0):
    
    total_return = (portfolio_value[-1]-1000000)/1000000
    
    avg_daily_return = return_series.mean()
    
    std_return = return_series.std()
    
    sharpe_ratio = (total_return-risk_free_rate) / std_return
    
    if benchmark_series is not None:
        total_outperformance = (return_series - benchmark_series).sum()
    
        avg_daily_outperformance = (return_series - benchmark_series).mean()
    
        tracking_error = (return_series - benchmark_series).std()
    
        information_ratio = (return_series - benchmark_series).mean() / tracking_error
    
        correlation_coefficient = return_series.corr(benchmark_series)
    
    best_day = return_series.idxmax(), return_series.max()
    
    worst_day = return_series.idxmin(), return_series.min()
    
    cumulative_returns = (1 + return_series).cumprod()
    max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
    
    if benchmark_series is not None:
        msquared = (sharpe_ratio * benchmark_series.std()) + risk_free_rate
    
    return pd.DataFrame({
        'Total Return': total_return,
        'Average Daily Return': avg_daily_return,
        'Standard Deviation': std_return,
        'Sharpe Ratio': sharpe_ratio,
        'Total Outperformance': total_outperformance,
        'Average Daily Outperformance': avg_daily_outperformance,
        'Tracking Error': tracking_error,
        'Information Ratio': information_ratio,
        'Correlation Coefficient': correlation_coefficient,
        'Best Day': best_day,
        'Worst Day': worst_day,
        'Maximum Drawdown': max_drawdown,
        'M2': msquared
    }).T