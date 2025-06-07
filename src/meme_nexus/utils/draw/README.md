# 添加新绘图模块

1. 先在 config.py 中添加配置（如需要）                                                                  

```                                                                            
# 如果需要新的颜色或样式配置                                                                      
class ColorScheme(TypedDict):                                                                           
    # ... 现有颜色                                                                                      
    new_color: str  # 新增颜色                                                                          
                                                                                                        
def get_color_scheme() -> ColorScheme:                                                                  
    return ColorScheme(                                                                                 
        # ... 现有颜色                                                                                  
        new_color="#123456",  # 新增颜色值                                                              
    )                                                                                                   
```

2. 在 indicators.py 中添加计算逻辑（如需要）                                                            

```                                                                                                    
def calculate_new_indicator(                                                                            
    df: pd.DataFrame,                                                                                   
    param1: int = 20,                                                                                   
    param2: float = 0.5                                                                                 
) -> pd.DataFrame:                                                                                      
    """                                                                                                 
    计算新指标                                                                                          
                                                                                                        
    Args:                                                                                               
        df: OHLC数据                                                                                    
        param1: 参数1                                                                                   
        param2: 参数2                                                                                   
                                                                                                        
    Returns:                                                                                            
        包含新指标的DataFrame                                                                           
    """                                                                                                 
    # 实现计算逻辑                                                                                      
    pass                                                                                                

```

3. 在 elements.py 中添加绘图函数                                                                        

```                                                                                                    
def draw_new_element(                                                                                   
    ax,                                                                                                 
    ohlc: pd.DataFrame,                                                                                 
    indicator_data: pd.DataFrame,                                                                       
    dark_mode: bool = True                                                                              
) -> list:                                                                                              
    """                                                                                                 
    绘制新的图表元素                                                                                    
                                                                                                        
    Args:                                                                                               
        ax: matplotlib轴对象                                                                            
        ohlc: OHLC数据                                                                                  
        indicator_data: 指标数据                                                                        
        dark_mode: 是否深色模式                                                                         
                                                                                                        
    Returns:                                                                                            
        addplots列表                                                                                    
    """                                                                                                 
    colors = get_color_scheme()                                                                         
    addplots = []                                                                                       
                                                                                                        
    # 实现绘图逻辑                                                                                      
                                                                                                        
    return addplots                                                                                     
```                                                                                                    


4. 在 main.py 中集成新功能                                                                              

```                                                                                                  
# 1. 添加新参数                                                                                         
def plot_candlestick(                                                                                   
    # ... 现有参数                                                                                      
    is_draw_new_element=False,  # 新增参数                                                              
    # ...                                                                                               
):                                                                                                      
    # 2. 在指标计算部分添加逻辑                                                                         
    if is_draw_new_element:                                                                             
        if indicators is not None and "new_indicator" in indicators:                                    
            new_data = indicators["new_indicator"]                                                      
        else:                                                                                           
            new_data = calculate_new_indicator(ohlc)                                                    
                                                                                                        
        # 3. 调用绘图函数                                                                               
        new_plots = draw_new_element(ax, ohlc, new_data, dark_mode)                                     
        addplots.extend(new_plots)                                                                      
```                                                                                                      


5. 更新示例文件（可选）                                                                                 

```                                                                                         
# examples/draw_with_new_element.py                                                                     
async def main():                                                                                       
    # 展示如何使用新功能                                                                                
    file_path, mime_type, base64_string = plot_candlestick(                                             
        df,                                                                                             
        symbol="BTC/USDT",                                                                              
        timeframe="h",                                                                                  
        aggregate=1,                                                                                    
        is_draw_new_element=True,  # 启用新功能                                                         
    )                                                                                                   
```

实际例子：添加移动平均线模块                                                                            

 1 在 elements.py 中添加：                                                                              

```                                                                                                    
def draw_moving_averages(                                                                               
    ohlc: pd.DataFrame,                                                                                 
    ma_periods: list[int] = [20, 50, 200]                                                               
) -> list:                                                                                              
    """绘制移动平均线"""                                                                                
    colors = get_color_scheme()                                                                         
    addplots = []                                                                                       
                                                                                                        
    ma_colors = [colors["yellow"], colors["cyan"], colors["blue"]]                                      
                                                                                                        
    for period, color in zip(ma_periods, ma_colors):                                                    
        if len(ohlc) >= period:                                                                         
            ma = ohlc["close"].rolling(window=period).mean()                                            
            ma_plot = mpf.make_addplot(                                                                 
                ma,                                                                                     
                color=color,                                                                            
                width=1,                                                                                
                panel=0,                                                                                
                label=f"MA{period}"                                                                     
            )                                                                                           
            addplots.append(ma_plot)                                                                    
                                                                                                        
    return addplots                                                                                     
```                                                                                                    

2 在 main.py 中添加：                                                                                  

```                                                                                                    
def plot_candlestick(                                                                                   
    # ... 现有参数                                                                                      
    is_draw_ma=False,                                                                                   
    ma_periods: list[int] = [20, 50, 200],                                                              
    # ...                                                                                               
):                                                                                                      
    # 在适当位置添加                                                                                    
    if is_draw_ma:                                                                                      
        ma_plots = draw_moving_averages(ohlc, ma_periods)                                               
        addplots.extend(ma_plots)                                                                       
```                                                                                                

这种顺序的好处是：                                                                                      

 • 配置先行：确保所有需要的配置都已准备好                                                               
 • 逻辑分离：计算和绘图逻辑分开，便于测试和维护                                                         
 • 渐进集成：最后才修改主函数，降低破坏现有功能的风险                                                   
 • 易于回滚：如果出现问题，可以轻松禁用新功能       