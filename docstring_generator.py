def display_attributes(obj):
    """[summary]

    Args:
        obj ([type]): [description]

    Returns:
        (str): The instance attributes of the object
    """    
    attributes = ', '.join(i for i in dir(obj) if not i.startswith('_'))
    # source note: https://tinyurl.com/yyef5trl | stack overflow
    return attributes


def convertOutputToTrades(close_price, out_labels, capital=100000):
    """[summary]

    Returns:
        [type]: [description]
    """    
    start_c = capital
    num_shares = 0
    total_return = 0
    total_invested = 0
    rev = 0
    last_buy_prices = []
    for i in range(len(out_labels)):
        print('Num_shares = ', num_shares)
        if out_labels[i] == 1 and capital >= close_price[i]:
            num_shares += 1
            capital -= close_price[i]
            total_invested += close_price[i]
            last_buy_prices.append(close_price[i])
        elif out_labels[i] == 0 and num_shares > 0:
            capital += num_shares*close_price[i]
            rev += num_shares*close_price[i]
            for price in last_buy_prices[-num_shares:]:
                total_return += close_price[i] - price
                #total_invested -= price
            num_shares = 0
    if num_shares > 0:
        capital += num_shares*close_price[-1]
        rev += num_shares*close_price[-1]
        total_return += num_shares*(close_price[-1] - last_buy_prices[-1])
        num_shares = 0
        
    TR = (capital - start_c)/start_c
    TR2 = (rev - total_invested)/total_invested
    
    return total_invested, rev, TR, TR2