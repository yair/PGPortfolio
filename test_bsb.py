from pgportfolio.trade.buysellbot import BuySellBot

bsb = BuySellBot(300, ['reversed_USDT', 'ETH', 'BCH', 'LTC', 'STR'])
bsb.rebalance_portfolio([1, 0, 0, 0, 0, 0], # prev_omega,
                        [0, 1, 0, 0, 0, 0], # next_omega,
                        [0.01, 0, 0, 0, 0, 0], # prev_balances,
                        0.01, # total_capital,
                        [1.00000000e+00, 1.33582682e-04, 7.64150023e-02, 1.32082298e-01, 1.57477893e-02, 3.91799986e-05]) # prices)
                                     
