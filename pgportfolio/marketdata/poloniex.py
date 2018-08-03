import json
import time
import sys
import requests
from datetime import datetime
# pip3 install https://github.com/s4w3d0ff/python-poloniex/archive/v0.4.7.zip
from poloniex import Poloniex as pololib
from traceback import print_stack

if sys.version_info[0] == 3:
    from urllib.request import Request, urlopen
    from urllib.parse import urlencode
else:
    from urllib2 import Request, urlopen
    from urllib import urlencode

# Possible Commands
PUBLIC_COMMANDS = ['returnTicker', 'return24hVolume', 'returnOrderBook', 'returnTradeHistory', 'returnChartData', 'returnCurrencies', 'returnLoanOrders']
PRIVATE_COMMANDS = ['returnBalances', 'returnCompleteBalances']

class Poloniex:
    def __init__(self, APIKey='', Secret=''):
        self.APIKey = APIKey.encode()
        self.Secret = Secret.encode()
        self._proxies = {'http': 'socks5://127.0.0.1:4711', # socks5://<usr>:<pwd>@<addr>:<port>
                         'https': 'socks5://127.0.0.1:4711'} #use only if you are using **socks**
        # Conversions
        self.timestamp_str = lambda timestamp=time.time(), format="%Y-%m-%d %H:%M:%S": datetime.fromtimestamp(timestamp).strftime(format)
        self.str_timestamp = lambda datestr=self.timestamp_str(), format="%Y-%m-%d %H:%M:%S": int(time.mktime(time.strptime(datestr, format)))
        self.float_roundPercent = lambda floatN, decimalP=2: str(round(float(floatN) * 100, decimalP)) + "%"
        self.banlist = { 'FLO':1, 'FLDC':1, 'XVC':1, 'BCY':1, 'NXC':1, 'RADS':1, 'BLK':1, 'PINK':1, 'RIC':1 }   # 2.8.2018 delisting

        # PUBLIC COMMANDS
        self.marketTicker = lambda x=0: self.api('returnTicker')
        self.marketVolume = lambda x=0: self.api('return24hVolume')
        self.marketStatus = lambda x=0: self.api('returnCurrencies')
        self.marketLoans = lambda coin: self.api('returnLoanOrders', {'currency': coin})
        self.marketOrders = lambda pair='all', depth=10:\
            self.api('returnOrderBook', {'currencyPair': pair, 'depth': depth})
        self.marketChart = lambda pair, period=const.DAY, start=time.time() - (const.WEEK * 1), end=time.time(): self.api('returnChartData', {'currencyPair': pair, 'period': period, 'start': start, 'end': end})
        self.marketTradeHist = lambda pair: self.api('returnTradeHistory', {'currencyPair': pair})  # NEEDS TO BE FIXED ON Poloniex

        # PRIVATE COMMANDS
        self.balances = lambda x=0: self.api('returnBalances')
        self.completeBalances = lambda x=0: self.api('returnCompleteBalances')
        self.polo = pololib ('L7SOV94G-OEML34LQ-04HKAAGN-KM2QK0AV',         # TODO: move to ungitted file
                             'aa68905cc5eca8556eac2e5c5edee1ddbfc2679f5d30b137236da380e89c5a0e8a129263b628e3c5edde4b65abf2f9f5c919221eec6d8205323c3fbcc8f09696',
                             proxies=self._proxies)

    #####################
    # Main Api Function #
    #####################
    def api(self, command, args={}):
        """
        returns 'False' if invalid command or if no APIKey or Secret is specified (if command is "private")
        returns {"error":"<error message>"} if API error
        """
#        print_stack()
        #proxies = {'http': 'socks5://127.0.0.1:1080', # socks5://<usr>:<pwd>@<addr>:<port>
        #           'https': 'socks5://127.0.0.1:1080'} #use only if you are using **socks**
        proxies = {'http': 'socks5://127.0.0.1:4711', # socks5://<usr>:<pwd>@<addr>:<port>
                   'https': 'socks5://127.0.0.1:4711'} #use only if you are using **socks**
        if command in PUBLIC_COMMANDS:
            url = 'https://poloniex.com/public?'
            args['command'] = command
#            return json.loads(requests.get(url+urlencode(args), proxies=proxies).text)
            return json.loads(requests.get(url+urlencode(args), proxies=self._proxies).text)
#            return json.loads(requests.get(url+urlencode(args)).text)
#            ret = urlopen(Request(url + urlencode(args)))
#            return json.loads(ret.read().decode(encoding='UTF-8'))
        elif command in PRIVATE_COMMANDS:
            return self.polo(command)
        else:
            return False
