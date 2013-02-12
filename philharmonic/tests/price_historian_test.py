'''
Created on Oct 22, 2012

@author: kermit
'''
import unittest
from philharmonic.energy_price import historian
import pandas as pd
from datetime import datetime
class Test(unittest.TestCase):


    def testParseHourly(self):
        prices = historian.parse_prices("./io/tests/energy_price_data-test.csv")
        self.assertIsInstance(prices, pd.Series, "must get a series")
        self.assertIsInstance(prices, pd.TimeSeries, "must get a TimeSeries")
        
        places = 5 # decimal places to consider for equality
        t1 = datetime(2012,5,5,0,0)
        t2 = datetime(2012,5,5,1,0)
        self.assertEqual(prices.index[0], t1, "expecting to start on this date")
        self.assertAlmostEqual(prices[0], 0.01855, places, "expecting to start with this value")
        self.assertEqual(prices.index[1], t2, "expecting to continue on this date")
        self.assertAlmostEqual(prices[1], 0.01826, places, "expecting to continue with this value")
        
        #expected_end = datetime.strptime("2012-09-04", "%Y-%m-%d")
        tn = datetime(2012,9,4,23,0)
        self.assertEqual(prices.index[-1], tn, "expecting to end on this date")
        self.assertAlmostEqual(prices[-1], 0.02372, places,
                               "expecting to end with this value")
        
    def testRealign(self):
        prices = historian.parse_prices("./io/tests/energy_price_data-test.csv")
        prices = historian.realign(prices, datetime(2012,5,8))
        t1 = datetime(2012,5,8,0,0)
        t2 = datetime(2012,5,8,1,0)
        self.assertEqual(prices.index[0], t1, "expecting to start on this date")
        self.assertEqual(prices.index[1], t2, "expecting to continue on this date")
        
        tn = datetime(2012,9,7,23,0)
        #expected_end = datetime.strptime("2012-09-07", "%Y-%m-%d")
        self.assertEqual(prices.index[-1], tn, "expecting to end on this date")

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testParseHourly']
    unittest.main()