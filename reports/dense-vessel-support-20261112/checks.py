"""Tiny metadata-only negative cases for the supplementary support analysis."""
import copy
import unittest
import analyze


class CountPairChecks(unittest.TestCase):
    def fixture(self):
        return {'p': {'record_ids':['r'],'dataset':'d','channel_datasets':{'vessel':'d'},'channels':{'vessel':{'tp':4,'fp':2,'fn':1,'valid_pixels':10}}}}
    def test_actual_valid_pair(self):
        a=self.fixture();analyze.validate_pair(a,copy.deepcopy(a))
    def test_changed_GT_positive_rejected(self):
        a=self.fixture();b=copy.deepcopy(a);b['p']['channels']['vessel']['fn']=2
        with self.assertRaises(ValueError):analyze.validate_pair(a,b)
    def test_changed_valid_pixels_rejected(self):
        a=self.fixture();b=copy.deepcopy(a);b['p']['channels']['vessel']['valid_pixels']=11
        with self.assertRaises(ValueError):analyze.validate_pair(a,b)
    def test_changed_source_rejected(self):
        a=self.fixture();b=copy.deepcopy(a);b['p']['channel_datasets']['vessel']='other'
        with self.assertRaises(ValueError):analyze.validate_pair(a,b)
    def test_missing_photo_rejected(self):
        with self.assertRaises(ValueError):analyze.validate_pair(self.fixture(),{})
    def test_boolean_or_impossible_count_rejected(self):
        for bad in [{'tp':True,'fp':0,'fn':0,'valid_pixels':1},{'tp':3,'fp':3,'fn':3,'valid_pixels':8}]:
            with self.assertRaises(ValueError):analyze.checked_counts(bad)

if __name__=='__main__':unittest.main()
