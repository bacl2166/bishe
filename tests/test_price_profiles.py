import copy
import json
import unittest
from collections import Counter
from pathlib import Path

from exp1.constraints import build_requests


ROOT = Path(__file__).resolve().parents[1]


class PriceProfileTests(unittest.TestCase):
    def test_price_profiles_keep_catalog_and_protocol_with_independent_outputs(self):
        prep = json.loads((ROOT/'configs/expanded/experiment1_musical.json').read_text())
        first = json.loads((ROOT/'configs/price-exp1/experiment1_musical.json').read_text())
        self.assertEqual(first['constraint_families'], ['no_filter','price','random_matched'])
        self.assertEqual(first['max_requests'], first['max_users'] * 13)
        self.assertEqual(first['max_items'], prep['max_items'])
        self.assertNotEqual(first['output'], prep['output'])
        self.assertEqual(first['encodings'], prep['encodings'])
        self.assertEqual(len(first['encodings']), 12)
        self.assertEqual(first['tokenizer_overrides']['attribute_first']['attributes'], ['price'])
        self.assertEqual(first['scorers'], ['most_popular', 'item_knn', 'sasrec'])
        self.assertEqual(first['max_requests'] * len(first['encodings']) * len(first['scorers']) *
                         len(first['modes']) * len(first['budgets']), 5616000)

    def test_each_user_has_six_price_variants_and_six_intact_controls(self):
        config = json.loads((ROOT/'configs/price-exp1/experiment1_musical.json').read_text())
        items = [{'id':str(i),'price':(i+1)*1000,'train_count':5} for i in range(8)]
        data = {'items':items,'histories':{f'u{i}':['0','1'] for i in range(5)},
                'test':[{'user':f'u{i}','positive':str(i)} for i in range(5)]}
        requests = build_requests(data, config)
        self.assertEqual(len(requests), 65)
        self.assertEqual(set(Counter(r['user'] for r in requests).values()), {13})
        lookup = {r['id']:r for r in requests}
        for user in data['histories']:
            prices = [r for r in requests if r['user']==user and r['family']=='price']
            self.assertEqual({(r['strength'],r['variant']) for r in prices},
                             {(s,v) for s in ('loose','medium','tight') for v in ('max','interval')})
        for row in requests:
            if row['family']=='random_matched':
                source = lookup[row['matched_request']]
                self.assertEqual(source['family'],'price')
                self.assertEqual(source['user'],row['user'])
                self.assertEqual(len(source['feasible']),len(row['feasible']))
        changed = copy.deepcopy(data)
        for row in changed['test']:
            row['positive']='outside'
        stable=lambda rows:[(r['id'],r['rule'],r['feasible']) for r in rows]
        self.assertEqual(stable(requests),stable(build_requests(changed,config)))
