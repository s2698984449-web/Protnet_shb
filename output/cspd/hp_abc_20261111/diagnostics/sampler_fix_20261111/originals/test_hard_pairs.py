"""CPU-only, in-memory tests. No datasets, downloads, output writes or training."""
import argparse
import importlib.util
import json
import random
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cspd.hard_pairs import (HardPairBuffer, choose_classes, hard_probability,
                             metric_bucket, record_predictions, serial_metrics)
from cspd.data import CurriculumNewSplitEpisodes as Sampler
from cspd.runtime import build_model, output_dir


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


queue = load_script('run_p1_queue')
train = load_script('train_cspd')
summary = load_script('summarize_p1_abc')


def bucket(error=4, opp=20):
    return {'episodes': 2, 'per_class': {'a': [opp-error, opp], 'b': [opp-error, opp]},
            'confusion': {('a', 'b'): error, ('b', 'a'): error},
            'opportunities': {('a', 'b'): opp, ('b', 'a'): opp}}


def bare_sampler(classes=7, strict=True):
    s = Sampler.__new__(Sampler)
    s.rng = random.Random(4)
    s.hard_rng = random.Random(23)
    s.hard_pairs = []
    s.same_hard_pairs = {'S': [('c0', 'c1')]}
    s.hard_pair_enabled = True
    s.hard_probability = 1.0
    s.strict_groups = strict
    s.group_failures = {}
    s.n_way_cap = 5
    s.q_query = 2
    s.k_choices, s.k_probs = [2], [1]
    s.source_classes = {'S': ['c'+str(i) for i in range(classes)]}
    s.sources = ['S']
    s.by_class, s.group_of = {}, {}
    for name in s.source_classes['S']:
        pool = []
        for i in range(12):
            p = Path(name) / 'S/train' / ('p'+str(i)+'.png')
            pool.append((p, None))
            s.group_of[(name, 'S', p.stem)] = str(i//3)
        s.by_class[name] = {'S': pool}
    def fake_load(path, mask, source):
        return {'x': torch.zeros(3, 8, 8), 'y_cond': torch.zeros(6),
                'y_cond_mask': torch.ones(6), 'mask': torch.ones(1, 8, 8),
                'has_mask': True, 'path': str(path)}
    s._load = fake_load
    return s


class ScheduleTests(unittest.TestCase):
    def test_boundaries(self):
        expected = {1:0, 10:0, 11:.25, 50:.25, 51:.35, 100:.35, 101:.4, 1000:.4}
        for step, value in expected.items():
            self.assertEqual(hard_probability(step), value)

    def test_observed_request_rate(self):
        rng, hr = random.Random(9), random.Random(99)
        n = sum(choose_classes(list('abcdefg'),5,rng,hr,[('a','b')],.25)[1]['requested'] for _ in range(10000))
        self.assertLess(abs(n/10000-.25),.02)


class BufferTests(unittest.TestCase):
    def test_entry_exit_reentry(self):
        b = HardPairBuffer(['S'])
        b.update({'S': bucket()}); self.assertEqual(b.active['S'], [])
        b.update({'S': bucket()}); self.assertEqual(b.active['S'], [('a','b')])
        b.update({}); self.assertEqual(b.active['S'], [])
        b.update({'S': bucket()}); self.assertEqual(b.active['S'], [])

    def test_reverse_merged_once(self):
        b = HardPairBuffer(['S']); report=b.update({'S':bucket()})
        self.assertEqual(len(report['S']['top']),1)
        self.assertEqual(report['S']['top'][0]['errors'],8)

    def test_top3_rate_not_raw_count(self):
        m=metric_bucket()
        for i, (n,d) in enumerate([(100,10000),(3,10),(4,10),(5,10)]):
            a,b='a'+str(i),'b'+str(i)
            m['confusion'][(a,b)]=n
            m['opportunities'][(a,b)]=d
            m['opportunities'][(b,a)]=d
        report=HardPairBuffer(['S']).update({'S':m})
        pairs=[r['pair'] for r in report['S']['top']]
        self.assertEqual(len(pairs),3)
        self.assertNotIn(['a0','b0'],pairs)

    def test_holdout_rejected(self):
        with self.assertRaises(ValueError): HardPairBuffer(['S','H'], 'H')
        b=HardPairBuffer(['S'],'H')
        with self.assertRaises(ValueError): b.update({'H':bucket()})

    def test_source_local(self):
        b=HardPairBuffer(['S','T']); b.update({'S':bucket()}); b.update({'T':bucket()})
        self.assertEqual(b.active,{'S':[],'T':[]})

    def test_single_error_not_selected(self):
        m=bucket(0); m['confusion']={('a','b'):1}
        self.assertEqual(HardPairBuffer(['S']).update({'S':m})['S']['top'],[])


class CompositionTests(unittest.TestCase):
    def test_disabled_preserves_random_draw(self):
        expected=random.Random(1).sample(list('abcdefg'),5)
        chosen, info=choose_classes(list('abcdefg'),5,random.Random(1),random.Random(2),[],0)
        self.assertEqual(chosen,expected); self.assertFalse(info['requested'])

    def test_inject_keeps_way_and_random_pair(self):
        used=set(); changed=0
        rng,hr=random.Random(12),random.Random(4)
        for _ in range(100):
            chosen,info=choose_classes(list('abcdefgh'),5,rng,hr,[('a','b'),('c','d')],1)
            self.assertEqual(len(set(chosen)),5)
            self.assertTrue(set(info['pair']).issubset(chosen))
            used.add(tuple(info['pair'])); changed+=info['changed']
        self.assertEqual(len(used),2); self.assertGreater(changed,0)

    def test_saturation_no_composition_change(self):
        for n in [2,3,5]:
            pool=list('abcde')[:n]
            chosen,info=choose_classes(pool,n,random.Random(1),random.Random(2),[('a','b')],1)
            self.assertEqual(set(chosen),set(pool)); self.assertTrue(info['saturated'])
            self.assertTrue(info['applied']); self.assertFalse(info['changed'])

    def test_ineligible_falls_back(self):
        chosen,info=choose_classes(list('abc'),2,random.Random(1),random.Random(2),[('x','y')],1)
        self.assertTrue(info['requested']); self.assertTrue(info['fallback'])
        self.assertFalse(info['applied'])

    def test_same_source_episode_and_eval_no_injection(self):
        s=bare_sampler()
        e=s._sample_source('S',False,.1,True)
        self.assertTrue(e['hard_pair']['applied'])
        self.assertEqual(e['n_way'],5)
        self.assertEqual(e['support_x'].shape,(10,3,8,8))
        self.assertEqual(e['query_x'].shape,(10,3,8,8))
        self.assertIsNone(s._sample_source('S',False,.9,False)['hard_pair'])

    def test_saturated_full_sampler(self):
        e=bare_sampler(classes=3)._sample_source('S',False,.1,True)
        self.assertTrue(e['hard_pair']['saturated'])
        self.assertFalse(e['hard_pair']['changed'])


class GroupsTests(unittest.TestCase):
    def test_disjoint_groups_many_draws(self):
        s=bare_sampler(); pool=s.by_class['c0']['S']
        for _ in range(40):
            support,query=s._split_by_group(pool,3,3,'c0','S')
            gids=lambda items:{s.group_of[('c0','S',p.stem)] for p,_ in items}
            self.assertFalse(gids(support)&gids(query))

    def test_missing_or_single_group_fail_closed(self):
        s=bare_sampler(); s.group_of={}
        with self.assertRaisesRegex(RuntimeError,'strict groups unavailable'):
            s._sample_source('S',False,.2,True)
        s=bare_sampler(); s.group_of={key:'one' for key in s.group_of}
        with self.assertRaisesRegex(RuntimeError,'strict groups unavailable'):
            s._sample_source('S',False,.2,True)

    def test_legacy_group_fallback_opt_in_unchanged(self):
        s=bare_sampler(strict=False); s.group_of={}
        self.assertEqual(s._sample_source('S',False,.1,False)['n_way'],5)

    def test_constructor_excludes_holdout_even_query_only(self):
        root=ROOT/'tests'/'VIRTUAL_DATA_NOT_ON_DISK'
        mapping={root:[root/'a',root/'b']}
        for name in ['a','b']:
            c=root/name; mapping[c]=[c/'S',c/'H']
            for source in ['S','H']:
                directory=c/source
                mapping[directory]=[]
                mapping[directory/'val']=[directory/'val'/('p'+str(i)+'.png') for i in range(15)]
        reads=[]
        def read_text(path,**kw):
            reads.append(path)
            return json.dumps({'p'+str(i):str(i//3) for i in range(15)})
        cfg={'data_root':str(root),'holdout_source':'H','query_only':'H','cond_dim':6}
        with patch.object(Path,'is_dir',lambda p:p in mapping), patch.object(Path,'iterdir',lambda p:iter(mapping.get(p,[]))), patch.object(Path,'is_file',lambda p:True), patch.object(Path,'read_text',read_text):
            s=Sampler(cfg,'val',0)
        self.assertEqual(s.sources,['S'])
        self.assertFalse(any(p.parent.name=='H' for p in reads))


class MetricsTests(unittest.TestCase):
    def test_directed_denominators(self):
        m=metric_bucket();record_predictions(m,['a','b','c'],[0,0,1],[1,0,2])
        self.assertEqual(m['opportunities'][('a','b')],2)
        self.assertEqual(m['confusion'][('a','b')],1)
        self.assertEqual(m['per_class']['a'],[1,2])
        self.assertIn('a -> b',serial_metrics(m)['confusion'])

    def test_eval_records_per_source_and_opportunities(self):
        s=bare_sampler();s.sample=lambda **kw:s._sample_source('S',False,0,False)
        class Model:
            training=False
            def eval(self): pass
        def forward(model,episode,ways):
            logits=torch.zeros(len(episode['query_y']),ways)
            logits.scatter_(1,episode['query_y'][:,None],5.)
            return {'logits':logits}
        with patch.object(train,'forward_episode',forward):
            m=train.evaluate_curriculum(Model(),s,torch.device('cpu'),2,False)
        self.assertEqual(m['acc'],1);self.assertEqual(m['query_count'],20)
        self.assertEqual(m['by_source']['S']['episodes'],2)
        self.assertTrue(m['by_source']['S']['opportunities'])

    def test_pair_macro_and_missing_support(self):
        p=summary.pair_report(serial_metrics(bucket()),('a','b'))
        self.assertEqual(p['macro_recall'],.8)
        self.assertEqual(p['directed'][0]['rate'],.2)
        self.assertIsNone(summary.pair_report({},('a','b'))['macro_recall'])


class QueueTests(unittest.TestCase):
    def args(self,**kw):
        d=dict(suite=True,run_id='unit_test_no_write',config=None,experiments=None,seeds=None,folds=None)
        d.update(kw);return argparse.Namespace(**d)

    def test_default_21_and_matched_protocol(self):
        jobs=queue.build_jobs(self.args());self.assertEqual(len(jobs),21)
        self.assertEqual(len({name for name,_ in jobs}),21)
        a=next(c for n,c in jobs if n=='A_NEU_s0');b=next(c for n,c in jobs if n=='B_NEU_s0')
        ignore={'same_hard_pairs','exp_name','experiment_variant'}
        self.assertEqual({k:v for k,v in a.items() if k not in ignore},{k:v for k,v in b.items() if k not in ignore})
        self.assertTrue(a['strict_groups']);self.assertEqual(a['n_way'],5)
        self.assertIsNone(a['acceptance_neu_mde'])

    def test_select_single_fold_variant(self):
        jobs=queue.build_jobs(self.args(experiments=['B'],seeds=[1],folds=['GC10']))
        self.assertEqual([n for n,_ in jobs],['B_GC10_s1'])

    def test_dry_plan_does_not_create_files(self):
        with patch.object(Path,'mkdir',side_effect=AssertionError('write')),patch.object(Path,'write_text',side_effect=AssertionError('write')):
            plan=queue.plan_jobs(queue.build_jobs(self.args()))
        self.assertEqual(len(plan),21)

    def test_collision_refused_and_no_implicit_resume(self):
        jobs=queue.build_jobs(self.args(experiments=['A'],seeds=[0],folds=['NEU']))
        with patch.object(Path,'exists',return_value=True):
            with self.assertRaises(FileExistsError):queue.plan_jobs(jobs)
        with patch.object(Path,'exists',return_value=True),patch.object(Path,'is_file',return_value=False):
            with self.assertRaises(FileExistsError):queue.plan_jobs(jobs,True)

    def test_explicit_skip_checks_completion_config(self):
        jobs=queue.build_jobs(self.args(experiments=['A'],seeds=[0],folds=['NEU']))
        marker=json.dumps({'status':'complete','config':jobs[0][1]})
        out=ROOT/'output/cspd/unit_test_no_write/A_NEU_s0'
        with patch.object(Path,'exists',lambda p:p==out),patch.object(Path,'is_file',return_value=True),patch.object(Path,'read_text',return_value=marker):
            self.assertEqual(queue.plan_jobs(jobs,True)[0][3],'skip-completed')
        with patch.object(Path,'exists',lambda p:p==out),patch.object(Path,'is_file',return_value=True),patch.object(Path,'read_text',return_value='{}'):
            with self.assertRaises(ValueError):queue.plan_jobs(jobs,True)

    def test_no_output_overwrite(self):
        with patch.object(Path,'mkdir',side_effect=FileExistsError) as mkdir:
            with self.assertRaises(FileExistsError): output_dir({'exp_name':'new','output_group':'test'},ROOT,fresh=True)
            self.assertFalse(mkdir.call_args.kwargs['exist_ok'])

    def test_unsafe_run_id(self):
        with self.assertRaises(ValueError):queue.build_jobs(self.args(run_id='../elsewhere'))


class ModelTests(unittest.TestCase):
    def test_layers_shapes_and_checkpoint_keys(self):
        torch.set_num_threads(2)
        for layer,size,key,other in [(3,14,'proj3.weight','proj4.weight'),(4,7,'proj4.weight','proj3.weight')]:
            cfg={'train_mode':'curriculum','feature_layer':layer,'pretrained_backbone':False,
                 'freeze_backbone':True,'freeze_policy':True,'use_policy':False,'feat_dim':16}
            model=build_model(cfg).eval()
            with torch.no_grad():
                feature=model._maps(torch.zeros(1,3,224,224))
                self.assertEqual(tuple(feature.shape),(1,16,size,size))
                core,ring=model._core_ring(feature,torch.zeros(1,1,224,224))
                self.assertTrue(torch.isfinite(core).all());self.assertTrue(torch.isfinite(ring).all())
            state=model.state_dict();self.assertIn(key,state);self.assertNotIn(other,state)
            clone=build_model(cfg);clone.load_state_dict(state,strict=True)
            if layer==3:
                default=dict(cfg);default.pop('feature_layer');build_model(default).load_state_dict(state,strict=True)

    def test_ring_input_space_dilation_and_empty_core_fallback(self):
        import torch.nn.functional as F
        from cspd.model import _masked_mean
        for layer, size in [(3,14),(4,7)]:
            model=build_model({'train_mode':'curriculum','feature_layer':layer,
                               'pretrained_backbone':False,'feat_dim':1})
            mask=torch.zeros(1,1,224,224);mask[:,:,20,20]=1
            feat=torch.arange(size*size,dtype=torch.float32).reshape(1,1,size,size)
            original=F.interpolate(mask,size=(size,size),mode='nearest').squeeze(1)>=.5
            self.assertFalse(original.any())
            dilated=F.interpolate(F.max_pool2d(mask,33,1,16),size=(size,size),mode='nearest').squeeze(1)>=.5
            self.assertTrue(dilated.any())
            core,ring=model._core_ring(feat,mask)
            self.assertTrue(torch.equal(core,_masked_mean(feat,dilated)))
            self.assertTrue(torch.equal(ring,_masked_mean(feat,dilated)))
            self.assertNotEqual(float(core),float(feat.mean()))

    def test_invalid_feature_layer(self):
        with self.assertRaises(ValueError):build_model({'train_mode':'curriculum','feature_layer':2,'pretrained_backbone':False})


if __name__=='__main__':
    unittest.main(verbosity=2)
