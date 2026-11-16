"""Pure argument/environment validation and static experiment-config checks."""
import ast
import copy
import hashlib
import json
import pathlib
import unittest
from unittest.mock import patch,Mock
import collect_effective as ce
from collect_effective import validate_native_argv,validate_environment,DIAGNOSTICS
R=pathlib.Path(__file__).resolve().parent

def config():return json.loads((R/'C1.config.json').read_text())
def args(c=None):
    c=c or config()
    return [c['exe'],'--serve',*c['args'],'--vision','--batch','2']
def wire(a):return ('\0'.join(a)+'\0').encode()
class EffectiveTests(unittest.TestCase):
    def test_valid_numeric_only(self):
        a=args();out=validate_native_argv(wire(a),config())
        self.assertEqual(out['native_numeric_selectors']['--spec'],4)
        self.assertEqual(out['native_numeric_selectors']['--batch'],2)
        self.assertEqual(out['native_argv_sha256'],hashlib.sha256(wire(a)).hexdigest())
        self.assertNotIn('/models',json.dumps(out));self.assertNotIn('/opt',json.dumps(out))
    def test_changed_numeric(self):
        for flag in ['--spec','--max-context','--batch','--prefill','--expert-cache']:
            a=args();a[a.index(flag)+1]='1'
            with self.assertRaises(ValueError):validate_native_argv(wire(a),config())
    def test_duplicates(self):
        for tail in [['--spec','4'],['--batch','2'],['--vision']]:
            with self.assertRaises(ValueError):validate_native_argv(wire(args()+tail),config())
    def test_forbid_drafter(self):
        for tail in [['--draft','dflash2'],['--mtp','4'],['--batch-mtp'],['--draft-costs','/tmp/table']]:
            with self.assertRaises(ValueError):validate_native_argv(wire(args()+tail),config())
    def test_reject_equals_override(self):
        with self.assertRaises(ValueError):validate_native_argv(wire(args()+['--spec=1']),config())
    def test_missing_pair(self):
        a=args();i=a.index('--spec');del a[i:i+2]
        with self.assertRaises(ValueError):validate_native_argv(wire(a),config())
    def test_path_change(self):
        a=args();a[a.index('--pack')+1]='/other'
        with self.assertRaises(ValueError):validate_native_argv(wire(a),config())
    def test_reordered_config_pairs(self):
        a=args();a[2:6]=a[4:6]+a[2:4]
        with self.assertRaises(ValueError):validate_native_argv(wire(a),config())
    def test_server_additions_can_interleave(self):
        c=config();a=[c['exe'],'--vision','--serve','--batch','2',*c['args']]
        self.assertTrue(validate_native_argv(wire(a),c)['native_argv_matches_config'])
    def test_raw_shape(self):
        for raw in [b'',b'notnul',wire(args())+b'\0']:
            with self.assertRaises(ValueError):validate_native_argv(raw,config())
    def test_diagnostics_absent_even_zero(self):
        for key in DIAGNOSTICS:
            for value in [b'0',b'',b'/tmp/private']:
                with self.assertRaises(ValueError):validate_environment({key.encode():value})
    def test_env_sanitized(self):
        self.assertEqual(validate_environment({b'HOME':b'/private',b'STRATA_PF_FUSED':b'1'}),{'STRATA_PF_FUSED':'1'})
    def test_configs_matched_except_fused(self):
        cs=[json.loads((R/(arm+'.config.json')).read_text()) for arm in ['B1','C0','C1']]
        self.assertEqual([c['env']['STRATA_PF_FUSED'] for c in cs],['1','0','1'])
        for c in cs:
            self.assertEqual(c['drafting'],'off');self.assertEqual(c['args'][c['args'].index('--spec')+1],'4')
            self.assertNotIn('STRATA_START_RETRIES',c['env'])
            self.assertTrue(set(DIAGNOSTICS).isdisjoint(c['env']))
            c['env'].pop('STRATA_PF_FUSED')
        self.assertEqual(cs[0],cs[1]);self.assertEqual(cs[1],cs[2])
    @unittest.skipUnless((R/'launch_server.py').is_file(), 'Private service launcher intentionally not published; checked in sealed campaign')
    def test_launch_sets_server_retries(self):
        tree=ast.parse((R/'launch_server.py').read_text())
        call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='execve')
        environment=ast.literal_eval(call.args[2]);self.assertEqual(environment['STRATA_START_RETRIES'],'0')
        self.assertTrue(set(DIAGNOSTICS).isdisjoint(environment))
    def test_identity_mode_skips_maps_and_hashing(self):
        cfg=config();nodes={};reads=[]
        class Node:
            def __init__(self,name):self.value=name;self.name=name.rsplit('/',1)[-1]
            def __truediv__(self,suffix):return node(self.value+'/'+suffix)
            def resolve(self):return node(cfg['exe']) if self.value=='/proc/123/exe' else self
            def iterdir(self):return iter([node('/proc/123')])
            def read_text(self):
                reads.append(self.value)
                if self.value=='/control/config.json':return json.dumps(cfg)
                if self.value=='/proc/123/stat':return '123 (native) '+' '.join(['S']+['0']*18+['456'])
                raise AssertionError('unexpected file read: '+self.value)
            def read_bytes(self):
                reads.append(self.value)
                if self.value=='/proc/123/environ':return b'STRATA_PF_FUSED=1\0'
                if self.value=='/proc/123/cmdline':return wire(args(cfg))
                raise AssertionError('unexpected byte read: '+self.value)
        def node(name):
            if name not in nodes:nodes[name]=Node(name)
            return nodes[name]
        with patch.object(ce.pathlib,'Path',side_effect=node):out=ce.collect(identity_only=True)
        self.assertEqual((out['pid'],out['start_ticks']),(123,'456'))
        self.assertFalse(out['mapped_libraries_hashed']);self.assertEqual(out['mapped_libraries'],{})
        self.assertEqual(reads.count('/proc/123/stat'),2)
        self.assertNotIn('/proc/123/maps',reads)
        self.assertTrue(out['native_argv_matches_config'])
        self.assertIn('STRATA_EXPERT_I8',out['observed_selector_names'])
        self.assertIn('STRATA_EXPERT_I8',out['absent_selector_names'])
        self.assertIsNone(out['binary_sha256'])
        self.assertTrue(set(DIAGNOSTICS)<=set(out['observed_selector_names']))
        self.assertTrue(set(DIAGNOSTICS)<=set(out['absent_selector_names']))
    def test_cli_modes(self):
        with patch.object(ce,'collect',return_value={}) as call,patch('builtins.print'):
            ce.main([]);call.assert_called_with(identity_only=False)
            ce.main(['--identity-only']);call.assert_called_with(identity_only=True)
            with self.assertRaises(ValueError):ce.main(['--identity-only','extra'])
    @unittest.skipUnless((R.parent/'source-C/serve/server.py').is_file(), 'Archived source tree is not in this public package; checked in sealed campaign')
    def test_source_contract_static(self):
        source=(R.parent/'source-C/serve/server.py').read_text()
        self.assertIn('"STRATA_START_RETRIES", "3"',source)
        self.assertIn('drafting.normalize(cfg.get("drafting")) != "auto"',source)
        self.assertIn('return ["--batch", str(n)]',source)
        self.assertIn('args.append("--vision")',source)
if __name__=='__main__':unittest.main()
