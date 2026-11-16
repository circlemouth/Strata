"""Pure synthetic-log tests. No services, subprocesses, devices or GPU calls."""
import copy
import hashlib
import importlib.util
import pathlib
import unittest
from unittest.mock import patch
import collect_diagnostics as d

FP16='strata: prompt kernels: every chunk runs 48 of 48 layers on the FP16 path.'
def fused(floor=1024,small='48 on the staged fused entry'):
 return f'strata: prompt kernels: chunks of {floor} tokens or more run 48 of 48 layers on the fused kernels; shorter chunks run {small}.'


class PlanTests(unittest.TestCase):
 def test_import_does_not_read_log_or_print(self):
  spec=importlib.util.spec_from_file_location('isolated_collect_diagnostics',d.__file__)
  module=importlib.util.module_from_spec(spec)
  with patch.object(pathlib.Path,'read_bytes',side_effect=AssertionError('log IO')),patch.object(pathlib.Path,'exists',side_effect=AssertionError('log IO')),patch('builtins.print') as output:
   spec.loader.exec_module(module)
  output.assert_not_called()
 def test_every_chunk_fp16(self):
  p=d.parse_prompt_kernel_plan(FP16)
  self.assertEqual(p['big'],dict(fused=0,staged=0,i8=0,fp16=48))
  self.assertEqual(p['small'],p['big']);self.assertIsNone(p['floor_tokens'])
  self.assertFalse(p['kernel_launch_trace']);self.assertEqual(p['line'],FP16)
  self.assertFalse(d.validate_prompt_kernel_plan(p,'C0')['ok'])
  self.assertFalse(d.validate_prompt_kernel_plan(p,'C1')['ok'])
 def test_both_planned_chunks_above_floor(self):
  p=d.parse_prompt_kernel_plan(fused())
  self.assertEqual(p['small'],dict(fused=0,staged=48,i8=0,fp16=0))
  for arm in ('C1','B1'):
   v=d.validate_prompt_kernel_plan(p,arm)
   self.assertTrue(v['ok']);self.assertEqual(v['applicable_fused_chunk_tokens'],[2048,1521])
   self.assertTrue(all(c['matches_expected'] is True for c in v['planned_chunks']))
 def test_below_floor_paths_recorded_without_overrejecting(self):
  for small in ('48 on the staged fused entry','48 on the FP16 path','48 on the int8 kernels','8 on the staged fused entry and 40 on the FP16 path'):
   with self.subTest(small=small):
    v=d.validate_prompt_kernel_plan(d.parse_prompt_kernel_plan(fused(1536,small)),'C1')
    self.assertTrue(v['ok']);self.assertEqual(v['applicable_fused_chunk_tokens'],[2048])
    self.assertEqual(v['planned_chunks'][1]['chunk_class'],'small')
    self.assertIsNone(v['planned_chunks'][1]['matches_expected'])
    self.assertEqual(v['planned_chunks'][1]['expectation'],'record_only_below_floor')
 def test_floor_boundary_and_no_long_chunk_coverage(self):
  p=d.parse_prompt_kernel_plan(fused(2048));v=d.validate_prompt_kernel_plan(p,'B1')
  self.assertEqual(v['planned_chunks'][0]['chunk_class'],'big')
  v=d.validate_prompt_kernel_plan(d.parse_prompt_kernel_plan(fused(3072)),'B1')
  self.assertTrue(v['ok']);self.assertEqual(v['applicable_fused_chunk_tokens'],[])
 def test_mixed_big_counts_are_parsed_but_fail_expected_arm(self):
  line='strata: prompt kernels: chunks of 1024 tokens or more run 20 of 48 layers on the fused kernels, 25 on the int8 kernels and 3 on the FP16 path; shorter chunks run 20 on the staged fused entry, 25 on the int8 kernels and 3 on the FP16 path.'
  p=d.parse_prompt_kernel_plan(line)
  self.assertEqual(p['big'],dict(fused=20,staged=0,i8=25,fp16=3))
  for arm in ('C0','C1','B1'):self.assertFalse(d.validate_prompt_kernel_plan(p,arm)['ok'])
 def test_c0_rejects_small_i8_even_when_planned_chunks_are_big(self):
  line='strata: prompt kernels: chunks of 1024 tokens or more run 48 of 48 layers on the FP16 path; shorter chunks run 48 on the int8 kernels.'
  self.assertFalse(d.validate_prompt_kernel_plan(d.parse_prompt_kernel_plan(line),'C0')['ok'])
 def test_peer_suffix_and_all_i8(self):
  line='strata: prompt kernels: every chunk runs 40 of 48 layers on the int8 kernels and 8 on the FP16 path (--peer-device: the fused kernels are off).'
  p=d.parse_prompt_kernel_plan(line);self.assertTrue(p['peer_fused_off'])
  self.assertFalse(d.validate_prompt_kernel_plan(p,'C0')['ok'])
  p=d.parse_prompt_kernel_plan('strata: prompt kernels: every chunk runs 48 of 48 layers on the int8 kernels.')
  self.assertEqual(p['big']['i8'],48)
 def test_malformed_counts_totals_duplicates_and_private_text_rejected(self):
  lines=[FP16.replace('48 of 48','47 of 48'),FP16.replace('48 of 48','48 of 49'),
   FP16.replace('48 of 48 layers','48'),FP16.replace('48 of 48','0 of 48'),FP16[:-1],
   FP16+' /private/prompt',FP16.replace('FP16 path','FP16 path and 1 on the FP16 path'),
   fused().replace('48 on the staged','49 on the staged'),fused().replace('chunks of 1024','chunks of 0')]
  for line in lines:
   with self.subTest(line=line):
    with self.assertRaises(ValueError):d.parse_prompt_kernel_plan(line)
 def test_missing_tampered_and_wrong_layer_evidence_fails(self):
  self.assertFalse(d.validate_prompt_kernel_plan(None,'C0')['ok'])
  p=d.parse_prompt_kernel_plan(FP16);p['big']['fp16']=47
  self.assertFalse(d.validate_prompt_kernel_plan(p,'C0')['ok'])
  p=d.parse_prompt_kernel_plan(FP16.replace('48','40'))
  self.assertEqual(d.validate_prompt_kernel_plan(p,'C0')['reason'],'unexpected_layer_count')
 def test_invalid_validation_arguments(self):
  with self.assertRaises(ValueError):d.validate_prompt_kernel_plan(None,'A0')
  for chunks in ((),(True,),(0,),(-1,)):
   with self.assertRaises(ValueError):d.validate_prompt_kernel_plan(None,'C0',chunks)


class CollectorTests(unittest.TestCase):
 def test_retains_plan_and_existing_diagnostics(self):
  raw=(FP16+'\nstrata prefill timing: total 100 ms\nprivate prompt not retained\n').encode()
  got=d.collect_diagnostics(raw)
  self.assertEqual(got['selected_engine_lines'],[FP16,'strata prefill timing: total 100 ms'])
  self.assertEqual(got['prompt_kernel_plan_status'],'observed');self.assertEqual(got['engine_log_sha256'],hashlib.sha256(raw).hexdigest())
  self.assertEqual(got['engine_log_bytes'],len(raw));self.assertFalse(got['raw_log_exported'])
 def test_missing_duplicate_and_conflicting_one_shot_plan(self):
  got=d.collect_diagnostics(None)
  self.assertEqual(got['prompt_kernel_plan_status'],'missing');self.assertIsNone(got['engine_log_sha256'])
  got=d.collect_diagnostics((FP16+'\n'+FP16).encode())
  self.assertEqual(got['prompt_kernel_plan_line_count'],2);self.assertEqual(got['prompt_kernel_plan_status'],'observed')
  got=d.collect_diagnostics((FP16+'\n'+fused()).encode())
  self.assertEqual(got['prompt_kernel_plan_status'],'conflicting');self.assertIsNone(got['prompt_kernel_plan'])
 def test_invalid_plan_does_not_export_raw_bad_line(self):
  got=d.collect_diagnostics((FP16+'\n'+d.PLAN_PREFIX+'/private/person').encode())
  self.assertEqual(got['prompt_kernel_plan_status'],'invalid');self.assertIsNone(got['prompt_kernel_plan'])
  self.assertEqual(got['selected_engine_lines'],[FP16])
 def test_failure_scan_covers_filtered_lines(self):
  got=d.collect_diagnostics(b'private /path: HIP error nan\nGPU reset\n')
  self.assertEqual(got['failure_indications'],['nonfinite','gpu-runtime-error'])
  self.assertEqual(got['selected_engine_lines'],[])
 def test_ordinary_tuning_line_and_exact_known_path_only(self):
  ordinary='prefill gemm: hipBLASLt tuning enabled (59 rows, gfx1151, version 100401)'
  known=f'prefill gemm: hipBLASLt tuning enabled (59 rows, {d.TUNING_PATH}, version 100401)'
  got=d.collect_diagnostics((ordinary+'\n'+known).encode())
  self.assertEqual(got['selected_engine_lines'],[ordinary,known.replace(d.TUNING_PATH,'gfx1151-hipblaslt-100401.txt')])
 def test_arbitrary_or_disguised_paths_stay_private(self):
  for path in ('/private/table.txt',d.TUNING_PATH+'.extra',d.TUNING_PATH+'/../secret','/prefix'+d.TUNING_PATH,'C:\\private\\table.txt'):
   line=f'prefill gemm: hipBLASLt tuning enabled (59 rows, {path}, version 100401)'
   self.assertEqual(d.collect_diagnostics(line.encode())['selected_engine_lines'],[])
  line='hipBLASLt tuning fallback: '+d.TUNING_PATH
  self.assertEqual(d.collect_diagnostics(line.encode())['selected_engine_lines'],[])
 def test_existing_output_bounds_preserved(self):
  self.assertEqual(d.collect_diagnostics(('strata prefill: '+'x'*1600).encode())['selected_engine_lines'],[])
  with self.assertRaisesRegex(RuntimeError,'diagnostic_bound'):d.collect_diagnostics(('strata prefill: ok\n'*3001).encode())
 def test_engine_start_markers_are_count_only_and_only_from_engine_log(self):
  marker='[strata] 2026-11-16 12:31:00 engine started: /private/engine --serve --pack /private/model'
  malformed='[strata] date engine started: /private/engine'
  got=d.collect_diagnostics((marker+'\n'+malformed+'\n'+marker).encode(),marker.encode())
  self.assertEqual(got['engine_start_count'],2)
  self.assertEqual(got['engine_start_count_kind'],'pre_exec_start_attempt_markers')
  self.assertEqual(got['selected_engine_lines'],[])
  self.assertNotIn('/private',str(got))
  self.assertEqual(got['lifecycle_logs_present'],dict(engine=True,server=True))
 def test_narrow_restart_and_fallback_indications_from_both_logs(self):
  examples={
   'starting_it_again':'[strata] the engine exited before it wrote anything to its log; starting it again in 2 s (try 2 of 3)',
   'engine_had_stopped':'[strata] the engine had stopped (exit code 1); starting it again (a minute or two) ...',
   'engine_start_retry':'[strata] the engine did not start (try 1 of 3); again in 15 s',
   'drafter_fallback':'[strata] 2026-11-16 12:32:00 the engine ended before it was ready; starting it again without a drafter',
   'split_engine_fallback':'[strata] The all-cards split engine could not start (hipBLASLt tuning failed). Starting the normal layer split instead.',
   'replica_fallback':'[strata] replica(s) 1 did not start; starting again on the shared expert arena'}
  for key,line in examples.items():
   for engine,server in ((line.encode(),b''),(b'',line.encode())):
    with self.subTest(key=key,server=bool(server)):
     got=d.collect_diagnostics(engine,server)
     self.assertTrue(got['restart_fallback_indications'][key]);self.assertTrue(got['restart_or_fallback_indicated'])
     self.assertNotIn('/private',str(got));self.assertEqual(got['selected_engine_lines'],[])
 def test_legitimate_fp16_and_fused_fallback_not_restart(self):
  engine=(FP16+'\nstrata prefill: fused fallback to FP16\nprefill gemm: tuning unavailable; using hipBLASEx\n').encode()
  server=b'[strata] normal engine path\nuser said: starting it again\nsplit engine is supported\n'
  got=d.collect_diagnostics(engine,server)
  self.assertFalse(got['restart_or_fallback_indicated'])
  self.assertTrue(all(value is False for value in got['restart_fallback_indications'].values()))
 def test_missing_server_log_is_explicit_not_false_evidence_of_coverage(self):
  got=d.collect_diagnostics(b'')
  self.assertEqual(got['lifecycle_logs_present'],dict(engine=True,server=False))
  self.assertIsNone(got['server_log_sha256']);self.assertEqual(got['server_log_bytes'],0)




I8='strata: prompt kernels: every chunk runs 48 of 48 layers on the int8 kernels.'
def combined(arm='C0'):
 plan=d.parse_prompt_kernel_plan(I8 if arm=='C0' else fused())
 diagnostic=dict(native_fused_branch_entry_count=0 if arm=='C0' else 1,native_fused_branch_entry_kind=d.NATIVE_FUSED_BRANCH_KIND,prompt_kernel_plan=plan,prompt_kernel_plan_status='observed',prompt_kernel_plan_line_count=1,
                 prompt_kernel_plan_parse_errors=0,engine_start_count=1,engine_start_count_kind='pre_exec_start_attempt_markers',restart_or_fallback_indicated=False,
                 lifecycle_logs_present={'engine':True,'server':True},failure_indications=[])
 build=dict(d.SOURCES,build_options=dict(d.BUILD_OPTIONS),prefill_compile_definitions={side:dict(d.PREFILL_DEFINITIONS) for side in ('B','C')},rule_sha256={'B':dict(d.RULE_SHA256),'C':dict(d.RULE_SHA256)},
            binary_sha256={'B1':'b'*64,'C0':'c'*64,'C1':'c'*64})
 effective=dict(binary_sha256=build['binary_sha256'][arm],native_argv_matches_config=True,native_drafter_options_absent=True,
                pid=123,start_ticks='456',environment={'STRATA_PF_FUSED':'0' if arm=='C0' else '1','STRATA_PF_FUSED_NATIVE':'1',
                'STRATA_PF_F16':'0','STRATA_PF_DM16':'0','STRATA_PF_WMMA':'0'},
                observed_selector_names=list(d.SELECTORS+d.DIAGNOSTICS),absent_selector_names=['STRATA_EXPERT_I8',*d.DIAGNOSTICS])
 return diagnostic,effective,arm,build

class CombinedTests(unittest.TestCase):
 def verdict(self,parts):return d.validate_combined_dispatch_evidence(*parts)
 def test_all_arms(self):
  for arm in ('C0','C1','B1'):
   out=self.verdict(combined(arm));self.assertTrue(out['ok'],out)
   self.assertEqual(out['dispatch_family'],'own_i8' if arm=='C0' else 'native_fused')
   self.assertFalse(out['actual_kernel_launches_proven']);self.assertFalse(out['kernel_launch_trace'])
   self.assertIn('group size',out['limitations'][1])
 def test_int8_cannot_mean_native_fused(self):
  parts=combined('C1');parts[0]['prompt_kernel_plan']=d.parse_prompt_kernel_plan(I8)
  self.assertFalse(self.verdict(parts)['ok'])
 def test_fp16_cannot_mean_c0(self):
  parts=combined();parts[0]['prompt_kernel_plan']=d.parse_prompt_kernel_plan(FP16)
  self.assertFalse(self.verdict(parts)['ok'])
 def test_unknown_or_unobserved_i8(self):
  for mode in ('missing','unobserved','unknown'):
   parts=combined();eff=parts[1]
   if mode=='unobserved':eff['observed_selector_names'].remove('STRATA_EXPERT_I8')
   else:
    eff['absent_selector_names'].remove('STRATA_EXPERT_I8')
    if mode=='unknown':eff['environment']['STRATA_EXPERT_I8']='auto'
   self.assertFalse(self.verdict(parts)['ok'])
 def test_explicit_i8_values(self):
  for value,wanted in [('1',True),('0',False)]:
   parts=combined();parts[1]['absent_selector_names'].remove('STRATA_EXPERT_I8');parts[1]['environment']['STRATA_EXPERT_I8']=value
   self.assertEqual(self.verdict(parts)['ok'],wanted)
 def test_fused_i8_off_with_no_i8_plan_allowed(self):
  parts=combined('C1');parts[1]['absent_selector_names'].remove('STRATA_EXPERT_I8');parts[1]['environment']['STRATA_EXPERT_I8']='0'
  self.assertTrue(self.verdict(parts)['ok'])
  parts[0]['prompt_kernel_plan']=d.parse_prompt_kernel_plan(fused(small='48 on the int8 kernels'))
  self.assertFalse(self.verdict(parts)['ok'])
 def test_missing_required_flag(self):
  for key in d.SELECTORS[:-1]:
   parts=combined();parts[1]['environment'].pop(key)
   self.assertFalse(self.verdict(parts)['ok'])
 def test_binary_or_source_mismatch(self):
  for key in ('B_source','C_source'):
   parts=combined();parts[3][key]='a'*40;self.assertFalse(self.verdict(parts)['ok'])
  parts=combined();parts[1]['binary_sha256']='a'*64;self.assertFalse(self.verdict(parts)['ok'])
 def test_compile_and_rule_pins(self):
  for key in d.BUILD_OPTIONS:
   parts=combined();parts[3]['build_options'].pop(key);self.assertFalse(self.verdict(parts)['ok'])
  parts=combined();parts[3]['rule_sha256']['B'][next(iter(d.RULE_SHA256))]='a'*64
  self.assertFalse(self.verdict(parts)['ok'])
 def test_actual_prefill_compile_definitions_required(self):
  for side in ('B','C'):
   for key in d.PREFILL_DEFINITIONS:
    parts=combined();parts[3]['prefill_compile_definitions'][side].pop(key)
    self.assertEqual(self.verdict(parts)['reason'],'prefill_compile_definitions_not_proven')
  parts=combined();parts[3]['prefill_compile_definitions']['B']['STRATA_PREFILL_MMQ']='1'
  self.assertFalse(self.verdict(parts)['ok'])
 def test_diagnostics_presence_and_absence_evidence(self):
  for key in d.DIAGNOSTICS:
   parts=combined();parts[1]['absent_selector_names'].remove(key);parts[1]['environment'][key]='0'
   self.assertFalse(self.verdict(parts)['ok'])
 def test_lifecycle(self):
  for key,value in [('engine_start_count_kind',None),('engine_start_count_kind','kernel_launches'),('engine_start_count',2),('engine_start_count',True),('restart_or_fallback_indicated',True),('restart_or_fallback_indicated',None),('lifecycle_logs_present',{'engine':True,'server':False})]:
   parts=combined();parts[0][key]=value;self.assertFalse(self.verdict(parts)['ok'])
 def test_plan_conflict_or_missing_scope(self):
  for key,value in [('prompt_kernel_plan_status','conflicting'),('prompt_kernel_plan_line_count',2),('prompt_kernel_plan_parse_errors',1)]:
   parts=combined();parts[0][key]=value;self.assertFalse(self.verdict(parts)['ok'])
  for key in ('kernel_launch_trace','evidence_kind'):
   parts=combined();parts[0]['prompt_kernel_plan'].pop(key);self.assertFalse(self.verdict(parts)['ok'])
 def test_tampered_counts(self):
  parts=combined();parts[0]['prompt_kernel_plan']['big']['i8']=47
  self.assertFalse(self.verdict(parts)['ok'])
 def test_below_floor_record_only(self):
  parts=combined('C1');parts[3]['requested_chunks']=[85,2048,1521]
  out=self.verdict(parts);self.assertTrue(out['ok'])
  self.assertIsNone(out['plan_validation']['planned_chunks'][0]['matches_expected'])
  self.assertEqual(out['plan_validation']['applicable_fused_chunk_tokens'],[2048,1521])
 def test_impossible_every_chunk_fused_wording_rejected(self):
  parts=combined('C1');parts[0]['prompt_kernel_plan']=d.parse_prompt_kernel_plan('strata: prompt kernels: every chunk runs 48 of 48 layers on the fused kernels.')
  self.assertEqual(self.verdict(parts)['reason'],'source_plan_chunk_shape_mismatch')
 def test_historic_allowlist_not_absence(self):
  parts=combined();parts[1].pop('observed_selector_names');parts[1].pop('absent_selector_names')
  self.assertEqual(self.verdict(parts)['reason'],'selector_observation_missing')
 def test_native_identity_missing(self):
  for key in ('pid','start_ticks','native_argv_matches_config','native_drafter_options_absent'):
   parts=combined();parts[1].pop(key);self.assertFalse(self.verdict(parts)['ok'])
 def test_exact_native_branch_capture(self):
  line=d.NATIVE_FUSED_BRANCH_LINE
  out=d.collect_diagnostics((line+'\n'+line+'\n').encode())
  self.assertEqual(out['native_fused_branch_entry_count'],2)
  self.assertEqual(out['native_fused_branch_entry_kind'],'native_fused_branch_entry')
  for bad in [line+'.',line+' /private',line.replace('#136','#137')]:
   self.assertEqual(d.collect_diagnostics(bad.encode())['native_fused_branch_entry_count'],0)
 def test_same_int8_word_not_own_family(self):
  parts=combined();parts[0]['native_fused_branch_entry_count']=1
  self.assertFalse(self.verdict(parts)['ok'])
 def test_fused_long_requires_branch_entry(self):
  parts=combined('C1');parts[0]['native_fused_branch_entry_count']=0
  self.assertEqual(self.verdict(parts)['reason'],'branch_entry_contradicts_or_missing_for_arm')
  for arm in ('C1','B1'):
   out=self.verdict(combined(arm));self.assertTrue(out['branch_entry_observed'])
   self.assertTrue(out['branch_entry_required']);self.assertFalse(out['actual_kernel_launches_proven'])
 def test_below_floor_only_zero_marker_allowed(self):
  parts=combined('C1');parts[3]['requested_chunks']=[85];parts[0]['native_fused_branch_entry_count']=0
  out=self.verdict(parts);self.assertTrue(out['ok']);self.assertFalse(out['branch_entry_required'])
  self.assertFalse(out['branch_entry_observed']);self.assertFalse(out['actual_kernel_launches_proven'])
 def test_missing_unknown_branch_scope_rejects(self):
  for key,value in [('native_fused_branch_entry_count',None),('native_fused_branch_entry_count',True),('native_fused_branch_entry_kind','kernel_trace')]:
   parts=combined();parts[0][key]=value;self.assertEqual(self.verdict(parts)['reason'],'branch_entry_evidence_missing')
 @unittest.skipUnless(all((pathlib.Path(__file__).resolve().parent.parent/('source-'+a)).is_dir() for a in ('B','C')), 'Archived source trees are not in this public package; checked in sealed campaign')
 def test_pinned_rules_match_archived_sources_read_only(self):
  root=pathlib.Path(__file__).resolve().parent.parent
  for arm in ('B','C'):
   for relative,wanted in d.RULE_SHA256.items():
    self.assertEqual(hashlib.sha256((root/('source-'+arm)/relative).read_bytes()).hexdigest(),wanted)

if __name__=='__main__':unittest.main()
