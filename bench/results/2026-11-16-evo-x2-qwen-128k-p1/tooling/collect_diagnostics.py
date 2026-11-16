"""Allowlisted diagnostics and runtime-selected plans, never kernel launch traces.

Importing and the parse/validation helpers perform no IO. The one-shot plan may
arrive only when the first prompt initializes prefill; validate the cumulative
log after the first completed request, not every request's new log lines.
"""
import json, pathlib, re, hashlib
prefixes=('strata fused dispatch:', 'strata: prompt experts on the fused prompt-expert path',
          'strata prefill timing:', 'strata hipBLASLt', 'hipBLASLt tuning',
          'strata: hipBLASLt', 'strata generate: HIP runtime', 'strata HIP',
          'strata prefill: ', 'strata prompt: ', 'strata batch: ')
PLAN_PREFIX='strata: prompt kernels: '
NATIVE_FUSED_BRANCH_LINE='strata: prompt experts on the fused int8 kernels (STRATA_PF_FUSED=1, #136)'
NATIVE_FUSED_BRANCH_KIND='native_fused_branch_entry'
TUNING_PATH='/opt/strata/native/tuning/gfx1151-hipblaslt-100401.txt'
KINDS={'fused kernels':'fused','staged fused entry':'staged','int8 kernels':'i8','FP16 path':'fp16'}
ENGINE_START=re.compile(r'^\[strata\] [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2} engine started: .+')


def _split_counts(text, total=None):
 parts=re.split(r', | and ',text)
 counts=dict.fromkeys(('fused','staged','i8','fp16'),0)
 rendered=[];order=[]
 for index,part in enumerate(parts):
  match=re.fullmatch(r'([1-9][0-9]*)(?: of ([1-9][0-9]*) layers)? on the (fused kernels|staged fused entry|int8 kernels|FP16 path)',part)
  if not match:raise ValueError('plan_count_syntax')
  count,declared,label=match.groups();kind=KINDS[label]
  if total is None and index==0:
   if declared is None:raise ValueError('plan_total_missing')
   total=int(declared)
  elif declared is not None:raise ValueError('plan_repeated_total')
  if counts[kind]:raise ValueError('plan_duplicate_path')
  counts[kind]=int(count);order.append(('fused','staged','i8','fp16').index(kind));rendered.append(part)
 # Match the source formatter's order/separators, not free-form log text.
 joined=''.join(('' if i==0 else ' and ' if i==len(rendered)-1 else ', ')+part for i,part in enumerate(rendered))
 if joined!=text or order!=sorted(order) or (counts['fused'] and counts['staged']):raise ValueError('plan_path_order')
 if sum(counts.values())!=total:raise ValueError('plan_layer_count_mismatch')
 return total,counts


def parse_prompt_kernel_plan(line):
 """Parse only own_only_line.hpp's exact sentence; no hardware claim."""
 if not line.startswith(PLAN_PREFIX):return None
 if len(line)>1600 or '/' in line or '\\' in line:raise ValueError('plan_line_shape')
 text=line[len(PLAN_PREFIX):]
 peer=' (--peer-device: the fused kernels are off).'
 peer_off=text.endswith(peer)
 if peer_off:text=text[:-len(peer)]+'.'
 if not text.endswith('.'):raise ValueError('plan_sentence_shape')
 text=text[:-1]
 if text.startswith('every chunk runs '):
  layers,big=_split_counts(text[len('every chunk runs '):]);small=dict(big);floor=None;scope='every_chunk'
 else:
  match=re.fullmatch(r'chunks of ([1-9][0-9]*) tokens or more run (.+); shorter chunks run (.+)',text)
  if not match:raise ValueError('plan_sentence_shape')
  floor,large,short=match.groups();floor=int(floor);scope='size_split'
  layers,big=_split_counts(large);_,small=_split_counts(short,layers)
 if big['staged']:raise ValueError('plan_large_staged_path')
 return dict(evidence_kind='runtime_selected_dispatch_plan',kernel_launch_trace=False,
             line=line,scope=scope,layers_total=layers,floor_tokens=floor,
             big=big,small=small,peer_fused_off=peer_off)


def validate_prompt_kernel_plan(plan, arm, chunks=(2048,1521), expected_layers=48):
 """Validate the printed MMQ-OFF plan; classify planned, not observed, chunks.

 C0 must be entirely own INT8 at every size. C1/B1 must be entirely fused in the
 long class; below-floor staged/int8/FP16 choices are recorded, not rejected.
 Call after first request completion. Missing/ambiguous evidence never passes.
 """
 if arm not in ('C0','C1','B1'):raise ValueError('unknown_arm')
 if type(expected_layers) is not int or expected_layers<=0:raise ValueError('invalid_expected_layers')
 if not chunks or any(type(n) is not int or n<=0 for n in chunks):raise ValueError('invalid_planned_chunks')
 result=dict(ok=False,arm=arm,evidence_kind='runtime_selected_dispatch_plan',kernel_launch_trace=False,
             planned_chunks=[],reason='plan_not_observed',applicable_fused_chunk_tokens=[])
 if plan is None:return result
 try:
  parsed=parse_prompt_kernel_plan(plan['line'])
  if parsed is None or parsed!=plan:raise ValueError('plan_shape')
 except (KeyError,TypeError,ValueError):result['reason']='invalid_plan';return result
 result.update(floor_tokens=plan['floor_tokens'],layers_total=plan['layers_total'])
 if plan['layers_total']!=expected_layers:result['reason']='unexpected_layer_count';return result
 def all_on(counts,kind):return counts[kind]==expected_layers and sum(counts.values())==expected_layers
 ok=(all_on(plan['big'],'i8') and all_on(plan['small'],'i8')) if arm=='C0' else all_on(plan['big'],'fused')
 for tokens in chunks:
  below=plan['floor_tokens'] is not None and tokens<plan['floor_tokens']
  counts=plan['small'] if below else plan['big']
  expectation='all_own_i8' if arm=='C0' else 'record_only_below_floor' if below else 'all_fused'
  if expectation=='all_fused':result['applicable_fused_chunk_tokens'].append(tokens)
  result['planned_chunks'].append(dict(tokens=tokens,chunk_class='small' if below else 'big',
      selected_layer_counts=dict(counts),expectation=expectation,
      matches_expected=None if expectation=='record_only_below_floor' else all_on(counts,'i8' if arm=='C0' else 'fused')))
 result.update(ok=ok,reason='matched' if ok else 'arm_dispatch_plan_mismatch')
 return result


# Frozen source rules, not measurements: both archived source trees have these bytes.
# prefill.cpp:204-215 sets the MMQ-free default before main; i8_expert_rule.hpp:28-56
# resolves unset I8 and orders fused/staged > own INT8 > FP16. iq_kernels.cu:6240-6307
# applies that rule. own_only_line.hpp:24-26,56-69 formats the selected plan.
RULE_SHA256={
 'include/strata/kernels/i8_expert_rule.hpp':'ac4886cba2b2383cb8bcc35cc0c556a7a5c2b66c06bbaa5ccf16b188e280515b',
 'include/strata/prefill/own_only_line.hpp':'63ed30fe36fe0a37db943bc080a1a0bdd8432747d343491cfbd6f440d5f534aa',
 'src/prefill/prefill.cpp':'2260c05c83090bb5faee90633c83c8d6e763dc7b98da55fe93a010dccec6a3ab',
 'src/kernels/cuda/iq_kernels.cu':'b3d3bd6d24633068f16c0ad481c4677945645a417719a98d0be7e2d0b061ed6e'}
SOURCES={'B_source':'3af562d541ec76c8da63d568248a919c8a54906b',
         'C_source':'4d754ceba2d8bf54cbd1094e00352c27478f6198'}
BUILD_OPTIONS={'STRATA_MMQ':'OFF','STRATA_PREFILL_MMQ':'OFF','STRATA_NATIVE_EXPERTS':'ON',
               'STRATA_ENABLE_HIP':'ON','CMAKE_HIP_ARCHITECTURES':'gfx1151'}
PREFILL_DEFINITIONS={'STRATA_PREFILL_FUSED':'1','STRATA_NATIVE_EXPERTS':'1','STRATA_USE_HIP':'1','STRATA_PREFILL_MMQ':None}
SELECTORS=('STRATA_PF_FUSED','STRATA_PF_FUSED_NATIVE','STRATA_PF_F16','STRATA_PF_DM16','STRATA_PF_WMMA','STRATA_EXPERT_I8')
DIAGNOSTICS=('STRATA_PREFILL_TIMING','STRATA_HIPBLASLT_VERBOSE','STRATA_PF_FUSED_TRACE','STRATA_MANIFEST','STRATA_I8_COUNT')
LIVE_LIMITATIONS=('A selected dispatch plan is not a kernel launch trace.',
                 'Live dispatch also depends on group size, buffers, native layout and sharing conditions.',
                 'Below-floor paths are recorded separately; no fused launch is inferred for them.',
                 'A fused branch-entry marker precedes kernel launches and does not prove all 48 layers or calls.',
                 'An absent selector is evidence only when its name was explicitly observed.')

def validate_combined_dispatch_evidence(diagnostics,effective,arm,build):
 """Pure, fail-closed evidence join. Never reads source, environment or devices.

 build binds commits, compile options, both rule hash maps and arm binary hashes;
 requested_chunks optionally gives positive planned sizes (default 2048,1521).
 effective must carry explicit observed/absent selector lists from the same native
 process plus its actual binary hash. Missing historical I8 observation fails.
 Return a structured verdict, not an exception or an actual-kernel assertion.
 """
 result=dict(ok=False,arm=arm,reason='invalid_evidence',dispatch_family=None,
             evidence_kind='runtime_selected_dispatch_plan',kernel_launch_trace=False,
             actual_kernel_launches_proven=False,limitations=list(LIVE_LIMITATIONS))
 def require(condition,reason):
  if not condition:raise ValueError(reason)
 try:
  require(arm in ('C0','C1','B1'),'unknown_arm')
  require(all(type(x) is dict for x in (diagnostics,effective,build)),'evidence_shape')
  require(all(build.get(k)==v for k,v in SOURCES.items()),'source_pair_mismatch')
  options=build.get('build_options',{})
  require(type(options) is dict and all(options.get(k)==v for k,v in BUILD_OPTIONS.items()),'MMQ_free_build_not_proven')
  definitions=build.get('prefill_compile_definitions',{})
  require(type(definitions) is dict and all(type(definitions.get(side)) is dict and all(k in definitions[side] and definitions[side][k]==v for k,v in PREFILL_DEFINITIONS.items()) for side in ('B','C')),'prefill_compile_definitions_not_proven')
  rules=build.get('rule_sha256',{})
  require(type(rules) is dict and rules.get('B')==RULE_SHA256 and rules.get('C')==RULE_SHA256,'source_rules_not_pinned_identical')
  binaries=build.get('binary_sha256',{})
  require(type(binaries) is dict and all(type(binaries.get(k)) is str and re.fullmatch('[0-9a-f]{64}',binaries[k]) for k in ('B1','C0','C1')),'binary_provenance_missing')
  require(binaries['C0']==binaries['C1'] and effective.get('binary_sha256')==binaries[arm],'actual_binary_mismatch')
  require(effective.get('native_argv_matches_config') is True and effective.get('native_drafter_options_absent') is True,'native_arguments_not_verified')
  require(type(effective.get('pid')) is int and effective['pid']>0 and type(effective.get('start_ticks')) is str and effective['start_ticks'].isdigit(),'native_generation_missing')
  env=effective.get('environment');observed=effective.get('observed_selector_names');absent=effective.get('absent_selector_names')
  require(type(env) is dict and type(observed) is list and type(absent) is list,'selector_observation_missing')
  require(all(type(k) is str for k in observed+absent) and len(set(observed))==len(observed) and len(set(absent))==len(absent),'selector_list_shape')
  require(set(SELECTORS+DIAGNOSTICS).issubset(observed) and set(absent).issubset(observed),'selector_observation_incomplete')
  require(not set(absent).intersection(env),'selector_presence_contradiction')
  require(all(k in env or k in absent for k in observed),'selector_observation_ambiguous')
  require(all(k in absent and k not in env for k in DIAGNOSTICS),'diagnostic_absence_not_proven')
  expected={'STRATA_PF_FUSED':'0' if arm=='C0' else '1','STRATA_PF_FUSED_NATIVE':'1',
            'STRATA_PF_F16':'0','STRATA_PF_DM16':'0','STRATA_PF_WMMA':'0'}
  require(all(type(env.get(k)) is str and env[k]==v for k,v in expected.items()),'arm_selectors_mismatch')
  i8=env.get('STRATA_EXPERT_I8')
  require(i8 in ('0','1') if 'STRATA_EXPERT_I8' in env else 'STRATA_EXPERT_I8' in absent,'expert_i8_unknown')
  require(type(diagnostics.get('engine_start_count')) is int and diagnostics['engine_start_count']==1,'native_start_count')
  require(diagnostics.get('engine_start_count_kind')=='pre_exec_start_attempt_markers','native_start_marker_scope')
  require(diagnostics.get('restart_or_fallback_indicated') is False,'retry_or_fallback_unproved')
  logs=diagnostics.get('lifecycle_logs_present',{})
  require(type(logs) is dict and set(logs)=={'engine','server'} and all(v is True for v in logs.values()),'lifecycle_logs_missing')
  require(diagnostics.get('failure_indications')==[],'native_failures_or_missing_scan')
  require(diagnostics.get('prompt_kernel_plan_status')=='observed' and type(diagnostics.get('prompt_kernel_plan_line_count')) is int and diagnostics['prompt_kernel_plan_line_count']==1 and diagnostics.get('prompt_kernel_plan_parse_errors')==0,'one_selected_plan_not_proven')
  plan=diagnostics.get('prompt_kernel_plan')
  require(type(plan) is dict and plan.get('evidence_kind')=='runtime_selected_dispatch_plan' and plan.get('kernel_launch_trace') is False,'plan_scope_missing_or_incorrect')
  parsed=parse_prompt_kernel_plan(plan.get('line',''))
  require(json.dumps(parsed,sort_keys=True)==json.dumps(plan,sort_keys=True),'plan_structure_mismatch')
  require(plan.get('peer_fused_off') is False,'peer_override_not_in_scope')
  require((plan['scope']=='every_chunk' and plan['floor_tokens'] is None) if arm=='C0' else (plan['scope']=='size_split' and type(plan['floor_tokens']) is int and plan['floor_tokens']>0 and plan['small']['fused']==0),'source_plan_chunk_shape_mismatch')
  if i8=='0':require(not plan['big']['i8'] and not plan['small']['i8'],'explicit_i8_off_contradicts_plan')
  chunks=build.get('requested_chunks',(2048,1521))
  require(type(chunks) in (tuple,list) and len(chunks)>0 and all(type(n) is int and n>0 for n in chunks),'requested_chunks_invalid')
  validation=validate_prompt_kernel_plan(plan,arm,chunks=chunks)
  require(validation['ok'],'arm_dispatch_plan_mismatch')
  count=diagnostics.get('native_fused_branch_entry_count')
  require(type(count) is int and count>=0 and diagnostics.get('native_fused_branch_entry_kind')==NATIVE_FUSED_BRANCH_KIND,'branch_entry_evidence_missing')
  branch_required=arm!='C0' and bool(validation['applicable_fused_chunk_tokens'])
  require(count==0 if arm=='C0' else count>=1 if branch_required else True,'branch_entry_contradicts_or_missing_for_arm')
  result.update(branch_entry_observed=count>0,branch_entry_count=count,branch_entry_kind=NATIVE_FUSED_BRANCH_KIND,
                branch_entry_required=branch_required,branch_entry_scope='cumulative process log; not per-layer or per-request launch proof')
  result.update(ok=True,reason='matched',dispatch_family='own_i8' if arm=='C0' else 'native_fused',
                expert_i8_setting='unset_with_proven_MMQ_free_default' if i8 is None else i8,
                plan_validation=validation,source_rules_pinned_identical=True,actual_binary_bound=True)
 except (ValueError,TypeError,KeyError) as exc:
  result['reason']=str(exc) if isinstance(exc,ValueError) else 'malformed_evidence'
 return result


def _retained_line(line):
 # Only this exact known table may lose its directory. Any remaining path is
 # rejected, including suffixes, parent traversal, and unrelated tuning paths.
 if line.startswith('prefill gemm: hipBLASLt tuning enabled ('):
  line=re.sub(r'(?<![\w./\\-])'+re.escape(TUNING_PATH)+r'(?![\w./\\-])',TUNING_PATH.rsplit('/',1)[1],line)
 if '/' in line or '\\' in line or len(line)>1600:return None
 return line


def lifecycle_indications(engine_lines,server_lines):
 """Counts/booleans only. The start marker records intent before native exec."""
 lines=[line for line in (*engine_lines,*server_lines) if line.startswith('[strata] ')]
 indications=dict(
  starting_it_again=any('starting it again' in line for line in lines),
  engine_had_stopped=any('the engine had stopped' in line for line in lines),
  engine_start_retry=any(re.match(r'^\[strata\] the engine did not start \(try [0-9]+ of [0-9]+\); again in ',line) for line in lines),
  drafter_fallback=any('starting again without a drafter' in line or 'starting again with the MTP' in line
                      or 'starting it again without a drafter' in line or 'starting it again with the MTP' in line for line in lines),
  split_engine_fallback=any(line.startswith('[strata] The all-cards split engine could not start (')
                            and '). Starting the normal layer split instead.' in line for line in lines),
  replica_fallback=any(line.startswith('[strata] replica(s) ') and ' did not start; starting again on the shared expert arena' in line for line in lines))
 return dict(engine_start_count=sum(bool(ENGINE_START.match(line)) for line in engine_lines),
             engine_start_count_kind='pre_exec_start_attempt_markers',restart_fallback_indications=indications,
             restart_or_fallback_indicated=any(indications.values()))


def collect_diagnostics(raw,server_raw=None):
 """Pure collector. Failure scan still covers every line, even rejected ones."""
 lines=raw.decode(errors='replace').splitlines() if raw is not None else []
 server_lines=server_raw.decode(errors='replace').splitlines() if server_raw is not None else []
 selected=[];plans=[];parse_errors=0
 for line in lines:
  if line.startswith('[strata] '):continue  # lifecycle text is counts/booleans only
  if line==NATIVE_FUSED_BRANCH_LINE:
   selected.append(line);continue
  if line.startswith(PLAN_PREFIX):
   try:plan=parse_prompt_kernel_plan(line)
   except ValueError:parse_errors+=1;continue
   plans.append(plan);selected.append(line)
  elif line.startswith(prefixes) or re.search(r'hip[Bb][Ll][Aa][Ss][Ll][Tt].*(tuning|solution|fallback|launch|shape)',line):
   retained=_retained_line(line)
   if retained is not None:selected.append(retained)
 if len(selected)>3000:raise RuntimeError('diagnostic_bound')
 failures=[]
 for name,pattern in [('nonfinite',r'(?i)(?:\bnan\b|\b[+-]?inf(?:inity)?\b)'),('gpu-runtime-error',r'(?i)(?:HIP error|hipError|GPU reset|GPU fault|out of memory|allocation failed|fatal error)')]:
  if any(re.search(pattern,line) for line in lines):failures.append(name)
 status='invalid' if parse_errors else 'missing' if not plans else 'observed' if all(p==plans[0] for p in plans) else 'conflicting'
 return dict(native_fused_branch_entry_count=sum(line==NATIVE_FUSED_BRANCH_LINE for line in lines),
             native_fused_branch_entry_kind=NATIVE_FUSED_BRANCH_KIND,
             failure_indications=failures,engine_log_bytes=len(raw) if raw is not None else 0,
             engine_log_sha256=hashlib.sha256(raw).hexdigest() if raw is not None else None,
             server_log_bytes=len(server_raw) if server_raw is not None else 0,
             server_log_sha256=hashlib.sha256(server_raw).hexdigest() if server_raw is not None else None,
             lifecycle_logs_present=dict(engine=raw is not None,server=server_raw is not None),
             selected_engine_lines=selected,raw_log_exported=False,
             prompt_kernel_plan=plans[0] if status=='observed' else None,
             prompt_kernel_plan_status=status,prompt_kernel_plan_line_count=len(plans),
             prompt_kernel_plan_parse_errors=parse_errors,**lifecycle_indications(lines,server_lines))


def main():
 p=pathlib.Path('/tmp/runtime/engine.log')
 server=pathlib.Path('/tmp/runtime/server.log')
 print(json.dumps(collect_diagnostics(p.read_bytes() if p.exists() else None,
                                    server.read_bytes() if server.exists() else None)),flush=True)


if __name__=='__main__':main()
