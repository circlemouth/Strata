"""Read-only exact native process evidence; importing this module performs no IO."""
import pathlib,os,json,hashlib,sys

DIAGNOSTICS=('STRATA_PREFILL_TIMING','STRATA_HIPBLASLT_VERBOSE','STRATA_PF_FUSED_TRACE','STRATA_MANIFEST','STRATA_I8_COUNT')
ALLOWED=('STRATA_PF_FUSED','STRATA_PF_FUSED_NATIVE','STRATA_PF_F16','STRATA_PF_DM16','STRATA_HC_Q8','STRATA_ROUTE_TAIL_SKIP','STRATA_PF_WMMA','STRATA_EXPERT_I8')
NUMERIC=('--expert-cache','--resident-budget-gib','--max-context','--kv-resident','--prefill','--prompt-cache','--conversation-cache-mib','--conversation-cache-slots','--spec','--adapt-every','--batch')

def validate_environment(env):
    if any(k.encode() in env for k in DIAGNOSTICS):raise ValueError('diagnostic_environment_present')
    selected={k:env[k.encode()].decode('ascii') for k in ALLOWED if k.encode() in env}
    if any(v not in ('0','1') for v in selected.values()):raise ValueError('environment_value_not_allowlisted')
    return selected

def validate_native_argv(raw,cfg):
    """Require unchanged configured pairs, permitting only known server additions.

    Return no paths or raw arguments. The hash includes exact NUL-delimited bytes.
    --spec remains prompt lookup; --draft/--mtp options are forbidden entirely.
    """
    if not isinstance(raw,bytes) or not raw.endswith(b'\0') or len(raw)>65536:
        raise ValueError('native_argv_shape')
    argv=[x.decode('utf-8',errors='strict') for x in raw[:-1].split(b'\0')]
    if not argv or argv[0]!=cfg['exe']:raise ValueError('native_argv_executable')
    required=cfg.get('args')
    if cfg.get('drafting')!='off' or not isinstance(required,list) or len(required)%2:
        raise ValueError('config_argv_shape')
    expected={}
    for i in range(0,len(required),2):
        flag,value=required[i:i+2]
        if not isinstance(flag,str) or not flag.startswith('--') or flag in expected or not isinstance(value,str) or value.startswith('--'):
            raise ValueError('config_argv_pairs')
        expected[flag]=value
    if any(x.startswith(('--draft','--mtp','--spec-min-p','--batch-mtp')) for x in expected):
        raise ValueError('draft_options_forbidden')
    # engine_args adds vision and parallel_args adds batch; effort_position=start
    # adds no tail-role token. Linux has no implicit draft-costs option.
    if cfg.get('effort_position','start')!='start' or cfg.get('parallel')!=2:
        raise ValueError('unsupported_server_additions')
    extras={'--serve':None,'--batch':'2'}
    if isinstance(cfg.get('vision'),dict):extras['--vision']=None
    if set(expected)&set(extras):raise ValueError('duplicate_config_server_option')
    allowed={**expected,**extras};seen={};required_seen=[];i=1
    while i<len(argv):
        flag=argv[i]
        if flag not in allowed or flag in seen:raise ValueError('unexpected_or_duplicate_native_option')
        value=allowed[flag];i+=1
        if value is not None:
            if i>=len(argv) or argv[i]!=value:raise ValueError('native_argument_changed')
            i+=1
        seen[flag]=value
        if flag in expected:required_seen.extend([flag,value])
    if set(seen)!=set(allowed) or required_seen!=required:raise ValueError('missing_or_reordered_native_arguments')
    numeric={}
    for k in NUMERIC:
        if k in seen:
            if not seen[k].isascii() or not seen[k].isdecimal():raise ValueError('numeric_selector_shape')
            numeric[k]=int(seen[k])
    return {'native_argv_sha256':hashlib.sha256(raw).hexdigest(),
            'native_numeric_selectors':numeric,'native_argv_matches_config':True,
            'native_drafter_options_absent':True,'native_expected_server_flags_verified':True}

def collect(identity_only=False):
    if type(identity_only) is not bool:raise ValueError("identity_mode_shape")
    cfg=json.loads(pathlib.Path('/control/config.json').read_text());exe=pathlib.Path(cfg['exe']).resolve()
    found=[]
    for p in pathlib.Path('/proc').iterdir():
     if not p.name.isdecimal():continue
     try:
      if (p/'exe').resolve()!=exe:continue
      before=(p/'stat').read_text().rsplit(')',1)[1].split()[19]
      env=dict(x.split(b'=',1) for x in (p/'environ').read_bytes().split(b'\0') if b'=' in x)
      selected=validate_environment(env)
      observed=list(ALLOWED)+list(DIAGNOSTICS)
      absent=[k for k in observed if k.encode() not in env]
      argv_proof=validate_native_argv((p/'cmdline').read_bytes(),cfg)
      binary_sha256=None if identity_only else hashlib.sha256((p/'exe').read_bytes()).hexdigest()
      libs={}
      if not identity_only:
       for line in (p/'maps').read_text().splitlines():
        bits=line.split()
        if len(bits)<6:continue
        f=pathlib.Path(bits[-1])
        if f.name in ('libhipblas.so.3','librocblas.so.5','libhipblaslt.so.1','libamdhip64.so.7') or f.name.startswith(('libhipblas.so.3.','librocblas.so.5.','libhipblaslt.so.1.','libamdhip64.so.7.')):
         if f.name not in libs:libs[f.name]={'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
      assert before==(p/'stat').read_text().rsplit(')',1)[1].split()[19]
      found.append({'pid':int(p.name),'start_ticks':before,'environment':selected,'observed_selector_names':observed,'absent_selector_names':absent,'binary_sha256':binary_sha256,'mapped_libraries':libs,'mapped_libraries_hashed':not identity_only,**argv_proof,'diagnostic_variables_absent':list(DIAGNOSTICS),'environment_observation_scope':'initial exec environment from /proc; dynamically added own-only defaults are not claimed observable here'})
     except (FileNotFoundError,ProcessLookupError):continue
    if len(found)!=1:raise ValueError("native_process_count")
    return found[0]

def main(argv=None):
    argv=sys.argv[1:] if argv is None else argv
    if argv not in ([],['--identity-only']):raise ValueError('unsupported_collector_option')
    print(json.dumps(collect(identity_only=bool(argv))),flush=True)

if __name__=="__main__":main()
