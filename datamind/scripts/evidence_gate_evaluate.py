"""Offline annotated audit evaluation. Labels never enter runtime retrieval."""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def ratio(n, d):
    return n/d if d else None

def reliability(pairs):
    if not pairs:
        return {"n":0,"brier":None,"ece":None}
    bins=defaultdict(list)
    for p,y in pairs:
        bins[min(int(p*10),9)].append((p,y))
    return {"n":len(pairs), "brier":mean((p-y)**2 for p,y in pairs),
        "ece":sum(len(v)*abs(mean(p for p,y in v)-mean(y for p,y in v)) for v in bins.values())/len(pairs)}

def evaluate(logs, annotations):
    labels={row['request_id']:row for row in annotations}
    if len(labels)!=len(annotations):
        raise ValueError('Duplicate annotation request_id')
    samples=[]
    protocols=defaultdict(int)
    verdicts=defaultdict(int)
    execution=defaultdict(int)
    rejection_reasons=defaultdict(int)
    multiclass=[]
    total_atoms=0
    scoring_errors=0
    scoring_unavailable=0
    scores=defaultdict(list)
    latencies=defaultdict(list)
    tokens=defaultdict(list)
    stability=defaultdict(list)
    citation_valid=0
    citation_total=0
    unknown=[]
    for log in logs:
        bench=log.get('datamind_v2')
        key=log.get('request_id') or (bench.get('query_id') if bench else None)
        if key is None:
            raise ValueError('Missing request_id/query_id')
        if key not in labels:
            unknown.append(key)
            continue
        label=labels[key]
        benchmark_result = None
        if bench:
            benchmark_result={'decision':{'COMPLETED':'answer','REJECTED':'insufficient','ERROR':'error'}.get(bench.get('state'),'error'),
                'answer':(bench.get('answer') or {}).get('answer'),
                'atoms':(bench.get('evidence_graph') or {}).get('atoms',[]), 'audit':bench.get('audit') or {},
                'reason_codes':(bench.get('decision') or {}).get('reason_code') or []}
        result=log.get('result') or benchmark_result or {'decision':log.get('decision','error'),'answer':None,
            'atoms':(log.get('draft') or {}).get('atoms',[]),
            'audit':{'citation_checks':log.get('checks',[]),'atom_reports':log.get('reports',[]),'timings_ms':log.get('timings',{})}}
        audit=result['audit']
        released=result['decision']=='answer'
        protocol=log.get('protocol_version') or (bench.get('protocol_version') if bench else None) or ('abc-v1' if any(r.get('score_type')=='raw_abc_expected_value' for r in audit.get('atom_reports',[])) else 'ab-legacy')
        protocols[protocol]+=1
        for code in set(result.get('reason_codes', log.get('reason_codes', []))):
            rejection_reasons[code]+=1
        supported=label['answer_supported']
        correct=label.get('answer_correct')
        if type(supported) is not bool or (correct is not None and type(correct) is not bool):
            raise ValueError('Annotations must use boolean labels')
        samples.append((label,released,supported,correct))
        for check in audit.get('citation_checks',[]):
            citation_total+=1
            citation_valid+=int(check['valid'])
        for report in audit.get('atom_reports',[]):
            total_atoms+=1
            execution[report.get('execution_status',report.get('status','unknown'))]+=1
            if report.get('execution_status')=='scored':
                verdicts[report['verdict']]+=1
            scoring_errors+=int(report.get('execution_status',report.get('status'))=='error')
            scoring_unavailable+=int('SCORING_UNAVAILABLE' in report.get('reason_codes',[]))
            semantic_label=label.get('atom_verdict',{}).get(report['atom_id'])
            if semantic_label is not None:
                names={'support':'A','uncertain':'B','refute':'C'}
                if semantic_label not in names:
                    raise ValueError('atom_verdict must be support/uncertain/refute')
                probs=report.get('label_probs')
                if probs is not None:
                    if set(probs)!={'A','B','C'} or any(not 0<=v<=1 for v in probs.values()) or abs(sum(probs.values())-1)>1e-9:
                        raise ValueError('Invalid ABC probabilities')
                    multiclass.append((probs,names[semantic_label],report.get('verdict')==semantic_label))
            atom_label=label.get('atom_supported',{}).get(report['atom_id'])
            if atom_label is not None:
                if type(atom_label) is not bool:
                    raise ValueError('atom_supported values must be booleans')
                for score_name in ('p_raw','p_cal'):
                    if report.get(score_name) is not None:
                        scores[score_name].append((report[score_name],int(atom_label)))
                if report.get('label_probs'):
                    scores['p_support'].append((report['label_probs']['A'],int(atom_label)))
            if report.get('elapsed_ms') is not None:
                atom=next(a for a in result['atoms'] if a['atom_id']==report['atom_id'])
                latencies['final_audit' if atom['kind']=='answer' else 'local_audit'].append(report['elapsed_ms'])
            if report.get('scores'):
                usage=report['scores'].get('usage',{})
                tokens['audit_input'].append(usage.get('prompt_tokens',0))
                tokens['audit_output'].append(usage.get('completion_tokens',0))
        for stage,elapsed in audit.get('timings_ms',{}).items():
            latencies[stage].append(elapsed)
        usage=log.get('retrieval_usage',{})
        for field in ('input_tokens','output_tokens'):
            if field in usage:
                tokens['retrieval_'+field].append(usage[field])
        stages = ('answer_generation', 'atom_building') if ('answer_generation_usage' in log
            or 'atom_building_usage' in log) else ('reasoning',)
        for stage in stages:
            for usage in log.get(stage+'_usage',[]):
                for field in ('input_tokens','output_tokens'):
                    if field in usage:
                        tokens[stage+'_'+field].append(usage[field])
        if label.get('perturbation_group'):
            final=next((r for r in audit.get('atom_reports',[]) if r['atom_id']==next((a['atom_id'] for a in result['atoms'] if a['kind']=='answer'),None)),None)
            stability[label['perturbation_group']].append((released,result['answer'], final.get('expected_score',final.get('p_raw')) if final else None))
    count=len(samples)
    accepted=[x for x in samples if x[1]]
    unsupported=[x for x in samples if not x[2]]
    supported=[x for x in samples if x[2]]
    accuracy_labeled=[x for x in accepted if x[3] is not None]
    categories={}
    for kind in ('insufficient_evidence','invalid_citation','invalid_derivation','incomplete_answer','retrieval_failure'):
        subset=[x for x in samples if x[0].get('failure_type')==kind]
        categories[kind]={'n':len(subset),'false_release_rate':ratio(sum(x[1] for x in subset),len(subset))}
    perturbations=[]
    for key,values in stability.items():
        probabilities=[p for _,_,p in values if p is not None]
        perturbations.append({'group':key,'n':len(values),'decisions_agree':len({x[0] for x in values})==1,
            'released_answers_agree':len({json.dumps(x[1],sort_keys=True,ensure_ascii=False) for x in values if x[0]})<=1,
            'raw_score_range':max(probabilities)-min(probabilities) if probabilities else None})
    return {'protocol_counts':dict(protocols),'atom_verdict_counts':dict(verdicts),
        'execution_status_counts':dict(execution),'rejection_reason_counts':dict(rejection_reasons),
        'scoring_error_rate':ratio(scoring_errors,total_atoms),'scoring_unavailable_rate':ratio(scoring_unavailable,total_atoms),
        'abc_classification':{'n':len(multiclass),
            'accuracy':mean(x[2] for x in multiclass) if multiclass else None,
            'brier':mean(sum((p[k]-int(k==y))**2 for k in ('A','B','C')) for p,y,_ in multiclass) if multiclass else None},
        'answer_accuracy_labeled':ratio(sum(x[3] for x in samples if x[3] is not None),sum(x[3] is not None for x in samples)),
        'published_error_rate':ratio(sum(not x[3] for x in accuracy_labeled),len(accuracy_labeled)),
        'annotated_n':count,'unannotated_request_ids':unknown,'coverage':ratio(len(accepted),count),
        'false_release_rate':ratio(sum(x[1] for x in unsupported),len(unsupported)),
        'supported_rejection_rate':ratio(sum(not x[1] for x in supported),len(supported)),
        'released_accuracy_labeled_n':len(accuracy_labeled),
        'released_accuracy':ratio(sum(x[3] for x in accuracy_labeled),len(accuracy_labeled)),
        'citation_validity':ratio(citation_valid,citation_total),'categories':categories,
        'reliability':{key:reliability(values) for key,values in scores.items()},
        'latency_ms_mean':{key:mean(values) for key,values in latencies.items()},
        'token_means':{key:mean(values) for key,values in tokens.items()},'perturbation_stability':perturbations}

def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',required=True)
    parser.add_argument('--annotations',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    result=evaluate(read_jsonl(args.audit),read_jsonl(args.annotations))
    Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__': main()
