"""Select raw ABC Gate policies on disjoint validation materials, never fit AB Platt.

Input: one row per audited question, with material_group, request_id,
answer_correct (boolean), integrity_valid (boolean), final_atom_id, atom_reports.
Each scored report carries true ABC logprobs under scores.logprobs. Offline
annotations must never be forwarded to runtime evidence/context.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from datamind.core.protocols import LabelScores
from datamind.agent.evidence_gated.scoring import normalize_abc, classify
from datamind.scripts.evidence_gate_evaluate import read_jsonl


def fit(rows, *, max_risk=.05, min_accepted=20, seed=42, require_final_support=False):
    if not 0 <= max_risk <= 1 or min_accepted < 1:
        raise ValueError('Invalid policy selection parameters')
    if len({r.get('material_group') for r in rows}) < 5:
        raise ValueError('Need at least five disjoint material groups')
    parsed=[]
    for row in rows:
        if 'logprob_a' in row or 'supported_label' in row:
            raise ValueError('ABC_CONFIG_MIGRATION_REQUIRED: AB calibration cannot supply C probabilities')
        if (not isinstance(row.get('material_group'),str) or not row['material_group']
                or type(row.get('answer_correct')) is not bool or type(row.get('integrity_valid')) is not bool):
            raise ValueError('Require nonempty material_group and boolean answer_correct/integrity_valid')
        reports=row.get('atom_reports',[])
        ids=[r['atom_id'] for r in reports]
        if len(set(ids))!=len(ids):
            raise ValueError('Duplicate audit atom')
        scores={}
        complete=bool(reports) and row.get('final_atom_id') in ids and row['integrity_valid']
        for report in reports:
            if report.get('execution_status')!='scored':
                complete=False
                continue
            dto=LabelScores.model_validate(report['scores'])
            p=normalize_abc(dto.logprobs)
            scores[report['atom_id']]=p['A']-p['C']
        parsed.append((complete,scores))
    groups=[r['material_group'] for r in rows]
    train,rest=next(GroupShuffleSplit(n_splits=1,test_size=.4,random_state=seed).split(rows,groups=groups))
    val_idx,test_idx=next(GroupShuffleSplit(n_splits=1,test_size=.5,random_state=seed+1).split(rest,groups=[groups[i] for i in rest]))
    validation,test=rest[val_idx],rest[test_idx]

    def allows(index, policy):
        complete,scores=parsed[index]
        if not complete:
            return False
        verdicts={k:classify(v,policy['atom_support_threshold'],policy['atom_refute_threshold']) for k,v in scores.items()}
        return (all(v!='refute' for v in verdicts.values())
            and sum(v=='support' for v in verdicts.values())/len(verdicts)>=policy['min_support_ratio']
            and (not policy['require_final_support'] or verdicts[rows[index]['final_atom_id']]=='support'))

    def metrics(indices, policy):
        accepted=[i for i in indices if policy and allows(i,policy)]
        return {'n':len(indices),'accepted':len(accepted),'coverage':len(accepted)/len(indices) if len(indices) else None,
            'published_error_rate':sum(not rows[i]['answer_correct'] for i in accepted)/len(accepted) if accepted else None}

    candidates=[]
    for refute,support,rho in itertools.product((-.5,-.3,-.1),(.1,.3,.5),(.6,.8,1.0)):
        policy={'scoring_mode':'abc','atom_refute_threshold':refute,'atom_support_threshold':support,
                'min_support_ratio':rho,'require_final_support':require_final_support}
        result=metrics(validation,policy)
        if result['accepted']>=min_accepted and result['published_error_rate']<=max_risk:
            candidates.append((result,policy))
    best=max(candidates,key=lambda x:(x[0]['accepted'],-x[0]['published_error_rate'],x[1]['min_support_ratio'],x[1]['atom_support_threshold'])) if candidates else None
    policy=best[1] if best else None
    return {'protocol_version':'abc-v1','selection_version':'abc-raw-'+hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()[:16],
        'selected_config':policy,'validation':best[0] if best else metrics(validation,None),
        'reason':None if best else 'INSUFFICIENT_VALIDATION_COVERAGE','heldout':metrics(test,policy),'seed':seed,
        'split_material_groups':{name:sorted({groups[i] for i in indices}) for name,indices in [('train',train),('validation',validation),('test',test)]},
        'warning':'Raw ABC scores are not calibrated correctness probabilities; train split reserved, policy selected on validation only. Empirical risk is not a future-risk guarantee.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--max-risk',type=float,default=.05)
    parser.add_argument('--min-accepted',type=int,default=20)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--require-final-support',action='store_true')
    args=parser.parse_args()
    result=fit(read_jsonl(args.input),max_risk=args.max_risk,min_accepted=args.min_accepted,seed=args.seed,
               require_final_support=args.require_final_support)
    Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__': main()
