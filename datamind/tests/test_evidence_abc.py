"""ABC numerical, integrity, dependency and publication acceptance tests.

Synthetic token probabilities exercise policy/flow only, not model accuracy.
"""
import asyncio
from copy import deepcopy
import json
import math
from uuid import uuid4
import pytest
from datamind.config import EvidenceGateConfig
from datamind.core.protocols import LabelScores
from datamind.core.model_clients import OpenAIChatCompletionsModelClient
from datamind.core.errors import ExternalServiceError, ConfigError
from datamind.agent.options import _probe_scoring
from datamind.agent.evidence_gated import AtomAudit, AnswerCandidate, EvidenceGroup, ReasoningDraft, AuditReport
from datamind.agent.evidence_gated.scoring import normalize_abc, classify, TokenScoringAdapter, ScoringUnavailable
from datamind.tests.test_evidence_atoms import agent, Client, body

A=(math.log(.85),math.log(.1),math.log(.05))
B=(math.log(.2),math.log(.75),math.log(.05))
C=(math.log(.05),math.log(.1),math.log(.85))


def scores(values=A, **updates):
    data=dict(logprobs=dict(zip(('A','B','C'),values)),model='fixture',call_id=uuid4().hex,
              tokens={'A':'A','B':'B','C':'C'})
    data.update(updates)
    return LabelScores(**data)


@pytest.mark.parametrize('distribution,expected', [((.85,.1,.05),.8),((.2,.75,.05),.15),((.45,.1,.45),0),((.05,.1,.85),-.8)])
def test_abc_weights_include_uncertainty(distribution,expected):
    p=normalize_abc(dict(zip(('A','B','C'),map(math.log,distribution))))
    assert sum(p.values())==pytest.approx(1)
    assert p['A']-p['C']==pytest.approx(expected)
    assert p['B']==pytest.approx(distribution[1])


def test_extreme_logprobs_stable_and_uncertainty_not_dropped():
    p=normalize_abc({'A':-9001.,'B':-9000.,'C':-9002.})
    assert sum(p.values())==pytest.approx(1)
    p=normalize_abc({'A':-50.,'B':-.001,'C':-55.})
    assert p['B']>0.999999 and classify(p['A']-p['C'],.3,-.3)=='uncertain'


@pytest.mark.parametrize('score,expected',[(-.31,'refute'),(-.3,'uncertain'),(0,'uncertain'),(.3,'uncertain'),(.31,'support')])
def test_thresholds_strict_with_inclusive_uncertain(score,expected):
    assert classify(score,.3,-.3)==expected


@pytest.mark.parametrize('updates',[
    {'logprobs':{'A':-.1,'B':-2.}},
    {'logprobs':{'A':-.1,'B':-2.,'C':float('nan')}},
    {'logprobs':{'A':-.1,'B':-2.,'C':float('inf')}},
    {'logprobs':{'A':-.1,'B':-2.,'C':.1}},
    {'logprobs':{'A':-.1,'B':-2.,'C':-9999.}},
    {'logprobs':{'A':-.1,'B':-2.,'C':-3.,'D':-4.}},
    {'tokens':{'A':'A','B':'B','C':' C'}},
    {'labels':['A','C','B']}, {'position':1}, {'call_id':''}, {'model':''},
])
def test_label_schema_rejects_invalid_abc_metadata(updates):
    with pytest.raises(ValueError): scores(**updates)


@pytest.mark.parametrize('payload',[
    body(c=float('nan')),body(c=float('inf')),body(c=.01),body(c=-9999.),
    {'model':'fixture','choices':[{'logprobs':{'content':[{'token':'A','logprob':-.1,'top_logprobs':[{'token':'B','logprob':-2.}]}]}}]},
    {'model':'fixture','choices':[{'logprobs':{'content':[{'token':'A','logprob':-.1,'top_logprobs':[{'token':'B','logprob':-2.},{'token':' C','logprob':-3.}]}]}}]},
    {'model':'fixture','choices':[{'logprobs':{'content':[{'token':'A','logprob':-.1,'top_logprobs':[{'token':'B','logprob':-2.}]},{'token':'C','logprob':-3.}]}}]},
])
def test_wire_missing_or_invalid_c_is_error(payload):
    with pytest.raises(ExternalServiceError):
        OpenAIChatCompletionsModelClient._parse_score_labels(payload,['A','B','C'])


def test_ab_probe_cannot_pass_abc_capability_check():
    class Client:
        async def score_labels(self,prompt):
            return LabelScores.model_construct(logprobs={'A':-.1,'B':-2.},labels=['A','B'],tokens={'A':'A','B':'B'},model='fixture',call_id='s')
    with pytest.raises(ConfigError): asyncio.run(_probe_scoring(Client()))


def test_adapter_revalidates_mutated_scores():
    class Client:
        async def score_labels(self,prompt,**kwargs):
            dto=scores(); dto.position=1
            return dto
    with pytest.raises(ScoringUnavailable):
        asyncio.run(TokenScoringAdapter(Client(),model='fixture').score_labels('q'))


@pytest.mark.parametrize('values',[
    {'atom_support_threshold':float('inf')}, {'atom_refute_threshold':float('nan')},
    {'atom_support_threshold':1.1}, {'atom_refute_threshold':-1.1},
    {'atom_support_threshold':.3,'atom_refute_threshold':.3},
    {'min_support_ratio':0}, {'min_support_ratio':1.01}, {'min_support_ratio':float('nan')},
    {'scoring_mode':'ab'}, {'tau_local':.8}, {'tau_final':.8}, {'calibration_version':'old'},
])
def test_configuration_migration_and_ranges(values):
    with pytest.raises(ValueError): EvidenceGateConfig(**values)


def many_atoms(n):
    def edit(output):
        original=output['atoms'][0]
        output['atoms']=[]
        for i in range(n-1):
            fact=deepcopy(original)
            fact.update(atom_id=f'f{i}',statement=f'必要断言 {i}')
            output['atoms'].append(fact)
        output['final_support']['depends_on']=[x['atom_id'] for x in output['atoms']]
    return edit


@pytest.mark.asyncio
@pytest.mark.parametrize('values,allowed,code',[
    ([A]*8+[B]*2,True,None),([A]*7+[B]*3,False,'SUPPORT_RATIO_BELOW_THRESHOLD'),
    ([A]*9+[C],False,'REFUTING_ATOM_PRESENT'),([B]*10,False,'SUPPORT_RATIO_BELOW_THRESHOLD'),
    ([A]*10,True,None),([C]+[A]*9,False,'REFUTING_ATOM_PRESENT'),
])
async def test_gate_support_ratio_zero_refutations(tmp_path,values,allowed,code):
    a,_,client=agent(tmp_path,edit=many_atoms(10),scores=values)
    result=await a.query('q')
    assert (result['decision']=='answer') is allowed
    assert len(client.prompts)==10
    assert result['gate']['total_count']==10
    assert result['gate']['support_count']==values.count(A)
    assert result['gate']['uncertain_count']==values.count(B)
    assert result['gate']['refute_count']==values.count(C)
    assert result['gate']['support_ratio']==values.count(A)/10
    if code: assert code in result['reason_codes']


@pytest.mark.asyncio
async def test_final_uncertain_optional_strict_policy(tmp_path):
    client=Client(edit=many_atoms(10),scores=[A]*9+[B])
    a,_,_=agent(tmp_path,client=client)
    basic=await a.query('q')
    assert basic['decision']=='answer' and basic['gate']['final_verdict']=='uncertain'
    a.gate.require_final_support=True
    client.probabilities=[A]*9+[B]+[A]*9+[B]
    strict=await a.query('q')
    assert strict['decision']=='insufficient'
    assert 'FINAL_ANSWER_NOT_SUPPORTED' in strict['reason_codes']
    assert strict['gate']['support_ratio']==.9


@pytest.mark.asyncio
@pytest.mark.parametrize('first',[B,C])
async def test_dependency_state_is_context_not_execution_block(tmp_path,first):
    def edit(output):
        other=deepcopy(output['atoms'][0]); other.update(atom_id='derived',kind='derived',statement='独立证据推出的结论',depends_on=['f'])
        output['atoms'].append(other);output['final_support']['depends_on']=['derived']
    a,_,client=agent(tmp_path,edit=edit,scores=[first,A,A])
    a.gate.min_support_ratio=.6
    result=await a.query('q')
    assert len(client.prompts)==3
    dependencies=json.loads(client.prompts[1].split('\n')[-1])['dependency_audits']
    assert dependencies[0]['audit']['verdict']==('uncertain' if first==B else 'refute')
    assert dependencies[0]['evidence'][0]['content']==7
    assert result['audit']['atom_reports'][1]['verdict']=='support'
    assert result['decision']==('answer' if first==B else 'insufficient')


@pytest.mark.asyncio
async def test_execution_error_cannot_be_diluted_by_ratio_and_final_still_audited(tmp_path):
    a,_,client=agent(tmp_path,edit=many_atoms(10),scores=[A]*10)
    original=client.score_labels
    async def score(prompt,**kwargs):
        dto=await original(prompt,**kwargs)
        if len(client.prompts)==1: dto.logprobs.pop('C')
        return dto
    client.score_labels=score
    result=await a.query('q')
    assert len(client.prompts)==10 and result['decision']=='error'
    assert result['gate']['support_ratio']==.9 and not result['gate']['allow']
    failed=result['audit']['atom_reports'][0]
    assert failed['execution_status']=='error' and failed['verdict'] is None
    assert failed['label_probs'] is None and 'SCORING_UNAVAILABLE' in result['reason_codes']


@pytest.mark.asyncio
@pytest.mark.parametrize('mutation,code',[
    (lambda a:a.atom_reports.pop(),'MISSING_AUDIT_REPORT'),
    (lambda a:a.atom_reports.append(a.atom_reports[0]),'MISSING_AUDIT_REPORT'),
    (lambda a:setattr(a,'subject_sha256','forged'),'AUDIT_SUBJECT_MISMATCH'),
    (lambda a:setattr(a.atom_reports[0],'verdict','uncertain'),'ATOM_AUDIT_FAILED'),
    (lambda a:setattr(a.atom_reports[0],'expected_score',0.),'ATOM_AUDIT_FAILED'),
    (lambda a:a.atom_reports[0].label_probs.update(A=.99),'ATOM_AUDIT_FAILED'),
    (lambda a:setattr(a.atom_reports[0],'support_threshold',.2),'ATOM_AUDIT_FAILED'),
    (lambda a:setattr(a.atom_reports[0],'prompt_sha256','forged'),'AUDIT_PROMPT_MISMATCH'),
    (lambda a:setattr(a.atom_reports[0],'template_version','ab-v3'),'ATOM_AUDIT_FAILED'),
    (lambda a:setattr(a.atom_reports[1].scores,'call_id',a.atom_reports[0].scores.call_id),'ATOM_AUDIT_FAILED'),
    (lambda a:setattr(a.citation_checks[0].resolved_evidence[0],'content',999),'CITATION_CHECK_FAILED'),
])
async def test_gate_rechecks_coverage_policy_values_and_binding(tmp_path,mutation,code):
    a,_,_=agent(tmp_path)
    await a.query('q')
    log=json.loads((tmp_path/'audit.jsonl').read_text())
    group=EvidenceGroup.model_validate(log['group']);candidate=AnswerCandidate.model_validate(log['candidate'])
    draft=ReasoningDraft.model_validate(log['draft']);audit=AuditReport.model_validate(log['audit'])
    mutation(audit)
    result=a.gate.decide(group,candidate,draft,audit)
    assert not result.allow and code in result.reason_codes


@pytest.mark.asyncio
async def test_empty_fixed_set_no_publication(tmp_path):
    a,_,_=agent(tmp_path);await a.query('q')
    log=json.loads((tmp_path/'audit.jsonl').read_text())
    group=EvidenceGroup.model_validate(log['group']);candidate=AnswerCandidate.model_validate(log['candidate'])
    draft=ReasoningDraft.model_validate(log['draft']);audit=AuditReport.model_validate(log['audit'])
    draft.atoms=[];audit.atom_reports=[];audit.citation_checks=[]
    result=a.gate.decide(group,candidate,draft,audit)
    assert not result.allow and result.support_ratio is None and 'NO_AUDITED_ATOMS' in result.reason_codes


@pytest.mark.asyncio
@pytest.mark.parametrize('orphan',[True,False])
async def test_redundant_or_unrelated_facts_cannot_inflate_denominator(tmp_path,orphan):
    def edit(output):
        other=deepcopy(output['atoms'][0]);other['atom_id']='extra'
        if orphan: other['statement']='无关事实'
        else: output['final_support']['depends_on'].append('extra')
        output['atoms'].append(other)
    a,_,client=agent(tmp_path,edit=edit)
    result=await a.query('q')
    assert result['decision']=='insufficient' and not client.prompts
    assert ('UNRELATED_ATOM' if orphan else 'DUPLICATE_ATOM_CLAIM') in result['reason_codes']


@pytest.mark.asyncio
async def test_stream_exposes_states_statistics_but_no_unpublished_candidate(tmp_path):
    a,_,_=agent(tmp_path,scores=[B,C])
    events=[e async for e in a.stream_query(user_message='q')]
    results=[e.data for e in events if e.type=='atom_audit_result']
    assert [r['verdict'] for r in results]==['uncertain','refute']
    assert all('statement' not in r and 'value' not in r for r in results)
    assert not any(e.type=='text' for e in events)
    gate=next(e.data for e in events if e.type=='gate_result')
    assert gate['refute_count']==1 and gate['uncertain_count']==1


@pytest.mark.asyncio
async def test_evaluation_abc_annotations_errors_and_historical_logs(tmp_path):
    from datamind.scripts.evidence_gate_evaluate import evaluate
    a,_,_=agent(tmp_path,scores=[B,C]);await a.query('q')
    log=json.loads((tmp_path/'audit.jsonl').read_text())
    annotation={'request_id':log['request_id'],'answer_supported':False,'answer_correct':False,
                'atom_verdict':{'f':'uncertain','a_final':'refute'},'perturbation_group':'same'}
    result=evaluate([log],[annotation])
    assert result['protocol_counts']=={'abc-v1':1}
    assert result['abc_classification']['accuracy']==1
    assert result['atom_verdict_counts']=={'uncertain':1,'refute':1}
    assert result['rejection_reason_counts']['REFUTING_ATOM_PRESENT']==1
    assert result['scoring_unavailable_rate']==0
    legacy=deepcopy(log);legacy.pop('protocol_version');legacy['result']['audit']['atom_reports']=[
        {'atom_id':'f','status':'passed','score_type':'raw_ab','p_raw':.9,'p_cal':None}]
    annotation['atom_supported']={'f':True};annotation.pop('atom_verdict')
    result=evaluate([legacy],[annotation])
    assert result['protocol_counts']=={'ab-legacy':1} and result['reliability']['p_raw']['n']==1


def test_old_ab_calibration_requires_rerun():
    from datamind.scripts.evidence_gate_calibrate import fit
    rows=[{'material_group':f'g{i}','logprob_a':-.1,'logprob_b':-2.,'supported_label':1} for i in range(8)]
    with pytest.raises(ValueError,match='ABC_CONFIG_MIGRATION_REQUIRED'):
        fit(rows,min_accepted=1)


@pytest.mark.asyncio
async def test_direct_scoring_transport_error_is_not_uncertainty(tmp_path):
    from datamind.agent.evidence_gated import ABCJudge
    a,_,_=agent(tmp_path)
    class Unavailable:
        async def score_labels(self,prompt): raise RuntimeError('gateway failed')
    a.judge=ABCJudge(Unavailable())
    result=await a.query('q')
    assert result['decision']=='error'
    assert len(result['audit']['atom_reports'])==2
    assert all(r['execution_status']=='error' and r['verdict'] is None for r in result['audit']['atom_reports'])


@pytest.mark.parametrize('value',[False,True,'-0.1',None])
def test_wire_logprob_must_be_a_number(value):
    with pytest.raises(ExternalServiceError):
        OpenAIChatCompletionsModelClient._parse_score_labels(body(c=value),['A','B','C'])
