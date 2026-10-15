"""Scalar citations keep their own relationship record, not unrelated neighbors."""
import json
import pytest
from datamind.agent.evidence_gated import (
    ABCJudge, Atom, Citation, EvidenceGroup, EvidenceRegistry, LabelScores,
    ProvenanceValidator, ReasoningDraft, ToolRecord,
)
from datamind.agent.evidence_gated.evidence import record_context


@pytest.mark.parametrize('edge,endpoint', [
    ({'to':'F','rel':'task_requires_file'}, 'to'),
    ({'source':'task_300','target':'F','relation':'task_requires_file'}, 'target'),
    ({'head':'task_300','tail':'F','rel':'task_requires_file'}, 'tail'),
    ({'destination_key':'F','link_kind':'task_requires_file'}, 'destination_key'),
])
def test_scalar_edge_has_full_record_for_any_field_names(edge, endpoint):
    payload = {'edges':[edge, {'to':'OTHER_RECORD_SECRET','rel':'task_produces_output'}]}
    result = record_context(payload, {'kind':'json_pointer','pointer':'/edges/0/'+endpoint})
    assert result == {'pointer':'/edges/0','content':edge}
    assert 'OTHER_RECORD_SECRET' not in json.dumps(result)
    result['content'][endpoint] = 'changed'
    assert payload['edges'][0][endpoint] == 'F'


def test_nested_endpoint_keeps_relationship_from_the_edge():
    edge = {'target':{'id':'F'},'relation':'task_requires_file'}
    assert record_context({'edges':[edge]}, {'kind':'json_pointer','pointer':'/edges/0/target/id'}) == {
        'pointer':'/edges/0','content':edge}


def test_citing_output_does_not_acquire_required_input_relation():
    payload = [{'to':'F','rel':'task_produces_output'}, {'to':'F','rel':'task_requires_file'}]
    result = record_context(payload, {'kind':'json_pointer','pointer':'/0/to'})
    assert result['content'] == payload[0]
    assert 'task_requires_file' not in json.dumps(result)


def test_direct_record_citation_needs_no_duplicate_context():
    assert record_context([{'to':'F','rel':'required'}], {'kind':'json_pointer','pointer':'/0'}) is None


def test_table_cell_keeps_its_row_not_other_rows():
    payload = {'columns':['file','role'], 'rows':[['F','input'],['F','output']]}
    assert record_context(payload, {'kind':'json_pointer','pointer':'/rows/0/0'}) == {
        'pointer':'/rows/0','content':['F','input']}


def test_text_span_keeps_chunk_record():
    payload = {'results':[{'text':'F contains 18.74','source':'F','metadata':{'page':3}}]}
    selector = {'kind':'text_span','pointer':'/results/0/text','start':11,'end':16}
    assert record_context(payload, selector) == {'pointer':'/results/0','content':payload['results'][0]}


@pytest.mark.asyncio
async def test_provenance_and_judge_receive_verified_edge_and_call_scope():
    records = [{'to':'F','rel':'task_requires_file'}, {'to':'OTHER_RECORD_SECRET','rel':'task_produces_output'}]
    call = ToolRecord(call_id='c',tool_name='graph_neighbors',args={'node':'task_300'},
        result=records,read_at='fixture')
    group = EvidenceGroup(request_id='r',group_id='g',question='Which required file?',context={},
        answer_contract={},evidence=EvidenceRegistry().register([call],{}),tool_trace=[call])
    citation = Citation(evidence_id=group.evidence[0].evidence_id,
        selector={'kind':'json_pointer','pointer':'/0/to'},quote='F')
    atom = Atom(atom_id='a3',kind='fact',statement='F is a required input for task_300',citations=[citation])
    draft = ReasoningDraft(draft_id='d',group_id='g',candidate_id='candidate',candidate_sha256='fixture',
        status='candidate',atoms=[atom])
    check = ProvenanceValidator().validate(draft,group)[0]
    assert check.valid and check.resolved_evidence[0].content == 'F'
    class Scoring:
        async def score_labels(self, prompt):
            self.prompt = prompt
            return LabelScores(logprobs={"A":-.01,"B":-5.,"C":-5.},model='fixture',call_id='s',tokens={'A':'A','B':'B','C':'C'})
    scoring = Scoring()
    judge = ABCJudge(scoring,support_threshold=.3,refute_threshold=-.3)
    await judge.score(group=group,atom=atom,citations=check.resolved_evidence,dependencies=[],all_reports=[])
    data = json.loads(scoring.prompt.split('\n')[-1])
    resolved = data['resolved_citations'][0]
    assert resolved['record_context'] == {'pointer':'/0','content':records[0]}
    assert resolved['tool_context'] == {'tool_name':'graph_neighbors','tool_args':{'node':'task_300'}}
    assert resolved['source_bindings']['locator_context'][0]['identity']['rel'] == 'task_requires_file'
    assert 'OTHER_RECORD_SECRET' not in scoring.prompt
    resolved = check.resolved_evidence[0]
    resolved.tool_context['tool_args']['node'] = 'changed'
    assert call.args['node'] == 'task_300'


def _text_group(text):
    call = ToolRecord(call_id='c', tool_name='kb_find_in_document', args={'source_file':'f','query':'q'},
        result={'results':[{'text':text,'source':'f'}]}, read_at='fixture')
    group = EvidenceGroup(request_id='r', group_id='g', question='q', context={},
        answer_contract={}, evidence=EvidenceRegistry().register([call], {}), tool_trace=[call])
    return group


def test_server_anchors_quote_to_traceable_text_span():
    group = _text_group('alpha beta alpha')
    citation = Citation(evidence_id=group.evidence[0].evidence_id,
        selector={'kind':'json_pointer','pointer':'/results/0/text','occurrence':1}, quote='alpha')
    atom = Atom(atom_id='a1', kind='fact', statement='second alpha', citations=[citation])
    draft = ReasoningDraft(draft_id='d', group_id='g', candidate_id='c', candidate_sha256='h',
        status='candidate', atoms=[atom])
    check = ProvenanceValidator().validate(draft, group)[0]
    assert check.valid, check.reason_codes
    selector = check.resolved_evidence[0].selector
    assert selector == {'kind':'text_span','pointer':'/results/0/text','start':11,'end':16,'occurrence':1}


def test_server_rejects_ambiguous_quote_without_occurrence():
    group = _text_group('alpha beta alpha')
    citation = Citation(evidence_id=group.evidence[0].evidence_id,
        selector={'kind':'json_pointer','pointer':'/results/0/text'}, quote='alpha')
    atom = Atom(atom_id='a1', kind='fact', statement='some alpha', citations=[citation])
    draft = ReasoningDraft(draft_id='d', group_id='g', candidate_id='c', candidate_sha256='h',
        status='candidate', atoms=[atom])
    check = ProvenanceValidator().validate(draft, group)[0]
    assert not check.valid and 'AMBIGUOUS_TEXT_SPAN' in check.reason_codes


def test_server_rejects_quote_without_exact_match():
    group = _text_group('alpha beta')
    citation = Citation(evidence_id=group.evidence[0].evidence_id,
        selector={'kind':'json_pointer','pointer':'/results/0/text'}, quote='betq')
    atom = Atom(atom_id='a1', kind='fact', statement='typo quote', citations=[citation])
    draft = ReasoningDraft(draft_id='d', group_id='g', candidate_id='c', candidate_sha256='h',
        status='candidate', atoms=[atom])
    check = ProvenanceValidator().validate(draft, group)[0]
    assert not check.valid and 'NO_EXACT_MATCH' in check.reason_codes
