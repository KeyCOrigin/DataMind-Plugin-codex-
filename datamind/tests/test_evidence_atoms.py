"""New protocol acceptance tests; fake scores test flow, not model accuracy."""
import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace
from uuid import uuid4
import pytest
from datamind.agent.evidence_gated import (
    EvidenceGatedAgent, RetrievalBridge, AnswerGenerator, AtomBuilder, AnswerCandidate, ABCJudge, TokenScoringAdapter,
    LabelScores, EvidenceRegistry, SourceMappingResolver, ReasoningDraft,
    ProvenanceValidator, Gate, AuditReport, AtomAudit,
)
from datamind.agent.evidence_gated.recorder import EvidenceRecorderHook
from datamind.agent.evidence_gated.evidence import subject, resolve, resolve_text_span, digest
from datamind.agent.evidence_gated.provenance import structure_errors
from datamind.core.hooks import HookChain
from datamind.core.model_clients import OpenAIChatCompletionsModelClient
from datamind.core.errors import ExternalServiceError
from datamind.config import EvidenceGateConfig

class Retrieve:
    def __init__(self, value=7, *, fail=False, delay=0):
        self.value, self.fail, self.delay = value, fail, delay
        self.hooks = HookChain([EvidenceRecorderHook()])
        self.questions = []
    async def query(self, message, *, history=None, final_contract=None):
        self.questions.append((message, final_contract))
        args = {"sql": "SELECT n FROM t"}
        await self.hooks.pre("db_query_sql", args)
        await asyncio.sleep(self.delay)
        await self.hooks.post("db_query_sql", args, {"rows": [[self.value]], "truncated": False}, None)
        if self.fail:
            raise RuntimeError("retrieval finalization failed")
        return {"answer": "UNTRUSTED", "evidence": [{"content": "FAKE"}]}

class Client:
    def __init__(self, value=7, *, edit=None, scores=None, answer_edit=None):
        self.value, self.edit, self.probabilities = value, edit, scores or []
        self.answer_edit = answer_edit
        self.prompts, self.generation_prompts = [], []
        self.build_inputs = []
    async def generate_text(self, prompt, **kwargs):
        self.generation_prompts.append(prompt)
        data = json.loads(prompt[prompt.index('\n{"answer_contract"')+1:])
        if prompt.startswith('阶段：answer_generation'):
            result = {'status':'candidate', 'answer_value':self.value, 'basis_notes':[], 'gaps':[]}
            if self.answer_edit:
                self.answer_edit(result)
            return json.dumps(result)
        self.build_inputs.append(data)
        ev = data['evidence'][0]['evidence_id']
        citation = {'evidence_id':ev, 'selector':{'kind':'json_pointer','pointer':'/rows/0/0'}}
        result = {'status':'candidate', 'atoms':[
            {'atom_id':'f','kind':'fact','statement':'记录中的 n 值','value':self.value,
             'citations':[citation],'depends_on':[]}],
            'final_support':{'citations':[deepcopy(citation)],'depends_on':['f']}, 'gaps':[]}
        if self.edit:
            self.edit(result)
        return json.dumps(result)
    async def score_labels(self, prompt, **kwargs):
        index = len(self.prompts)
        self.prompts.append(prompt)
        a, b, c = self.probabilities[index] if index < len(self.probabilities) else (-0.01, -5.0, -5.0)
        return LabelScores(logprobs={"A":a, "B":b, "C":c}, model='fixture', call_id=uuid4().hex, tokens={'A':'A','B':'B','C':'C'})

def agent(tmp_path, *, value=7, edit=None, scores=None, retrieve=None, client=None):
    retrieve = retrieve or Retrieve(value)
    client = client or Client(value, edit=edit, scores=scores)
    instance = EvidenceGatedAgent(bridge=RetrievalBridge(retrieve),
        answer_generator=AnswerGenerator(client, model="fixture"),
        atom_builder=AtomBuilder(client, model="fixture"),
        judge=ABCJudge(TokenScoringAdapter(client, model="fixture"), support_threshold=.3, refute_threshold=-.3),
        audit_path=tmp_path/"audit.jsonl")
    return instance, retrieve, client

@pytest.mark.asyncio
@pytest.mark.parametrize('value,contract', [(0,{"type":"integer"}), (False,{"type":"boolean"}), ([],{"type":"array"}), (7,{"type":"integer"})])
async def test_exact_publication_and_one_retrieval(tmp_path, value, contract):
    a, retrieve, client = agent(tmp_path, value=value)
    result = await a.query("原问题", context={"task_id":"task_266","task_id_origin":"request_context"}, final_contract=contract)
    assert result["decision"] == "answer"
    assert type(result["answer"]) is type(value) and result["answer"] == value
    assert len(retrieve.questions) == 1 and retrieve.questions[0][1] is None
    assert "task_266" in retrieve.questions[0][0]
    assert len(client.prompts) == 2
    assert 'all_evidence' in client.prompts[-1]
    assert 'dependency_audits' in client.prompts[-1]
    saved = json.loads((tmp_path/"audit.jsonl").read_text())
    assert saved["group"]["tool_trace"][0]["result"]["rows"] == [[value]]
    assert saved["result"]["answer"] == value

@pytest.mark.asyncio
@pytest.mark.parametrize('mutation,code', [
    (lambda d: d['atoms'][0]['citations'][0].update(evidence_id='fake'), 'UNKNOWN_EVIDENCE'),
    (lambda d: d['atoms'][0]['citations'][0].update(quote='invented'), 'QUOTE_MISMATCH'),
    (lambda d: d['atoms'][0]['citations'][0]['selector'].update(pointer='/absent'), 'INVALID_SELECTOR'),
    (lambda d: d['atoms'][0].update(citations=[]), 'NO_CITATION'),
    (lambda d: d['atoms'][0].update(depends_on=['a_final']), 'INVALID_DEPENDENCY'),
    (lambda d: d.update(answer_value=999), 'INVALID_ATOM_BUILD'),
    (lambda d: d['final_support'].update(citations=[]), 'INVALID_ATOM_BUILD'),
])
async def test_invalid_references_never_scored(tmp_path, mutation, code):
    a, _, client = agent(tmp_path, edit=mutation)
    result = await a.query('n?')
    assert result['decision'] != 'answer' and code in result['reason_codes']
    assert not client.prompts

@pytest.mark.asyncio
async def test_final_score_can_reject_despite_good_facts(tmp_path):
    a, _, client = agent(tmp_path, scores=[(-.01,-5.,-5.),(-5.,-.01,-5.)])
    result = await a.query('n?')
    assert result['decision'] == 'insufficient'
    assert 'SUPPORT_RATIO_BELOW_THRESHOLD' in result['reason_codes']
    assert len(client.prompts) == 2 and result['answer'] is None

@pytest.mark.asyncio
async def test_uncertain_dependency_continues_final(tmp_path):
    a, _, client = agent(tmp_path, scores=[(-5.,-.01,-5.)])
    result = await a.query('n?')
    assert result['decision'] == 'insufficient' and len(client.prompts) == 2
    assert result['audit']['atom_reports'][0]['verdict'] == 'uncertain'
    assert result['audit']['atom_reports'][1]['verdict'] == 'support'

@pytest.mark.asyncio
async def test_missing_real_scores_is_error(tmp_path):
    client = Client()
    async def missing(*args, **kwargs): return {'A': -.1}
    client.score_labels = missing
    a, _, _ = agent(tmp_path, client=client)
    result = await a.query('n?')
    assert result['decision'] == 'error'
    assert 'SCORING_UNAVAILABLE' in result['reason_codes']
    assert result['audit']['atom_reports'][0]['expected_score'] is None

@pytest.mark.asyncio
async def test_retrieval_answer_missing_or_failed_preserves_evidence(tmp_path):
    a, _, _ = agent(tmp_path, retrieve=Retrieve(fail=True))
    result = await a.query('n?')
    assert result['decision'] == 'answer'
    assert result['evidence'][0]['raw_payload']['rows'] == [[7]]
    assert result['answer'] != 'UNTRUSTED'

@pytest.mark.asyncio
async def test_concurrent_capture_isolation(tmp_path):
    a, _, _ = agent(tmp_path, value=1, retrieve=Retrieve(1, delay=.01))
    b, _, _ = agent(tmp_path, value=2, retrieve=Retrieve(2, delay=.001))
    one, two = await asyncio.gather(a.query('one'), b.query('two'))
    assert one['answer'] == 1 and two['answer'] == 2
    assert one['evidence'][0]['call_id'] != two['evidence'][0]['call_id']
    assert one['evidence'][0]['raw_payload']['rows'] == [[1]]

@pytest.mark.asyncio
async def test_stream_has_no_candidate_before_gate(tmp_path):
    a, _, _ = agent(tmp_path)
    events = [e async for e in a.stream_query(user_message='n?')]
    types = [e.type for e in events]
    assert types[:4] == ['retrieving', 'answer_generation', 'atom_building', 'citation_check']
    assert types.index('text') > types.index('gate')
    assert events[-1].type == 'done' and events[-1].data['answer'] == 7
    b, _, _ = agent(tmp_path, scores=[(-5.,-.01,-5.)])
    rejected = [e async for e in b.stream_query(user_message='n?')]
    assert not any(e.type == 'text' for e in rejected)
    assert rejected[-1].data['atoms'] == []

@pytest.mark.asyncio
async def test_subject_mutation_and_missing_reports_block_gate(tmp_path):
    a, _, _ = agent(tmp_path)
    await a.query('n?')
    saved = json.loads((tmp_path/'audit.jsonl').read_text())
    from datamind.agent.evidence_gated import EvidenceGroup
    group = EvidenceGroup.model_validate(saved['group'])
    candidate = AnswerCandidate.model_validate(saved['candidate'])
    draft = ReasoningDraft.model_validate(saved['draft'])
    audit = AuditReport.model_validate(saved['audit'])
    assert a.gate.decide(group,candidate,draft,audit).allow
    draft.answer_value = 999
    assert 'AUDIT_SUBJECT_MISMATCH' in a.gate.decide(group,candidate,draft,audit).reason_codes
    draft = ReasoningDraft.model_validate(saved['draft'])
    audit.atom_reports = []
    assert not a.gate.decide(group,candidate,draft,audit).allow
    group.evidence[0].raw_payload['rows'][0][0] = 999
    assert 'PAYLOAD_HASH_MISMATCH' in ProvenanceValidator().validate(draft,group)[0].reason_codes

@pytest.mark.asyncio
async def test_context_budget_fails_without_truncation(tmp_path):
    a, _, client = agent(tmp_path)
    a.judge.max_input_bytes = 10
    result = await a.query('n?')
    assert result['decision'] == 'error' and 'CONTEXT_BUDGET_EXCEEDED' in result['reason_codes']
    assert not client.prompts

@pytest.mark.asyncio
async def test_cancellation_logs_raw_tool_results(tmp_path):
    a, _, client = agent(tmp_path)
    async def hangs(*args, **kwargs): await asyncio.sleep(10)
    client.generate_text = hangs
    task = asyncio.create_task(a.query('n?'))
    await asyncio.sleep(.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError): await task
    saved = json.loads((tmp_path/'audit.jsonl').read_text())
    assert saved['reason_codes'] == ['CANCELLED'] and saved['group']['tool_trace']

def body(a=-.1,b=-2.,c=-5.):
    return {'model':'fixture','choices':[{'logprobs':{'content':[{'token':'A','logprob':a,
        'top_logprobs':[{'token':'B','logprob':b},{'token':'C','logprob':c}]}]}}]}

def test_first_position_exact_tokens_and_nonfinite_scores():
    parsed = OpenAIChatCompletionsModelClient._parse_score_labels(body(), ['A','B','C'])
    assert parsed.logprobs["A"] == -.1 and parsed.tokens['B'] == 'B'
    missing = body(); missing['choices'][0]['logprobs']['content'][0]['top_logprobs'] = []
    missing['choices'][0]['logprobs']['content'].append({'token':'B','logprob':-.2})
    with pytest.raises(ExternalServiceError):
        OpenAIChatCompletionsModelClient._parse_score_labels(missing,['A','B','C'])
    for value in (float('nan'), float('inf'), -9999., .1):
        with pytest.raises(ExternalServiceError):
            OpenAIChatCompletionsModelClient._parse_score_labels(body(b=value),['A','B','C'])
    spaced = body(); spaced['choices'][0]['logprobs']['content'][0]['top_logprobs'][0]['token'] = ' B'
    with pytest.raises(ExternalServiceError):
        OpenAIChatCompletionsModelClient._parse_score_labels(spaced,['A','B','C'])

def test_unicode_offsets_and_pointer_escape():
    assert resolve({'中/文~':[False]}, {'kind':'json_pointer','pointer':'/中~1文~0/0'}) is False
    assert resolve('甲😀乙',{'kind':'text_span','start':1,'end':2}) == '😀'
    for pointer in ('/rows/-1', '/rows/00', '/rows/~2'):
        with pytest.raises((ValueError,KeyError,IndexError)):
            resolve({'rows':[7]}, {'kind':'json_pointer','pointer':pointer})

@pytest.mark.asyncio
async def test_scoped_table_mapping_and_duplicate_evidence(tmp_path):
    receipt = {'receipt_id':'r1','revision':1,'profile':'default','source':{'uri':'f.xlsx'},
               'results':[{'surface':'db','status':'stored','details':{'source_file':'f.xlsx','tables':[
                   {'table':'t','sheet':'April'},{'table':'other','sheet':'May'}]}}]}
    path=tmp_path/'receipts.jsonl';path.write_text(json.dumps(receipt)+'\n')
    registry=EvidenceRegistry(SourceMappingResolver(path))
    bridge=RetrievalBridge(Retrieve(),registry=registry)
    group=await bridge.collect(question='n?',context={'profile':'default'},answer_contract={})
    mappings=group.evidence[0].source_bindings['import_mappings']
    assert len(mappings)==1 and mappings[0]['table']=='t' and mappings[0]['sheet']=='April'
    records=registry.register(group.tool_trace*2,{'profile':'default'})
    assert len(records)==1
    # NL2SQL table hints cannot connect arbitrary mappings to executed SQL.
    call=group.tool_trace[0].model_copy(deep=True)
    call.tool_name='db_query_nl';call.args={'tables':['t']};call.result={'sql':'SELECT n FROM other','rows':[[7]]}
    record=registry.register([call],{'profile':'default'})[0]
    assert record.source_bindings['import_mappings'][0]['table']=='other'

def test_removed_configuration_is_rejected():
    with pytest.raises(ValueError): EvidenceGateConfig(verifier_mode='structured')

@pytest.mark.asyncio
async def test_native_datamind_sql_end_to_end_and_logprob_wire(tmp_path, monkeypatch):
    """Real assembly/tool/hook/DB, and real wire parser with a fake HTTP provider."""
    import httpx
    from datamind.agent.options import build_datamind
    from datamind.config import Settings
    from datamind.core.protocols import ModelResponse, ModelUsage
    client = Client(value=7)
    calls=[]
    async def complete(**kwargs):
        calls.append(kwargs)
        if len(calls)==1:
            return ModelResponse(content=[{'type':'tool_use','name':'db_query_sql','id':'native1',
                'input':{'sql':'SELECT 7 AS n'}}], stop_reason='tool_use', usage=ModelUsage())
        return ModelResponse(content=[{'type':'text','text':'UNTRUSTED RETRIEVAL ANSWER'}], stop_reason='end_turn')
    client.complete=complete
    client.protocol='openai_chat_completions'
    async def close(): pass
    client.aclose=close
    monkeypatch.setattr('datamind.agent.options.build_model_client',lambda *args,**kwargs:client)
    settings=Settings(llm={'api_key':'fixture','protocol':'openai_chat_completions'},
        agent={'mode':'evidence_gated'}, hooks={'enabled':False})
    settings.data.base_dir=tmp_path
    system=await build_datamind(settings,enable={'db'})
    result=await system.query('n?', final_contract={'type':'integer'})
    assert result['decision']=='answer' and result['answer']==7
    assert result['evidence'][0]['tool_args']['sql']=='SELECT 7 AS n'
    assert result['evidence'][0]['raw_payload']['rows']==[[7]]
    assert system.retrieve.hooks.names()==['evidence_recorder']
    await system.aclose()
    wire_requests=[]
    def handler(request):
        wire_requests.append(json.loads(request.content))
        return httpx.Response(200,json=body())
    http=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    wire=OpenAIChatCompletionsModelClient(api_key='fixture',api_base='https://fixture.test',default_model='fixture',client=http)
    scores=await wire.score_labels('audit',labels=['A','B','C'])
    assert isinstance(scores,LabelScores) and scores.logprobs["B"]==-2.
    assert wire_requests[0]['max_tokens']==1 and wire_requests[0]['logprobs'] is True
    assert wire_requests[0]['temperature']==1.0
    await wire.aclose()

@pytest.mark.asyncio
async def test_scope_version_and_tool_failure_are_authenticity_failures(tmp_path):
    a, _, client=agent(tmp_path)
    result=await a.query('n?',context={'workspace_revision':'specific'})
    assert 'VERSION_MISMATCH' in result['reason_codes'] and not client.prompts
    class Failed(Retrieve):
        async def query(self, message, **kwargs):
            await self.hooks.post('db_query_sql',{'sql':'SELECT n FROM t'},None,RuntimeError('database failed'))
    a, _, client=agent(tmp_path,retrieve=Failed())
    # Build citations against the tool failure rather than assuming rows exist.
    result=await a.query('n?')
    assert 'TOOL_FAILED' in result['reason_codes'] and not client.prompts

@pytest.mark.asyncio
async def test_full_evidence_conflicts_and_partial_ranges_reach_final_judge(tmp_path):
    class Conflict(Retrieve):
        async def query(self, message, **kwargs):
            await super().query(message, **kwargs)
            await self.hooks.post('graph_neighbors',{'entity':'task_1'},
                {'edges':[{'head':'task_1','relation':'output','tail':'f'}], 'truncated':True},None)
    a, _, client=agent(tmp_path,retrieve=Conflict())
    await a.query('n?')
    assert 'output' in client.prompts[-1] and '"completeness":"partial"' in client.prompts[-1]
    # These assertions prove the audit sees conflicts and range flags;
    # they do not establish that a real model necessarily classifies them correctly.

@pytest.mark.asyncio
async def test_cte_alias_is_not_a_file_table_binding(tmp_path):
    bridge=RetrievalBridge(Retrieve())
    group=await bridge.collect(question='n?',context={},answer_contract={})
    call=group.tool_trace[0].model_copy(deep=True)
    call.args={'sql':'WITH t AS (SELECT * FROM other) SELECT * FROM t'}
    record=EvidenceRegistry().register([call],{})[0]
    assert record.source_bindings['queried_tables']==['other']

@pytest.mark.asyncio
async def test_offline_metrics_and_grouped_calibration(tmp_path):
    from datamind.scripts.evidence_gate_evaluate import evaluate
    from datamind.scripts.evidence_gate_calibrate import fit
    a, _, _=agent(tmp_path)
    await a.query('n?')
    log=json.loads((tmp_path/'audit.jsonl').read_text())
    annotation={'request_id':log['request_id'],'answer_supported':True,'answer_correct':True,
        'atom_supported':{'f':True,'a_final':True}}
    report=evaluate([log],[annotation])
    assert report['coverage']==1 and report['released_accuracy']==1
    assert report['reliability']['p_support']['n']==2
    assert report['latency_ms_mean']['total']>0
    rows=[{'material_group':f'material_{g}', 'request_id':f'r{g}-{y}',
           'answer_correct':bool(y),'integrity_valid':True,'final_atom_id':'a_final',
           'atom_reports':[{'atom_id':'a_final','execution_status':'scored','scores':LabelScores(
               logprobs={'A':-.01 if y else -5.,'B':-5.,'C':-5. if y else -.01},
               model='fixture',call_id=f's{g}-{y}',tokens={'A':'A','B':'B','C':'C'}).model_dump()}]}
          for g in range(20) for y in (0,1)]
    # Synthetic data validates the tooling only; this is not project calibration.
    calibration=fit(rows,min_accepted=1)
    splits=[set(v) for v in calibration['split_material_groups'].values()]
    assert not splits[0]&splits[1] and not splits[0]&splits[2] and not splits[1]&splits[2]
    assert calibration['selected_config']['scoring_mode']=='abc'
    assert calibration['heldout']['published_error_rate']==0


@pytest.mark.asyncio
async def test_kb_and_graph_nested_dtos_keep_exact_text_and_source_identity(tmp_path):
    from datamind.core.protocols import RetrievedChunk, Edge
    class NativeDTOs(Retrieve):
        async def query(self,message,**kwargs):
            await self.hooks.post('kb_search',{'query':'18.74'},
                {'results':[RetrievedChunk(id='c1',text='原文：18.74',source='f.docx',metadata={'page':3})]},None)
            await self.hooks.post('graph_neighbors',{'entity':'task_266'},
                {'edges':[Edge(source='task_266',target='f.docx',relation='task_requires_file')],'truncated':False},None)
    group=await RetrievalBridge(NativeDTOs()).collect(question='哪个文件?',context={},answer_contract={})
    assert len(group.evidence)==2
    text=group.evidence[0].raw_payload['results'][0]['text']
    assert text=='原文：18.74' and group.evidence[0].error is None
    draft=ReasoningDraft.model_validate({'draft_id':'d','group_id':group.group_id,'candidate_id':'c','candidate_sha256':'hash','status':'candidate',
        'atoms':[{'atom_id':'a','kind':'answer','statement':'f.docx 包含 18.74','value':'f.docx',
                  'citations':[{'evidence_id':group.evidence[0].evidence_id,
                    'selector':{'kind':'json_pointer','pointer':'/results/0/text'},'quote':'原文：18.74'}]}],
        'final_atom_id':'a','answer_value':'f.docx'})
    check=ProvenanceValidator().validate(draft,group)[0]
    assert check.valid
    identities=check.resolved_evidence[0].source_bindings['locator_context']
    assert identities[-1]['identity']['source']=='f.docx'
    assert identities[-1]['identity']['metadata']['page']==3
    edge=group.evidence[1].raw_payload['edges'][0]
    assert edge['relation']=='task_requires_file' and edge['target']=='f.docx'


class RetryClient(Client):
    """First atom build emits empty final_support.citations; the retry fixes it."""
    def __init__(self, value=7):
        super().__init__(value=value)
        self.build_calls = 0
        self.seen_candidates = []
    async def generate_text(self, prompt, **kwargs):
        if prompt.startswith('阶段：answer_generation'):
            return await super().generate_text(prompt, **kwargs)
        self.generation_prompts.append(prompt)
        self.build_calls += 1
        idx = prompt.index('\n{"answer_contract"') + 1
        data, _ = json.JSONDecoder().raw_decode(prompt, idx)
        self.seen_candidates.append(data['candidate']['answer_value'])
        ev = data['evidence'][0]['evidence_id']
        citation = {'evidence_id': ev, 'selector': {'kind':'json_pointer','pointer':'/rows/0/0'}}
        result = {'status':'candidate', 'atoms':[
            {'atom_id':'f','kind':'fact','statement':'记录中的 n 值','value':self.value,
             'citations':[deepcopy(citation)],'depends_on':[]}],
            'final_support':{'citations': [] if self.build_calls == 1 else [deepcopy(citation)],
                             'depends_on':['f']}, 'gaps':[]}
        return json.dumps(result)


@pytest.mark.asyncio
async def test_atom_builder_retries_empty_final_citations_and_freezes_candidate(tmp_path):
    a, _, client = agent(tmp_path, client=RetryClient())
    result = await a.query('n?')
    assert result['decision'] == 'answer' and result['answer'] == 7
    assert client.build_calls == 2
    assert client.seen_candidates == [7, 7]
    assert '上一次输出未通过校验' in client.generation_prompts[-1]


def test_resolve_text_span_anchors_exact_single_occurrence():
    payload = {"text": "alpha beta alpha"}
    span = resolve_text_span(payload, "/text", "beta")
    assert span == {"kind": "text_span", "pointer": "/text",
                    "start": 6, "end": 10, "occurrence": 0}
    assert resolve(payload, span) == "beta"


def test_resolve_text_span_requires_occurrence_on_multiple_matches():
    payload = {"text": "alpha beta alpha"}
    with pytest.raises(ValueError, match="AMBIGUOUS_TEXT_SPAN"):
        resolve_text_span(payload, "/text", "alpha")
    span = resolve_text_span(payload, "/text", "alpha", occurrence=1)
    assert span["start"] == 11 and span["end"] == 16
    assert resolve(payload, span) == "alpha"


def test_resolve_text_span_never_fuzzy_matches():
    payload = {"text": "alpha beta"}
    with pytest.raises(ValueError, match="NO_EXACT_MATCH"):
        resolve_text_span(payload, "/text", "betq")
    with pytest.raises(ValueError, match="INVALID_OCCURRENCE"):
        resolve_text_span(payload, "/text", "alpha", occurrence=5)
