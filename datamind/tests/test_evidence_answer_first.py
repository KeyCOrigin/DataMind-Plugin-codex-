"""Two sequential calls and immutable candidate release; fake judges test flow only."""
import asyncio
from copy import deepcopy
import json
import pytest
from datamind.agent.evidence_gated import AnswerCandidate, EvidenceGroup, ReasoningDraft, AuditReport
from datamind.agent.evidence_gated.evidence import candidate_digest
from datamind.config import EvidenceGateConfig
from datamind.core.logging import current_context
from datamind.tests.test_evidence_atoms import agent, Client, Retrieve


def saved(tmp_path):
    return json.loads((tmp_path / 'audit.jsonl').read_text())


@pytest.mark.asyncio
async def test_candidate_is_saved_before_second_call_and_exactly_published(tmp_path):
    a, _, client = agent(tmp_path)
    original_build = a.atom_builder.build
    async def build(group, candidate, state=None):
        assert state['candidate_snapshot']['answer_value'] == 7
        assert state['candidate_snapshot_sha256'] == candidate.candidate_sha256
        assert len(client.generation_prompts) == 1
        assert 'draft' not in state
        assert candidate is not state['candidate']
        return await original_build(group, candidate, state=state)
    a.atom_builder.build = build
    result = await a.query('n?')
    log = saved(tmp_path)
    assert result['decision'] == 'answer'
    assert len(client.generation_prompts) == 2
    assert client.generation_prompts[0].startswith('阶段：answer_generation')
    assert client.generation_prompts[1].startswith('阶段：atom_building')
    assert client.build_inputs[0]['candidate']['answer_value'] == result['answer'] == 7
    assert log['candidate']['candidate_sha256'] == log['draft']['candidate_sha256']
    assert log['candidate']['candidate_id'] == log['draft']['candidate_id'] == result['candidate_id']
    assert log['draft']['atoms'][-1]['kind'] == 'answer'
    assert log['generation_calls'] == 2
    assert result['iterations'] == 4


@pytest.mark.asyncio
@pytest.mark.parametrize('mutation', [
    lambda x: x.update(answer_value=99),
    lambda x: x['final_support'].update(value=99),
    lambda x: x['atoms'][0].update(kind='answer'),
    lambda x: x['atoms'][0].update(atom_id='a_final'),
])
async def test_builder_cannot_supply_or_replace_final_answer(tmp_path, mutation):
    a, _, client = agent(tmp_path, edit=mutation)
    result = await a.query('n?')
    assert result['decision'] == 'error' and result['reason_codes'] == ['INVALID_ATOM_BUILD']
    assert saved(tmp_path)['candidate']['answer_value'] == 7
    assert not client.prompts and result['answer'] is None


@pytest.mark.asyncio
async def test_mutating_builder_input_does_not_mutate_live_candidate(tmp_path):
    a, _, _ = agent(tmp_path)
    original_build = a.atom_builder.build
    async def build(group, candidate, state=None):
        candidate.answer_value = 999
        candidate.basis_notes.append('attempted rewrite')
        return await original_build(group, candidate, state=state)
    a.atom_builder.build = build
    result = await a.query('n?')
    assert result['decision'] == 'answer' and result['answer'] == 7
    assert saved(tmp_path)['candidate']['basis_notes'] == []


@pytest.mark.asyncio
async def test_wrong_candidate_is_not_replaced_by_supported_fact(tmp_path):
    client = Client(value=7, answer_edit=lambda x:x.update(answer_value=999),
        scores=[(-.01,-5.,-5.),(-5.,-.01,-5.)])
    a, _, _ = agent(tmp_path, client=client)
    result = await a.query('n?')
    assert result['decision'] == 'insufficient'
    final_data = json.loads(client.prompts[-1].split('\n')[-1])
    assert final_data['answer'] == 999
    assert final_data['all_evidence'][0]['raw_payload']['rows'] == [[7]]
    log = saved(tmp_path)
    assert log['draft']['atoms'][0]['value'] == 7
    assert log['draft']['atoms'][-1]['value'] == log['candidate']['answer_value'] == 999
    assert log['reports'][-1]['verdict'] == 'uncertain' and result['answer'] is None


@pytest.mark.asyncio
async def test_unverified_basis_notes_are_not_judge_evidence(tmp_path):
    marker = 'UNSUPPORTED_BASIS_NOTE_999'
    client = Client(answer_edit=lambda x:x.update(basis_notes=[marker]))
    a, _, _ = agent(tmp_path, client=client)
    await a.query('n?')
    assert marker in client.generation_prompts[1]
    assert all(marker not in prompt for prompt in client.prompts)
    assert saved(tmp_path)['candidate']['basis_notes'] == [marker]


@pytest.mark.asyncio
@pytest.mark.parametrize('field', ['answer_value', 'basis_notes', 'candidate_id', 'group_id'])
async def test_candidate_mutation_after_audit_cannot_pass_gate(tmp_path, field):
    a, _, _ = agent(tmp_path)
    await a.query('n?')
    log = saved(tmp_path)
    group = EvidenceGroup.model_validate(log['group'])
    candidate = AnswerCandidate.model_validate(log['candidate'])
    draft = ReasoningDraft.model_validate(log['draft'])
    audit = AuditReport.model_validate(log['audit'])
    setattr(candidate, field, ['forged'] if field == 'basis_notes' else 999 if field == 'answer_value' else 'forged')
    # Even recalculating a new hash cannot reuse the original audit.
    candidate.candidate_sha256 = candidate_digest(group, candidate)
    draft.candidate_sha256 = candidate.candidate_sha256
    decision = a.gate.decide(group, candidate, draft, audit)
    assert not decision.allow and 'AUDIT_SUBJECT_MISMATCH' in decision.reason_codes


@pytest.mark.asyncio
async def test_live_candidate_change_during_scoring_is_rejected(tmp_path):
    a, _, client = agent(tmp_path)
    original_generate = a.answer_generator.generate
    candidates = []
    async def generate(group, state=None):
        value = await original_generate(group, state=state)
        candidates.append(value)
        return value
    a.answer_generator.generate = generate
    original_score = client.score_labels
    async def score(prompt, **kwargs):
        result = await original_score(prompt, **kwargs)
        if len(client.prompts) == 2:
            candidates[0].answer_value = 999
        return result
    client.score_labels = score
    result = await a.query('n?')
    assert result['decision'] == 'insufficient' and 'CANDIDATE_CHANGED' in result['reason_codes']
    assert result['answer'] is None and saved(tmp_path)['candidate_snapshot']['answer_value'] == 7


@pytest.mark.asyncio
@pytest.mark.parametrize('stage', ['answer', 'build'])
async def test_insufficient_stages_keep_gaps_and_do_not_score(tmp_path, stage):
    gap = 'Missing evidence for the necessary premise'
    kwargs = {'answer_edit':lambda x:x.update(status='insufficient',answer_value=None,gaps=[gap])} if stage=='answer' else {
        'edit':lambda x:x.update(status='insufficient',atoms=[],final_support=None,gaps=[gap])}
    client = Client(**kwargs)
    a, _, _ = agent(tmp_path, client=client)
    result = await a.query('n?')
    assert result['decision'] == 'insufficient' and not client.prompts
    assert len(client.generation_prompts) == (1 if stage=='answer' else 2)
    log = saved(tmp_path)
    assert gap in (log['candidate']['gaps'] if stage=='answer' else log['draft']['gaps'])
    if stage=='build':
        assert log['candidate']['answer_value'] == 7 and log['draft']['answer_value'] is None


@pytest.mark.asyncio
async def test_direct_evidence_can_support_one_server_answer_atom(tmp_path):
    client = Client(edit=lambda x:(x.update(atoms=[]),x['final_support'].update(depends_on=[])))
    a, _, _ = agent(tmp_path, client=client)
    a.atom_builder.max_atoms = 1
    result = await a.query('n?')
    assert result['decision'] == 'answer' and len(result['atoms']) == 1
    assert result['atoms'][0]['atom_id'] == 'a_final' and len(client.prompts) == 1


@pytest.mark.asyncio
async def test_generation_context_budget_is_checked_separately(tmp_path):
    a, _, client = agent(tmp_path)
    a.atom_builder.max_input_bytes = 10
    result = await a.query('n?')
    assert result['decision'] == 'error' and result['reason_codes'] == ['CONTEXT_BUDGET_EXCEEDED']
    assert len(client.generation_prompts) == 1 and not client.prompts
    assert saved(tmp_path)['candidate']['answer_value'] == 7
    assert saved(tmp_path)['generation_calls'] == 1


@pytest.mark.asyncio
async def test_cancellation_in_second_call_preserves_first_candidate(tmp_path):
    a, _, client = agent(tmp_path)
    original = client.generate_text
    entered = asyncio.Event()
    async def generate(prompt, **kwargs):
        if prompt.startswith('阶段：atom_building'):
            entered.set()
            await asyncio.Event().wait()
        return await original(prompt, **kwargs)
    client.generate_text = generate
    task = asyncio.create_task(a.query('n?'))
    await asyncio.wait_for(entered.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    log = saved(tmp_path)
    assert log['candidate']['answer_value'] == 7 and log['reason_codes'] == ['CANCELLED']
    assert 'atom_building' in log['timings'] and not client.prompts


@pytest.mark.asyncio
async def test_per_stage_usage_and_model_metadata(tmp_path):
    a, _, client = agent(tmp_path)
    original = client.generate_text
    async def generate(prompt, **kwargs):
        current_context().extra.setdefault('nested_model_usage', []).append({'input_tokens':11,'output_tokens':3,'model':'fixture'})
        return await original(prompt, **kwargs)
    client.generate_text = generate
    result = await a.query('n?')
    for stage in ('answer_generation', 'atom_building'):
        assert len(result['usage'][stage]) == 1
        assert result['usage'][stage][0]['input_tokens'] == 11
        assert saved(tmp_path)[stage+'_call']['model'] == 'fixture'
    assert len(result['usage']['reasoning']) == 2


@pytest.mark.asyncio
async def test_concurrent_requests_on_one_agent_have_separate_candidates(tmp_path):
    class RoutedRetrieve(Retrieve):
        async def query(self, message, **kwargs):
            value = 1 if message.startswith('one') else 2
            await self.hooks.pre('db_query_sql', {'sql':'SELECT n FROM t'})
            await asyncio.sleep(.001)
            await self.hooks.post('db_query_sql', {'sql':'SELECT n FROM t'}, {'rows':[[value]]}, None)
            return {}
    class RoutedClient(Client):
        async def generate_text(self, prompt, **kwargs):
            data = json.loads(prompt[prompt.index('\n{"answer_contract"')+1:])
            delegate = Client(value=data['evidence'][0]['raw_payload']['rows'][0][0])
            return await delegate.generate_text(prompt, **kwargs)
    a, _, _ = agent(tmp_path, retrieve=RoutedRetrieve(), client=RoutedClient())
    one, two = await asyncio.gather(a.query('one'), a.query('two'))
    assert one['answer'] == 1 and two['answer'] == 2
    assert one['candidate_id'] != two['candidate_id']
    logs = [json.loads(line) for line in (tmp_path/'audit.jsonl').read_text().splitlines()]
    assert len({x['candidate_snapshot_sha256'] for x in logs}) == 2


def test_legacy_budget_migrates_and_conflicts_are_rejected():
    cfg = EvidenceGateConfig(max_output_tokens=1234)
    assert cfg.atom_max_output_tokens == 1234 and cfg.answer_max_output_tokens == 4096
    assert 'max_output_tokens' not in cfg.model_dump()
    with pytest.raises(ValueError):
        EvidenceGateConfig(max_output_tokens=1234, atom_max_output_tokens=4321)


@pytest.mark.asyncio
async def test_post_gate_candidate_mutation_resets_allow_and_audit_reasons(tmp_path):
    a, _, _ = agent(tmp_path)
    original = a.gate.decide
    def mutate(group, candidate, draft, audit):
        decision = original(group, candidate, draft, audit)
        assert decision.allow
        candidate.answer_value = 999
        return decision
    a.gate.decide = mutate
    result = await a.query('n?')
    assert result['decision'] == 'insufficient' and result['answer'] is None
    assert result['gate']['allow'] is False
    assert 'CANDIDATE_CHANGED' in result['audit']['reason_codes']
