"""Browser UI Markdown rendering regression tests."""

import json
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
NODE = shutil.which("node")


def test_repo_and_packaged_frontend_assets_stay_in_sync():
    assert (ROOT / "static" / "app.html").read_bytes() == (
        ROOT / "datamind" / "static" / "app.html"
    ).read_bytes()
    assert (ROOT / "static" / "markdown.js").read_bytes() == (
        ROOT / "datamind" / "static" / "markdown.js"
    ).read_bytes()


@pytest.mark.skipif(NODE is None, reason="Node.js is required to execute the browser renderer")
def test_markdown_renderer_handles_blocks_and_escapes_html():
    source = """## 查询结果

| 人员 | 销售额 |
| --- | ---: |
| Bob | **200** |

```sql
SELECT * FROM sales WHERE amount < 300;
```

- 已恢复
- `<script>alert(1)</script>`
"""
    module_path = str(ROOT / "static" / "markdown.js")
    script = (
        "const renderer=require(process.argv[1]);"
        "const fs=require('fs');"
        "process.stdout.write(renderer.renderMarkdown(fs.readFileSync(0,'utf8')));"
    )
    result = subprocess.run(
        [NODE, "-e", script, module_path],
        input=source,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
    )
    html = result.stdout
    assert "<h2>查询结果</h2>" in html
    assert "<table>" in html and "<th>人员</th>" in html
    assert "<td><strong>200</strong></td>" in html
    assert '<pre><code class="language-sql">' in html
    assert "SELECT * FROM sales WHERE amount &lt; 300;" in html
    assert "<ul><li>已恢复</li>" in html
    assert "<script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_chat_and_store_responses_use_markdown_renderer():
    app = (ROOT / "static" / "app.html").read_text(encoding="utf-8")
    assert '<script src="/static/markdown.js"></script>' in app
    assert "const html = DataMindMarkdown.renderMarkdown(raw);" in app
    assert "aiBubble.dataset.raw = [result.answer" in app
    assert "renderMarkdown(aiBubble);" in app


def test_answer_first_phases_have_frontend_labels():
    app = (ROOT / "static" / "app.html").read_text(encoding="utf-8")
    assert "answer_generation: '正在生成候选答案'" in app
    assert "atom_building: '正在构建可核验依据'" in app
    assert "'retrieving', 'answer_generation', 'atom_building', 'citation_check'" in app


def test_abc_states_and_gate_statistics_have_frontend_labels():
    app = (ROOT / "static" / "app.html").read_text(encoding="utf-8")
    assert "atom_audit_result" in app and "gate_result" in app
    for label in ("支持", "不确定", "反驳", "审核错误", "支持比例"):
        assert label in app
