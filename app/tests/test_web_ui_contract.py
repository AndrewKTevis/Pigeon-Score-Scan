from html.parser import HTMLParser
from pathlib import Path


WEB_ROOT = Path(__file__).resolve().parents[1] / "src" / "scorescan" / "web"


class _UiParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.buttons: list[dict[str, object]] = []
        self._button_stack: list[dict[str, object]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if element_id := attributes.get("id"):
            self.ids.append(element_id)
        if tag == "button":
            button: dict[str, object] = {"attrs": attributes, "text": []}
            self.buttons.append(button)
            self._button_stack.append(button)

    def handle_data(self, data: str) -> None:
        if self._button_stack:
            text = self._button_stack[-1]["text"]
            assert isinstance(text, list)
            text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "button" and self._button_stack:
            self._button_stack.pop()


def test_static_ui_has_unique_ids_and_named_buttons() -> None:
    parser = _UiParser()
    parser.feed((WEB_ROOT / "index.html").read_text(encoding="utf-8"))

    assert len(parser.ids) == len(set(parser.ids))
    for button in parser.buttons:
        attrs = button["attrs"]
        text = button["text"]
        assert isinstance(attrs, dict)
        assert isinstance(text, list)
        assert "".join(text).strip() or attrs.get("aria-label"), attrs.get("id")


def test_ui_exposes_one_clear_three_stage_workflow() -> None:
    html = (WEB_ROOT / "index.html").read_text(encoding="utf-8")

    assert html.index('id="workflowImport"') < html.index('id="workflowProcess"')
    assert html.index('id="workflowProcess"') < html.index('id="workflowOutput"')
    assert "New conversion" in html
    assert "Import score scans" in html
    assert "MXL + MusicXML" in html
    assert "Start conversion" in html
    assert 'role="dialog" aria-modal="true"' in html


def test_design_language_and_direct_copy_are_kept_by_contract() -> None:
    html = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
    css = (WEB_ROOT / "style.css").read_text(encoding="utf-8")
    javascript = (WEB_ROOT / "app.js").read_text(encoding="utf-8")
    visible_sources = html + javascript

    assert "gradient(" not in css
    assert "--paper:" in css
    assert "--ink:" in css
    assert "--blue:" in css
    assert 'data-design-language="editorial-score-desk@1"' in html
    assert "@media (max-width: 920px)" in css
    assert "@media (max-width: 680px)" in css
    assert "setWorkflow('process')" in javascript
    assert "setWorkflow('output')" in javascript
    assert "openUtilityPanel('systemPanel'" in javascript
    assert "item.critical === false" in javascript
    assert 'id="settingsPanel"' not in html
    assert 'id="logBox"' not in html
    assert "/api/preferences" not in javascript
    assert "CUDA" not in visible_sources
    assert "GPU" not in visible_sources
    for discarded_copy in (
        "自动放行",
        "可以直接使用",
        "查看完整支持范围",
        "几乎不会出错",
        "智能识别",
        "安全模式",
        "事务式",
        "门禁式",
        "静默漏识别",
    ):
        assert discarded_copy not in visible_sources


def test_ui_is_bilingual_with_english_default_and_english_progress() -> None:
    html = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
    javascript = (WEB_ROOT / "app.js").read_text(encoding="utf-8")

    assert '<html lang="en">' in html
    assert '<title>Pigeon Score Scan</title>' in html
    assert 'id="languageSelect"' in html
    assert '<option value="en">English</option>' in html
    assert '<option value="zh">中文</option>' in html
    assert 'localStorage.getItem(\'pigeon-score-scan-language\')' in javascript
    assert '02 / PROCESS' in html
    assert 'TIME REMAINING' in html
    assert 'SYSTEM CPU' in html
    assert 'SERVICE MEMORY' in html


def test_close_flow_uses_two_explicit_in_app_choices() -> None:
    html = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
    javascript = (WEB_ROOT / "app.js").read_text(encoding="utf-8")

    assert 'id="closeDialog"' in html
    assert 'id="cancelCloseButton"' in html
    assert 'id="exitCompletelyButton"' in html
    assert 'id="minimizeToTrayButton"' not in html
    assert 'Cancel' in html
    assert 'Exit completely' in html
    assert "confirm('" not in javascript
    assert "window.PigeonScoreScan = { showCloseDialog }" in javascript


def test_home_settings_and_mxl_first_output_are_kept_by_contract() -> None:
    html = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
    javascript = (WEB_ROOT / "app.js").read_text(encoding="utf-8")

    assert 'id="orientationConfirmed"' not in html
    assert 'id="outputName"' in html
    assert 'id="pdfDpi"' not in html
    assert 'id="finishView"' not in html
    assert 'Used for both output files' not in html
    assert 'data-i18n="outputFiles"' in html
    assert 'data-i18n="correctOrientation"' in html
    assert 'data-i18n="scenarioLimits"' in html
    assert 'class="format-mark"' not in html
    assert 'Scanned sheet music to MusicXML' not in html
    assert '<p>MusicXML and MXL</p>' not in html
    assert html.index('id="downloadMxl" class="button primary"') < html.index('id="downloadXml" class="button"')
    assert 'MXL files can be viewed and edited in MuseScore' in html
    assert 'MXL 文件可用 MuseScore 软件查看和编辑' in javascript
    assert 'id="previewDialog"' in html
    assert 'id="previewViewport"' in html
    assert "payload.page_count" in javascript
    assert 'target="_blank"' not in html


def test_full_height_layout_and_honest_remaining_time_are_kept_by_contract() -> None:
    css = (WEB_ROOT / "style.css").read_text(encoding="utf-8")
    javascript = (WEB_ROOT / "app.js").read_text(encoding="utf-8")

    assert "@media (min-width: 921px) and (min-height: 700px)" in css
    assert ".shell { height: 100vh;" in css
    assert ".workspace-card:not(.hidden)" in css
    assert "state.etaSeconds -" not in javascript
    assert "ESTIMATING" in javascript
    assert "FINALIZING" in javascript
    assert "roundedEta" in javascript
    assert "outputLimited: 'Manual review required'" in javascript
    assert "outputLimited: '需要人工复查'" in javascript
    assert ".result-status.needs-review { border-color: var(--success);" in css


def test_restrained_typography_and_review_copy_are_kept_by_contract() -> None:
    css = (WEB_ROOT / "style.css").read_text(encoding="utf-8")
    javascript = (WEB_ROOT / "app.js").read_text(encoding="utf-8")

    assert "h1 { margin: 0; color: var(--ink); font-size: 20px;" in css
    assert "h2 { margin: 0; color: var(--ink); font-size: 21px;" in css
    assert ".workflow-step strong { font-size: 12px; font-weight: 600; }" in css
    assert ".drop-zone > strong { color: var(--ink); font-size: 15px; font-weight: 600; }" in css
    assert ".workspace-card:not(.hidden) { min-height: 0; flex: 0 1 650px;" in css
    assert "if (stateName === 'review_recommended') return [t('outputLimited'), 'review'];" in javascript
    assert "if (stateName === 'reviewed_with_warnings') return [t('outputLimited'), 'review'];" in javascript


def test_drop_symbol_keeps_vertical_clearance_in_compact_desktop_layout() -> None:
    css = (WEB_ROOT / "style.css").read_text(encoding="utf-8")

    assert "padding: 14px 18px 12px" in css
    assert ".drop-zone { min-height: 190px; }" in css
