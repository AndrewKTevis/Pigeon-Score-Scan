from pathlib import Path

from scorescan.product_scope import (
    MAXIMUM_KEYBOARD_PARTS,
    MAXIMUM_KEYBOARD_STAVES,
    MAXIMUM_KEYBOARD_VOICES_PER_STAFF_PER_MEASURE,
    MAXIMUM_NON_KEYBOARD_VOICES_PER_STAFF_PER_MEASURE,
    MAXIMUM_PHYSICAL_STAVES_PER_SYSTEM,
    PRODUCTION_BOUNDARY_CONTRACT_VERSION,
)


def test_user_visible_scope_matches_machine_readable_contract() -> None:
    web_root = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "scorescan"
        / "web"
    )
    html = (web_root / "index.html").read_text(encoding="utf-8")
    bilingual_copy = html + (web_root / "app.js").read_text(encoding="utf-8")

    assert PRODUCTION_BOUNDARY_CONTRACT_VERSION in html
    assert f"最多 {MAXIMUM_PHYSICAL_STAVES_PER_SYSTEM} 行物理谱表" in bilingual_copy
    assert f"最多 {MAXIMUM_KEYBOARD_PARTS} 个键盘声部" in bilingual_copy
    assert f"最多 {MAXIMUM_KEYBOARD_STAVES} 行物理谱表" in bilingual_copy
    assert (
        f"最多 {MAXIMUM_KEYBOARD_VOICES_PER_STAFF_PER_MEASURE} 个"
        in bilingual_copy
    )
    assert (
        f"每谱表仅支持 "
        f"{MAXIMUM_NON_KEYBOARD_VOICES_PER_STAFF_PER_MEASURE} 个"
        in bilingual_copy
    )
    assert "不会自动旋转或自动纠斜" in bilingual_copy
    assert "各乐器不需要逐音符横向对齐" in bilingual_copy
    assert "列表从上到下合并为一份 MusicXML" in bilingual_copy
    assert "分别扫描的独立分谱请分开转换" in bilingual_copy
    assert 'id="gpuSetupTrack"' not in html
    assert 'id="gpuSetupPercent"' not in html
