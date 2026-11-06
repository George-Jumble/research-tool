"""Old (pre-rename) names still resolve so existing data keeps loading."""

from pathlib import Path

from sb_shared import legacy_names as ln


def test_old_module_tags_map_to_current():
    assert ln.canonical_module_tag("LHCB_TAV_ECHO") == "LHCB_FLAVOR"
    assert ln.canonical_module_tag("TAU_SB_DESI") == "DESI_BAO"
    assert ln.canonical_module_tag("DESI_BAO") == "DESI_BAO"


def test_classify_action_accepts_old_tag():
    from superblock_mode import classify_action

    assert classify_action("LHCB_TAV_ECHO", "Full Flavor Residual Report") == classify_action(
        "LHCB_FLAVOR", "Full Flavor Residual Report"
    )


def test_staging_dir_reuses_existing_legacy_dir(tmp_path: Path):
    legacy = tmp_path / ln.LEGACY_STAGING_DIRNAME
    legacy.mkdir()
    (legacy / "keep.txt").write_text("user data")
    assert ln.resolve_staging_dir(tmp_path) == legacy.resolve()
    assert (legacy / "keep.txt").read_text() == "user data"


def test_staging_dir_new_when_no_legacy(tmp_path: Path):
    assert ln.resolve_staging_dir(tmp_path).name == ".sb_project_staging"


def test_env_fallback(monkeypatch):
    monkeypatch.delenv("SB_CACHE_DIR", raising=False)
    monkeypatch.setenv("TAV_CACHE_DIR", "/tmp/x")
    assert ln.getenv_compat("SB_CACHE_DIR") == "/tmp/x"
    monkeypatch.setenv("SB_CACHE_DIR", "/tmp/y")
    assert ln.getenv_compat("SB_CACHE_DIR") == "/tmp/y"
