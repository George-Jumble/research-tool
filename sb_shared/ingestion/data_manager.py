"""Simple data fetch/cache helper for ingestion workflows."""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sb_shared.legacy_names import getenv_compat

# Keep existing default log destination while avoiding duplicate root handlers.
logging.basicConfig(
    level=logging.INFO,
    filename="research_tool.log",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ResearchTool")

DATA_MANAGER_PARAM_KEY = "data_manager"
DATA_MANAGER_LEGACY_PARAM_KEY = "__data_manager__"

_RUNTIME_DATA_MANAGER: "DataManager | None" = None
_RUNTIME_CONFIG: tuple[bool, str] | None = None


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _runtime_config_from_params(
    params: dict[str, Any] | None,
    *,
    use_mock: bool | None,
    cache_dir: str | os.PathLike[str] | None,
) -> tuple[bool, str]:
    options = params or {}
    env_use_mock = getenv_compat("SB_USE_MOCK_DATA")
    env_cache_dir = getenv_compat("SB_CACHE_DIR")

    resolved_use_mock = _to_bool(
        use_mock if use_mock is not None else options.get("use_mock", env_use_mock),
        default=False,
    )
    raw_cache = cache_dir or options.get("cache_dir") or env_cache_dir or "./cache"
    resolved_cache = str(Path(raw_cache).expanduser().resolve())
    return resolved_use_mock, resolved_cache


class DataManager:
    """Fetch data from cache, mock generator, or source files."""

    def __init__(
        self,
        use_mock: bool = False,
        cache_dir: str | os.PathLike[str] = "./cache",
        sparc_archive_path: str | os.PathLike[str] | None = None,
    ):
        self.use_mock = bool(use_mock)
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if sparc_archive_path is None:
            from sb_shared.project_paths import FINISHED_A_DIR

            sparc_archive_path = FINISHED_A_DIR
        self.archive_path = Path(sparc_archive_path).expanduser().resolve()
        self.cache: dict[str, Any] = {}
        logger.info("Initialized DataManager (Mode: %s)", "Mock" if self.use_mock else "Real")

    def fetch_sparc_catalog(self) -> pd.DataFrame:
        """Fetch the SPARC galaxy kinematic catalog for diagnostics/integration workflows."""
        if self.use_mock:
            # Mock key outliers and near-theory points for deterministic diagnostics.
            return pd.DataFrame(
                {
                    "galaxy_id": [
                        "CamB",
                        "F561-1",
                        "NGC0891",
                        "NGC3953",
                        "NGC3741",
                        "UGC01234",
                    ],
                    "r_shadow_baryon_obs": [0.02, 0.04, 0.05, 0.03, 6.57, 5.18],
                    "delta_phi_est": [2.4, 2.1, 1.9, 2.0, 0.1, 0.3],
                    "dist_to_filament_Mpc": [0.8, 1.1, 0.9, 1.2, 8.5, 4.2],
                    "xray_excess_lum": [1.4e40, 1.1e40, 2.3e40, 1.8e40, 1.2e38, 5.0e38],
                }
            )

        return self._load_real_sparc()

    def cross_match_cosmic_web(
        self,
        sparc_df: pd.DataFrame,
        lambda_web: float = 7.0,
    ) -> pd.DataFrame:
        """Assign cosmic-web environment tags from transverse filament distance."""
        if "dist_to_filament_Mpc" not in sparc_df.columns:
            raise ValueError("SPARC frame missing required column: dist_to_filament_Mpc")

        frame = sparc_df.copy()
        conditions = [
            frame["dist_to_filament_Mpc"] < 0.3 * float(lambda_web),
            frame["dist_to_filament_Mpc"] > 1.2 * float(lambda_web),
        ]
        choices = ["FILAMENT_1D", "VOID_3D"]
        frame["environment"] = np.select(conditions, choices, default="FIELD")
        return frame

    def load_processed_sparc_batch(self) -> pd.DataFrame:
        """Load processed SPARC batch rows from the staged archive path."""
        if not self.archive_path.exists():
            raise FileNotFoundError(
                f"SPARC archive path not found: {self.archive_path}. "
                "Update sparc_archive_path in DataManager initialization."
            )

        csv_files = sorted(self.archive_path.rglob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in SPARC archive: {self.archive_path}")

        frames: list[pd.DataFrame] = []
        for file_path in csv_files:
            try:
                frame = pd.read_csv(file_path)
            except Exception:
                logger.warning("Skipping unreadable SPARC batch file: %s", file_path)
                continue
            if not frame.empty:
                frames.append(frame)

        if not frames:
            raise ValueError(f"No readable non-empty CSV batch files in: {self.archive_path}")

        merged = pd.concat(frames, ignore_index=True)

        id_aliases = ["galaxy_id", "ID", "id", "galaxy", "name"]
        shear_aliases = ["shear_metric", "shear", "domain_shear", "shear_score"]

        id_col = next((c for c in id_aliases if c in merged.columns), None)
        if id_col is None:
            raise ValueError("Processed SPARC batch missing galaxy identifier column.")

        shear_col = next((c for c in shear_aliases if c in merged.columns), None)
        if shear_col is None:
            raise ValueError("Processed SPARC batch missing shear metric column.")

        out = merged.rename(columns={id_col: "galaxy_id", shear_col: "shear_metric"}).copy()
        out["galaxy_id"] = out["galaxy_id"].astype(str).str.strip()
        out["shear_metric"] = pd.to_numeric(out["shear_metric"], errors="coerce")
        out = out.dropna(subset=["galaxy_id", "shear_metric"])
        if out.empty:
            raise ValueError("Processed SPARC batch contained no valid galaxy_id/shear_metric rows.")
        return out.reset_index(drop=True)

    def cross_match_environment_and_xray(
        self,
        sparc_df: pd.DataFrame,
        lambda_web: float = 7.0,
    ) -> pd.DataFrame:
        """Cross-match SPARC rows with mock environment classes and diffuse X-ray excess."""
        frame = sparc_df.copy()
        if "shear_metric" not in frame.columns:
            raise ValueError("SPARC frame missing required column: shear_metric")

        frame["shear_metric"] = pd.to_numeric(frame["shear_metric"], errors="coerce")
        frame = frame.dropna(subset=["shear_metric"])
        if frame.empty:
            raise ValueError("SPARC frame has no valid shear_metric values.")

        # Keep lambda_web in signature for consistency with existing cosmic-web APIs.
        _ = float(lambda_web)

        conditions = [
            frame["shear_metric"] > 4.5,
            frame["shear_metric"] < 1.5,
        ]
        choices = ["FILAMENT_1D", "VOID_3D"]
        frame["environment"] = np.select(conditions, choices, default="FIELD")

        rng = np.random.default_rng()
        frame["xray_excess_lum"] = np.where(
            frame["environment"] == "FILAMENT_1D",
            rng.normal(2.0e40, 0.5e40, len(frame)),
            rng.normal(5.0e38, 1.0e38, len(frame)),
        )
        return frame

    def fetch(
        self,
        data_key: str,
        source_path: str | os.PathLike[str] | None = None,
        *,
        force_refresh: bool = False,
    ) -> Any:
        """Load data from cache when possible, otherwise from mock/source."""
        cache_file = self.cache_dir / f"{str(data_key).strip()}.pkl"

        if cache_file.is_file() and not force_refresh:
            logger.info("Loading %s from cache.", data_key)
            with cache_file.open("rb") as fh:
                return pickle.load(fh)

        if self.use_mock:
            logger.warning("Generating mock data for %s", data_key)
            data = np.random.normal(0.0, 1.0, (10, 5))
        else:
            logger.info("Fetching real data from %s", source_path)
            data = self._load_from_source(source_path)

        with cache_file.open("wb") as fh:
            pickle.dump(data, fh)
        return data

    def _load_from_source(self, path: str | os.PathLike[str] | None) -> Any:
        """Load supported local formats (.txt/.dat/.csv/.tsv/.npy/.npz/.pkl/.root)."""
        if not path:
            raise FileNotFoundError("Source path not provided.")

        src = Path(path).expanduser().resolve()
        if not src.is_file():
            raise FileNotFoundError(f"Source path {src} not found.")

        suffix = src.suffix.lower()
        if suffix in {".txt", ".dat", ".csv", ".tsv"}:
            delimiter = "\t" if suffix == ".tsv" else "," if suffix == ".csv" else None
            try:
                return np.loadtxt(src, delimiter=delimiter)
            except Exception:
                # genfromtxt is more tolerant of sparse/missing values.
                return np.genfromtxt(src, delimiter=delimiter)

        if suffix == ".npy":
            return np.load(src, allow_pickle=False)

        if suffix == ".npz":
            with np.load(src, allow_pickle=False) as archive:
                keys = list(archive.keys())
                if len(keys) == 1:
                    return archive[keys[0]]
                return {key: archive[key] for key in keys}

        if suffix in {".pkl", ".pickle"}:
            with src.open("rb") as fh:
                return pickle.load(fh)

        if suffix == ".root":
            try:
                import uproot
            except ImportError as exc:
                raise ValueError(
                    "Reading .root files requires uproot. Install with: pip install uproot"
                ) from exc

            with uproot.open(src) as root_file:
                tree_keys = [key for key, val in root_file.items() if hasattr(val, "arrays")]
                if not tree_keys:
                    raise ValueError(f"No TTree found in ROOT file: {src}")
                tree = root_file[tree_keys[0]]
                return tree.arrays(library="np")

        raise ValueError(f"Unsupported source format: {suffix}")

    def _load_real_sparc(self) -> pd.DataFrame:
        """Load and normalize real SPARC catalog rows for cross-domain diagnostics."""
        from menus.astronomical.sparc.fetcher import fetch_mass_models_table

        raw = fetch_mass_models_table()
        if raw is None or raw.empty:
            raise ValueError("SPARC catalog is empty or unavailable.")

        frame = raw.copy()
        # Expected SPARC mass-model columns from the fetcher parser.
        if "ID" not in frame.columns:
            raise ValueError("SPARC catalog missing expected ID column.")

        for col in ("Vobs", "Vgas", "Vdisk", "Vbul"):
            if col not in frame.columns:
                frame[col] = 0.0
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

        v_sq_obs = frame["Vobs"].to_numpy(dtype=float) ** 2
        v_sq_baryon = (
            frame["Vgas"].to_numpy(dtype=float) ** 2
            + frame["Vdisk"].to_numpy(dtype=float) ** 2
            + frame["Vbul"].to_numpy(dtype=float) ** 2
        )
        v_sq_shadow = np.maximum(0.0, v_sq_obs - v_sq_baryon)
        ratio = v_sq_shadow / np.maximum(v_sq_baryon, 1e-6)
        frame["r_shadow_baryon_obs"] = ratio

        grouped = (
            frame.groupby("ID", as_index=False)["r_shadow_baryon_obs"]
            .mean()
            .rename(columns={"ID": "galaxy_id"})
        )

        # Placeholders retained for later enrichments via external cross-match catalogs.
        grouped["delta_phi_est"] = np.nan
        grouped["dist_to_filament_Mpc"] = np.nan
        grouped["xray_excess_lum"] = np.nan

        ordered_cols = [
            "galaxy_id",
            "r_shadow_baryon_obs",
            "delta_phi_est",
            "dist_to_filament_Mpc",
            "xray_excess_lum",
        ]
        out = grouped[ordered_cols]
        out = out.dropna(subset=["galaxy_id", "r_shadow_baryon_obs"])
        if out.empty:
            raise ValueError("SPARC normalization produced no valid rows.")
        return out.reset_index(drop=True)


def get_runtime_data_manager(
    params: dict[str, Any] | None = None,
    *,
    use_mock: bool | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
) -> DataManager:
    """Return a shared runtime DataManager instance for all flow handlers."""
    global _RUNTIME_DATA_MANAGER, _RUNTIME_CONFIG

    cfg = _runtime_config_from_params(params, use_mock=use_mock, cache_dir=cache_dir)
    if _RUNTIME_DATA_MANAGER is None or _RUNTIME_CONFIG != cfg:
        _RUNTIME_DATA_MANAGER = DataManager(use_mock=cfg[0], cache_dir=cfg[1])
        _RUNTIME_CONFIG = cfg
    return _RUNTIME_DATA_MANAGER


def inject_data_manager(params: dict[str, Any] | None) -> dict[str, Any]:
    """Attach the shared runtime DataManager onto action params."""
    out = dict(params or {})
    manager = get_runtime_data_manager(out)
    out.setdefault(DATA_MANAGER_PARAM_KEY, manager)
    out.setdefault(DATA_MANAGER_LEGACY_PARAM_KEY, manager)
    return out