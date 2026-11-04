"""
SPARC rotation-curve analyzer for Tav-Superblock conformal shadow fits.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THEORY_RATIO_TARGET = 5.2
COLUMN_ALIASES = {
    "R": ["R", "Radius", "radius", "r", "kpc"],
    "Vobs": ["Vobs", "Velocity", "velocity", "V_obs", "vobs"],
    "Vgas": ["Vgas", "V_gas", "vgas"],
    "Vdisk": ["Vdisk", "V_disk", "vdisk"],
    "Vbulge": ["Vbulge", "V_bulge", "vbulge"],
    "name": ["name", "Name", "galaxy", "Galaxy", "id", "ID"],
}


class TavDiagnostics:
    """Diagnostic suite for Tav-Superblock framework calibration."""

    def __init__(self, target_ratio: float = THEORY_RATIO_TARGET):
        self.target_ratio = float(target_ratio)

    def calculate_domain_shear(self, obs_ratio: float) -> float:
        """Return absolute shear (eta) from theory-target ratio delta."""
        return float(np.abs(self.target_ratio - float(obs_ratio)))

    def apply_harmonic_damping(
        self,
        gamma_i: float,
        shear_r: float,
        tav_floor: float = 313.1,
    ) -> float:
        """Topological damping factor: gamma_adj = gamma_i * exp(-shear/tav_floor)."""
        damping = np.exp(-float(shear_r) / float(tav_floor))
        return float(float(gamma_i) * damping)


class TauSuperblockSolver:
    def __init__(self, lambda_tav: float = 7.0, r_baseline: float = 5.2):
        self.lambda_tav = float(lambda_tav)
        self.r_baseline = float(r_baseline)

    def compute_local_kappa(self, r_local: np.ndarray) -> np.ndarray:
        """Compute domain coupling coefficient kappa_fil = 1 - exp(-r^2/lambda_tav^2)."""
        r = np.asarray(r_local, dtype=float)
        return 1.0 - np.exp(-(r**2) / (self.lambda_tav**2))

    def simulate_local_kappa(
        self,
        dist_filament: np.ndarray,
        shear_penalty: np.ndarray,
    ) -> np.ndarray:
        """Compute local kappa with geometric coupling and phase-slip shear suppression."""
        dist = np.asarray(dist_filament, dtype=float)
        shear = np.asarray(shear_penalty, dtype=float)

        kappa_base = 1.0 - np.exp(-(dist**2) / (self.lambda_tav**2))
        phase_alignment_factor = np.clip(1.0 - (shear / 5.2), 0.01, 1.0)
        return kappa_base * phase_alignment_factor

    def compute_choked_shadow_ratio(self, kappa_local: np.ndarray) -> np.ndarray:
        """Compute effective shadow/baryon ratio under local choking suppression."""
        kappa = np.asarray(kappa_local, dtype=float)
        return self.r_baseline * kappa

    def compute_effective_ratio(
        self,
        r_local: np.ndarray,
        delta_phi: np.ndarray,
        f_align: float = 1.0,
        epsilon_tep: float = 0.15,
    ) -> np.ndarray:
        """Solve R_eff with flux-choking and TEP phase-slip penalty."""
        kappa = self.compute_local_kappa(r_local)
        phase = np.asarray(delta_phi, dtype=float)
        phase_penalty = np.clip(1.0 - float(epsilon_tep) * (phase / np.pi), 0.0, 1.5)
        return self.r_baseline * kappa * float(f_align) * phase_penalty

    def compute_quench_deficit_heating(
        self,
        e_kinetic: np.ndarray,
        kappa: np.ndarray,
        omega: float = 1.0,
    ) -> np.ndarray:
        """Compute quench-deficit energy Delta E_fluct proxy for WHIM/X-ray excess."""
        e = np.asarray(e_kinetic, dtype=float)
        k = np.asarray(kappa, dtype=float)
        return e * (1.0 - np.exp(-k * float(omega)))


def run_diagnostic_sweep(galaxy_data: dict[str, float]) -> dict[str, dict[str, float]]:
    """
    Run domain-shear diagnostics over observed ratio values keyed by galaxy name.
    """
    diag = TavDiagnostics()
    results: dict[str, dict[str, float]] = {}

    for galaxy, obs_ratio in galaxy_data.items():
        shear = diag.calculate_domain_shear(obs_ratio)
        results[galaxy] = {"shear": shear}
        print(f"[DIAGNOSTIC] Galaxy: {galaxy} | Shear: {shear:.4f}")

    return results


def calibrate_weighting_schema(
    current_weights: dict[str, float],
    shear_map: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Apply topological damping calibration using per-galaxy shear values."""
    diag = TavDiagnostics()
    calibrated_weights: dict[str, float] = {}

    for galaxy, weight in current_weights.items():
        shear = float((shear_map.get(galaxy) or {}).get("shear", 0.0))
        calibrated_weights[galaxy] = diag.apply_harmonic_damping(weight, shear)

    return calibrated_weights


def resolve_sparc_file(query: str, params: Optional[dict] = None) -> str:
    """Resolve a SPARC CSV from galaxy ID, explicit path, or entry-form override."""
    params = params or {}
    override = (params.get("sparc_file") or params.get("data_file") or "").strip()
    if override and os.path.isfile(override):
        return override
    if query.endswith(".csv") and os.path.isfile(query):
        return query
    if os.path.isabs(query) and os.path.isfile(query):
        return query
    try:
        from menus.astronomical.sparc.fetcher import DEFAULT_DATA_DIR

        return str(DEFAULT_DATA_DIR / f"{query}.csv")
    except ImportError:
        return str(Path("datasets/sparc") / f"{query}.csv")


class SPARCAnalyzer:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = self._load_and_normalize(file_path)

    @staticmethod
    def _pick_column(frame: pd.DataFrame, aliases: list[str]) -> Optional[str]:
        for alias in aliases:
            if alias in frame.columns:
                return alias
        return None

    def _load_and_normalize(self, file_path: str) -> pd.DataFrame:
        frame = pd.read_csv(file_path)
        normalized = pd.DataFrame()
        for canonical, aliases in COLUMN_ALIASES.items():
            source = self._pick_column(frame, aliases)
            if source is not None:
                normalized[canonical] = pd.to_numeric(frame[source], errors="coerce")

        if "R" not in normalized or "Vobs" not in normalized:
            raise ValueError(
                "SPARC CSV must include radius and observed velocity columns "
                "(R/Radius and Vobs/Velocity)."
            )

        for component in ("Vgas", "Vdisk", "Vbulge"):
            if component not in normalized:
                normalized[component] = 0.0

        if "name" not in normalized:
            stem = Path(file_path).stem
            normalized["name"] = stem

        normalized = normalized.dropna(subset=["R", "Vobs"]).sort_values("R")
        return normalized.reset_index(drop=True)

    def has_rotation_components(self) -> bool:
        baryon = self.data["Vgas"] ** 2 + self.data["Vdisk"] ** 2 + self.data["Vbulge"] ** 2
        return bool((baryon > 0).any())

    def get_galaxy_frame(self, galaxy_name: str) -> pd.DataFrame:
        if "name" in self.data.columns and self.data["name"].nunique() > 1:
            subset = self.data[self.data["name"].astype(str).str.upper() == galaxy_name.upper()]
            if subset.empty:
                available = ", ".join(sorted(self.data["name"].astype(str).unique())[:8])
                raise ValueError(f"Galaxy '{galaxy_name}' not found. Available: {available}")
            return subset.reset_index(drop=True)
        return self.data.copy()

    def plot_rotation_curve(self, galaxy_name: str, show: bool = True, save_path: Optional[str] = None):
        gal = self.get_galaxy_frame(galaxy_name)

        v_baryonic = np.sqrt(gal["Vgas"] ** 2 + gal["Vdisk"] ** 2 + gal["Vbulge"] ** 2)
        v_shadow = np.sqrt(np.maximum(0.0, gal["Vobs"] ** 2 - v_baryonic ** 2))

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(gal["R"], gal["Vobs"], "k.", label="Observed (Vobs)")
        if self.has_rotation_components():
            ax.plot(gal["R"], gal["Vgas"], "b--", label="Gas")
            ax.plot(gal["R"], gal["Vdisk"], "g--", label="Disk")
            if (gal["Vbulge"] > 0).any():
                ax.plot(gal["R"], gal["Vbulge"], "m--", label="Bulge")
            ax.plot(gal["R"], v_shadow, "r-", linewidth=2, label="Predicted Shadow (SBT)")
        else:
            ax.plot(gal["R"], gal["Vobs"], "r-", linewidth=1.5, label="Observed trend (no baryon split)")

        ax.set_title(f"Rotation Curve: {galaxy_name} - Tav-Superblock Fit")
        ax.set_xlabel("Radius (kpc)")
        ax.set_ylabel("Velocity (km/s)")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    def calculate_mass_ratio(self, galaxy_name: Optional[str] = None) -> float:
        gal = self.get_galaxy_frame(galaxy_name) if galaxy_name else self.data
        v_sq_obs = gal["Vobs"] ** 2
        v_sq_baryon = gal["Vgas"] ** 2 + gal["Vdisk"] ** 2 + gal["Vbulge"] ** 2
        v_sq_shadow = np.maximum(0.0, v_sq_obs - v_sq_baryon)
        ratio = v_sq_shadow / np.maximum(v_sq_baryon, 1e-6)
        return float(ratio.mean())

    def analysis_report(self, galaxy_name: str) -> dict:
        ratio = self.calculate_mass_ratio(galaxy_name)
        delta = abs(ratio - THEORY_RATIO_TARGET)
        gal = self.get_galaxy_frame(galaxy_name)
        v_sq_obs = gal["Vobs"].to_numpy(dtype=float) ** 2
        v_sq_baryon = (
            gal["Vgas"].to_numpy(dtype=float) ** 2
            + gal["Vdisk"].to_numpy(dtype=float) ** 2
            + gal["Vbulge"].to_numpy(dtype=float) ** 2
        )
        v_sq_shadow = np.maximum(0.0, v_sq_obs - v_sq_baryon)
        from tav_shared.bound_unbound_energy import partition_energy

        # V_extra (∝ V_shadow²) has no discrete Tav flag or E_th → AMBIGUOUS.
        # It must not stay in the bound DM ledger (T1/T2/T4 unevaluable; T5 aggregator).
        split = partition_energy(
            v_sq_shadow,
            e_free=v_sq_obs,
            e_th=None,
            is_discrete_harmonic=None,
            tau_bind_ok=None,
            from_total_aggregator=True,
            ledger="deficit",
        )
        return {
            "galaxy": galaxy_name,
            "file": self.file_path,
            "mean_shadow_baryon_ratio": ratio,
            "theory_target_ratio": THEORY_RATIO_TARGET,
            "delta_from_theory": delta,
            "within_10_percent": delta <= THEORY_RATIO_TARGET * 0.1,
            "E_bound_clean": float(split["sum_E_b"]),
            "E_unbound_clean": float(split["sum_E_u"]),
            "bound_flag_clean": bool(split["n_bound"] > 0),
            "f_b": float(split["sum_E_b"] / max(split["sum_E_tot"], 1e-30)),
            "f_u": float(split["sum_E_u"] / max(split["sum_E_tot"], 1e-30)),
            "n_ambiguous": int(split["n_ambiguous"]),
            "energy_ledger": "deficit",
            "note_energy": (
                "mean_shadow_baryon_ratio still uses raw V_shadow²/V_baryon² (E_tot extra). "
                "Without discrete-harmonic / E_th columns those extras are AMBIGUOUS and "
                "are booked as E_unbound_clean, not bound DM."
            ),
        }