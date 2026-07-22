import unittest

import numpy as np
import pandas as pd

from data_manager import DataManager
from theoretical_model import TauSuperblockSolver


class TestFluxChokingOutliers(unittest.TestCase):
    @staticmethod
    def _pick_column(frame: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
        for name in candidates:
            if name in frame.columns:
                return name
        if required:
            raise AssertionError(f"None of required columns found: {candidates}")
        return None

    @classmethod
    def setUpClass(cls):
        cls.dm = DataManager()
        cls.solver = TauSuperblockSolver()
        try:
            cls.df = cls.dm.cross_match_environment_and_xray(cls.dm.load_processed_sparc_batch())
        except (FileNotFoundError, ValueError) as exc:
            raise unittest.SkipTest(f"Processed SPARC batch unavailable for test run: {exc}")

    def test_sparc_175_choking_diagnostic(self):
        """Test 3: Apply choking diagnostic to the processed SPARC array."""
        df = self.df.copy()

        dist_col = self._pick_column(df, ["dist_filament", "dist_to_filament_Mpc"]) 
        galaxy_col = self._pick_column(df, ["galaxy", "galaxy_id", "ID"]) 

        df["kappa_local"] = self.solver.simulate_local_kappa(
            dist_filament=pd.to_numeric(df[dist_col], errors="coerce").fillna(0.0).to_numpy(),
            shear_penalty=pd.to_numeric(df["shear_metric"], errors="coerce").fillna(0.0).to_numpy(),
        )
        df["r_eff_sim"] = self.solver.compute_choked_shadow_ratio(df["kappa_local"].to_numpy())

        choked_targets = ["F574-2", "NGC3953", "NGC0891", "UGC06973"]
        outliers = df[df[galaxy_col].astype(str).isin(choked_targets)]

        if outliers.empty:
            self.skipTest("None of the requested choked targets were present in processed SPARC batch.")

        for _, galaxy in outliers.iterrows():
            self.assertLess(
                float(galaxy["r_eff_sim"]),
                0.1,
                (
                    f"Choking failed for {galaxy[galaxy_col]}. "
                    f"Expected near 0, got {galaxy['r_eff_sim']}"
                ),
            )
            self.assertGreater(
                float(galaxy["shear_metric"]),
                5.0,
                f"Missing topological shear for {galaxy[galaxy_col]}",
            )

    def test_multimessenger_xray_correlation(self):
        """Test 4: Search for correlated X-ray excess around low-ratio hosts."""
        df = self.df.copy()

        shadow_col = self._pick_column(df, ["shadow_ratio", "r_shadow_baryon_obs", "r_eff_sim"]) 

        shadow = pd.to_numeric(df[shadow_col], errors="coerce")
        xray = pd.to_numeric(df["xray_excess_lum"], errors="coerce")
        valid = shadow.notna() & xray.notna()
        df = df.loc[valid].copy()
        shadow = shadow.loc[valid]
        xray = xray.loc[valid]

        choked_regime = df[shadow < 0.5]
        coupled_regime = df[shadow >= 2.0]

        if choked_regime.empty or coupled_regime.empty:
            self.skipTest("Insufficient shadow-ratio separation for X-ray correlation assertion.")

        mean_choked_xray = pd.to_numeric(choked_regime["xray_excess_lum"], errors="coerce").mean()
        mean_coupled_xray = pd.to_numeric(coupled_regime["xray_excess_lum"], errors="coerce").mean()

        self.assertTrue(np.isfinite(mean_choked_xray))
        self.assertTrue(np.isfinite(mean_coupled_xray))
        self.assertGreater(
            mean_choked_xray,
            mean_coupled_xray * 5.0,
            "Failed: Flux-choking does not produce the expected correlated X-ray/WHIM excess.",
        )


if __name__ == "__main__":
    unittest.main()
