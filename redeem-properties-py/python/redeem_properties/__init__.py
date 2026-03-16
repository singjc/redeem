"""
redeem_properties
====================
Python bindings for the redeem-properties Rust crate, exposing peptide property
prediction models for retention time (RT), collisional cross-section (CCS), and
MS2 fragment intensities.

All three model classes delegate inference to the compiled Rust extension
(``_lib``).  The ``predict_df`` convenience methods additionally return results
as a ``pandas`` or ``polars`` DataFrame.

Quick start
-----------
>>> import redeem_properties as rp
>>>
>>> # Download pretrained models (only needed once)
>>> rp.download_pretrained_models()
>>>
>>> rt_model  = rp.RTModel.from_pretrained("rt")
>>> ccs_model = rp.CCSModel.from_pretrained("ccs")
>>> ms2_model = rp.MS2Model.from_pretrained("ms2")
>>>
>>> # numpy arrays / list[dict]
>>> rt_values   = rt_model.predict(["PEPTIDE", "SEQU[+42.0106]ENCE"])
>>> ccs_results = ccs_model.predict(["PEPTIDE"], charges=[2])
>>> ms2_results = ms2_model.predict(["PEPTIDE"], charges=[2], nces=[20])
>>>
>>> # pandas DataFrames
>>> rt_df  = rt_model.predict_df(["PEPTIDE", "SEQU[+42.0106]ENCE"])
>>> ccs_df = ccs_model.predict_df(["PEPTIDE"], charges=[2])
>>> ms2_df = ms2_model.predict_df(["PEPTIDE"], charges=[2], nces=[20])
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Optional

from redeem_properties._lib import (  # noqa: F401  (re-exported)
    CCSModel as _CCSLib,
    MS2Model as _MS2Lib,
    RTModel as _RTLib,
    locate_pretrained as locate_pretrained,
    validate_pretrained as validate_pretrained,
    download_pretrained_models as download_pretrained_models,
    compute_precursor_mz as compute_precursor_mz,
    compute_fragment_mzs as compute_fragment_mzs,
    compute_peptide_mz_info as compute_peptide_mz_info,
    match_fragment_mzs as match_fragment_mzs,
    ccs_to_mobility as ccs_to_mobility,
)

__all__ = [
    "__version__",
    "RTModel",
    "CCSModel",
    "MS2Model",
    "PropertyPrediction",
    "locate_pretrained",
    "download_pretrained_models",
    "compute_precursor_mz",
    "compute_fragment_mzs",
    "compute_peptide_mz_info",
    "match_fragment_mzs",
    "ccs_to_mobility",
]


def _detect_version() -> str:
    """Return the installed package version for redeem_properties."""
    for dist_name in ("redeem_properties", "redeem-properties-py"):
        try:
            return version(dist_name)
        except PackageNotFoundError:
            continue
    return "0+unknown"


__version__ = _detect_version()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_df(data: dict, framework: str):
    """Build a DataFrame from a column dict using *framework* ('pandas'/'polars')."""
    if framework == "pandas":
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas is required for predict_df(framework='pandas'). "
                "Install it with: pip install redeem-properties-py[pandas]"
            ) from exc
        return pd.DataFrame(data)
    elif framework == "polars":
        try:
            import polars as pl
        except ImportError as exc:
            raise ImportError(
                "polars is required for predict_df(framework='polars'). "
                "Install it with: pip install redeem-properties-py[polars]"
            ) from exc
        return pl.DataFrame(data)
    else:
        raise ValueError(f"Unknown framework '{framework}'. Use 'pandas' or 'polars'.")


def _expand_inputs(
    peptides: list[str],
    charges: int | list[int] | None = None,
    nces: int | float | list[int] | list[float] | None = None,
    instruments: str | list[Optional[str]] | None = None,
) -> tuple[list[str], list[int] | None, list[int] | None, list[Optional[str]] | None]:
    """
    Expands peptides and charges via Cartesian product if lengths differ.
    Broadcasts nces and instruments to match the expanded length.
    """
    # 1. Expand peptides and charges
    if charges is None:
        exp_pep = list(peptides)
        exp_charge = None
    elif isinstance(charges, int):
        exp_pep = list(peptides)
        exp_charge = [charges] * len(peptides)
    elif isinstance(charges, list):
        if len(charges) == len(peptides):
            # Assume 1:1 mapping
            exp_pep = list(peptides)
            exp_charge = list(charges)
        else:
            # Cartesian product
            exp_pep = []
            exp_charge = []
            for p in peptides:
                for c in charges:
                    exp_pep.append(p)
                    exp_charge.append(c)
    else:
        raise TypeError("charges must be an int or a list of ints")

    n = len(exp_pep)

    # 2. Broadcast nces
    exp_nces = None
    if nces is not None:
        if isinstance(nces, (int, float)):
            exp_nces = [int(nces)] * n
        elif isinstance(nces, list):
            if len(nces) == 1:
                exp_nces = [int(nces[0])] * n
            elif len(nces) == n:
                exp_nces = [int(x) for x in nces]
            else:
                raise ValueError(
                    f"nces must be a single value, a list of length 1, or match the expanded length {n}"
                )
        else:
            raise TypeError("nces must be an int, float, or list")

    # 3. Broadcast instruments
    exp_inst = None
    if instruments is not None:
        if isinstance(instruments, str):
            exp_inst = [instruments] * n
        elif isinstance(instruments, list):
            if len(instruments) == 1:
                exp_inst = [instruments[0]] * n
            elif len(instruments) == n:
                exp_inst = instruments
            else:
                raise ValueError(
                    f"instruments must be a single value, a list of length 1, or match the expanded length {n}"
                )
        else:
            raise TypeError("instruments must be a string or list")

    return exp_pep, exp_charge, exp_nces, exp_inst


# ---------------------------------------------------------------------------
# RTModel
# ---------------------------------------------------------------------------


class RTModel:
    """Retention time prediction model.

    Parameters
    ----------
    model_path:
        Path to the ``.pth`` model weights file.
    arch:
        Model architecture string (e.g. ``"rt_cnn_lstm"``).
    constants_path:
        Optional path to the ``.yaml`` constants file.
    use_cuda:
        Whether to run inference on GPU (requires CUDA build). Default ``False``.
    """

    def __init__(
        self,
        model_path: str,
        arch: str,
        constants_path: Optional[str] = None,
        use_cuda: bool = False,
    ) -> None:
        self._inner = _RTLib(
            model_path, arch, constants_path=constants_path, use_cuda=use_cuda
        )
        # preserve construction metadata for nicer repr/str
        self._model_path = model_path
        self._arch = arch

    @classmethod
    def from_pretrained(cls, name: str, use_cuda: bool = False) -> "RTModel":
        """Load an RTModel from the shipped pretrained weights.

        Accepted *name* values (case-insensitive):
        ``"rt"``, ``"alphapeptdeep-rt-cnn-lstm"``, ``"redeem-rt-cnn-tf"``.

        Parameters
        ----------
        name:
            Pretrained model identifier.
        use_cuda:
            Whether to run inference on GPU. Default ``False``.
        """
        obj = cls.__new__(cls)
        obj._inner = _RTLib.from_pretrained(name, use_cuda=use_cuda)
        # remember the requested pretrained name and located path (best-effort)
        obj._requested_name = name
        try:
            obj._model_path = locate_pretrained(name)
        except Exception:
            obj._model_path = None
        obj._arch = None
        return obj

    def __repr__(self) -> str:
        arch = getattr(self, "_arch", None) or getattr(self, "_requested_name", None)
        path = getattr(self, "_model_path", None)
        param_count = (
            self.param_count() if hasattr(self._inner, "param_count") else "unknown"
        )
        return f"<RTModel arch={arch!r} params={param_count} path={path!r}>"

    def __str__(self) -> str:
        return self.__repr__()

    def predict(self, peptides: list[str]):
        """Predict retention times for a list of peptides.

        Peptides may contain inline modification annotations
        (``[+X.X]`` mass-shift or ``(UniMod:N)`` notation).

        Parameters
        ----------
        peptides:
            List of peptide sequences.

        Returns
        -------
        numpy.ndarray
            1-D float32 array of predicted RT values, one per peptide.
        """
        return self._inner.predict(peptides)

    def param_count(self) -> int:
        """Return total number of parameters in the loaded model (if available).

        This delegates to the compiled Rust extension when present. If the
        underlying extension does not expose a param_count method an
        AttributeError is raised.
        """
        if hasattr(self._inner, "param_count"):
            try:
                return self._inner.param_count()
            except Exception as e:
                raise RuntimeError(f"failed to get param_count from inner model: {e}")
        raise AttributeError("underlying model does not expose 'param_count'")

    def summary(self) -> str:
        """Return a compact/detailed model summary string delegated to the Rust extension.

        Prefer the detailed Rust-side summary when available.
        """
        # Prefer pretty hierarchical summary if available
        if hasattr(self._inner, "summary_pretty"):
            try:
                return self._inner.summary_pretty()
            except Exception:
                pass
        if hasattr(self._inner, "summary"):
            try:
                return self._inner.summary()
            except Exception:
                pass
        # Fallback: try a compact repr using arch/requested name
        arch = getattr(self, "_arch", None) or getattr(self, "_requested_name", None)
        try:
            # if param_count available, include it
            pc = self.param_count() if hasattr(self._inner, "param_count") else None
            if pc is not None:
                return f"{arch} params={pc}"
        except Exception:
            pass
        return f"{arch}"

    def predict_df(self, peptides: list[str], framework: str = "pandas"):
        """Predict retention times and return the result as a DataFrame.

        Parameters
        ----------
        peptides:
            List of peptide sequences (inline modifications supported).
        framework:
            ``'pandas'`` (default) or ``'polars'``.

        Returns
        -------
        pandas.DataFrame or polars.DataFrame
            Columns: ``peptide`` (str), ``rt`` (float32).
        """
        rt_values = self.predict(peptides)
        return _make_df({"peptide": peptides, "rt": rt_values}, framework)


# ---------------------------------------------------------------------------
# CCSModel
# ---------------------------------------------------------------------------


class CCSModel:
    """Collisional cross-section prediction model.

    Parameters
    ----------
    model_path:
        Path to the ``.pth`` model weights file.
    arch:
        Model architecture string (e.g. ``"ccs_cnn_lstm"``).
    constants_path:
        Path to the ``.yaml`` constants file (required).
    use_cuda:
        Whether to run inference on GPU. Default ``False``.
    """

    def __init__(
        self,
        model_path: str,
        arch: str,
        constants_path: Optional[str] = None,
        use_cuda: bool = False,
    ) -> None:
        self._inner = _CCSLib(model_path, arch, constants_path, use_cuda=use_cuda)
        # preserve construction metadata for nicer repr/str
        self._model_path = model_path
        self._arch = arch

    @classmethod
    def from_pretrained(cls, name: str, use_cuda: bool = False) -> "CCSModel":
        """Load a CCSModel from the shipped pretrained weights.

        Accepted *name* values (case-insensitive):
        ``"ccs"``, ``"alphapeptdeep-ccs-cnn-lstm"``, ``"redeem-ccs-cnn-tf"``.
        """
        obj = cls.__new__(cls)
        obj._inner = _CCSLib.from_pretrained(name, use_cuda=use_cuda)
        obj._requested_name = name
        try:
            obj._model_path = locate_pretrained(name)
        except Exception:
            obj._model_path = None
        obj._arch = None
        return obj

    def __repr__(self) -> str:
        arch = getattr(self, "_arch", None) or getattr(self, "_requested_name", None)
        path = getattr(self, "_model_path", None)
        param_count = (
            self.param_count() if hasattr(self._inner, "param_count") else "unknown"
        )
        return f"<CCSModel arch={arch!r} params={param_count} path={path!r}>"

    def __str__(self) -> str:
        return self.__repr__()

    def predict(self, peptides: list[str], charges: int | list[int]):
        """Predict CCS values for a list of peptides.

        Parameters
        ----------
        peptides:
            List of peptide sequences (inline modifications supported).
        charges:
            Charge state per peptide. If a single integer is provided,
            it is broadcast to all peptides. If a list of charges is provided
            and its length differs from the number of peptides, a Cartesian
            product is performed (predicting each peptide at each charge state).

        Returns
        -------
        list[dict]
            One dict per peptide with keys:

            * ``"ccs"`` – predicted CCS value (Å²).
            * ``"charge"`` – charge state used for the prediction.
        """
        peptides, exp_charges, _, _ = _expand_inputs(peptides, charges=charges)
        return self._inner.predict(peptides, exp_charges)

    def param_count(self) -> int:
        """Return total number of parameters in the loaded model (if available)."""
        if hasattr(self._inner, "param_count"):
            try:
                return self._inner.param_count()
            except Exception as e:
                raise RuntimeError(f"failed to get param_count from inner model: {e}")
        raise AttributeError("underlying model does not expose 'param_count'")

    def summary(self) -> str:
        """Return a model summary string delegated to the Rust extension.

        Prefers the pretty hierarchical summary when available.
        """
        # Prefer pretty hierarchical summary if available
        if hasattr(self._inner, "summary_pretty"):
            try:
                return self._inner.summary_pretty()
            except Exception:
                pass
        if hasattr(self._inner, "summary"):
            try:
                return self._inner.summary()
            except Exception:
                pass
        # Fallback: compact repr using arch/requested name
        arch = getattr(self, "_arch", None) or getattr(self, "_requested_name", None)
        try:
            pc = self.param_count() if hasattr(self._inner, "param_count") else None
            if pc is not None:
                return f"{arch} params={pc}"
        except Exception:
            pass
        return f"{arch}"

    def predict_df(
        self,
        peptides: list[str],
        charges: int | list[int],
        annotate_mobility: bool = False,
        framework: str = "pandas",
    ):
        """Predict CCS values and return the result as a DataFrame.

        Parameters
        ----------
        peptides:
            List of peptide sequences (inline modifications supported).
        charges:
            Charge state per peptide. If a single integer is provided,
            it is broadcast to all peptides. If a list of charges is provided
            and its length differs from the number of peptides, a Cartesian
            product is performed (predicting each peptide at each charge state).
        annotate_mobility:
            If ``True``, compute and append an ``ion_mobility`` column
            converted from the predicted CCS value. Default ``False``.
        framework:
            ``'pandas'`` (default) or ``'polars'``.

        Returns
        -------
        pandas.DataFrame or polars.DataFrame
            Columns: ``peptide`` (str), ``ccs`` (float32), ``charge`` (int),
            and optionally ``ion_mobility`` (float).
        """
        peptides, exp_charges, _, _ = _expand_inputs(peptides, charges=charges)
        results = self.predict(peptides, exp_charges)  # type: ignore

        data = {
            "peptide": peptides,
            "ccs": [r["ccs"] for r in results],
            "charge": [r["charge"] for r in results],
        }

        if annotate_mobility:
            mobility_col = []
            for pep, ch, res in zip(peptides, exp_charges, results):  # type: ignore
                try:
                    mz = compute_precursor_mz(pep, ch)
                    mob = ccs_to_mobility(res["ccs"], float(ch), mz)
                    mobility_col.append(mob)
                except Exception:
                    mobility_col.append(float("nan"))
            data["ion_mobility"] = mobility_col

        return _make_df(data, framework)


# ---------------------------------------------------------------------------
# MS2Model
# ---------------------------------------------------------------------------


class MS2Model:
    """MS2 fragment intensity prediction model.

    Parameters
    ----------
    model_path:
        Path to the ``.pth`` model weights file.
    arch:
        Model architecture string (e.g. ``"ms2_bert"``).
    constants_path:
        Path to the ``.yaml`` constants file (required).
    use_cuda:
        Whether to run inference on GPU. Default ``False``.
    """

    def __init__(
        self,
        model_path: str,
        arch: str,
        constants_path: Optional[str] = None,
        use_cuda: bool = False,
    ) -> None:
        self._inner = _MS2Lib(model_path, arch, constants_path, use_cuda=use_cuda)
        # preserve construction metadata for nicer repr/str
        self._model_path = model_path
        self._arch = arch

    @classmethod
    def from_pretrained(cls, name: str, use_cuda: bool = False) -> "MS2Model":
        """Load an MS2Model from the shipped pretrained weights.

        Accepted *name* values (case-insensitive):
        ``"ms2"``, ``"alphapeptdeep-ms2-bert"``.
        """
        obj = cls.__new__(cls)
        obj._inner = _MS2Lib.from_pretrained(name, use_cuda=use_cuda)
        obj._requested_name = name
        try:
            obj._model_path = locate_pretrained(name)
        except Exception:
            obj._model_path = None
        obj._arch = None
        return obj

    def __repr__(self) -> str:
        arch = getattr(self, "_arch", None) or getattr(self, "_requested_name", None)
        path = getattr(self, "_model_path", None)
        param_count = (
            self.param_count() if hasattr(self._inner, "param_count") else "unknown"
        )
        return f"<MS2Model arch={arch!r} params={param_count} path={path!r}>"

    def __str__(self) -> str:
        return self.__repr__()

    def predict(
        self,
        peptides: list[str],
        charges: int | list[int],
        nces: int | float | list[int] | list[float],
        instruments: str | list[Optional[str]] | None = None,
        multiplier: float = 10_000.0,
    ):
        """Predict MS2 fragment intensities for a list of peptides.

        Parameters
        ----------
        peptides:
            List of peptide sequences (inline modifications supported).
        charges:
            Charge state per peptide. If a single integer is provided,
            it is broadcast to all peptides. If a list of charges is provided
            and its length differs from the number of peptides, a Cartesian
            product is performed (predicting each peptide at each charge state).
        nces:
            Normalized collision energy per peptide. Can be a single value
            (broadcast to all) or a list matching the expanded length.
        instruments:
            Instrument name per peptide (optional). Can be a single string
            (broadcast to all) or a list matching the expanded length.
        multiplier:
            Scalar to multiply predicted intensities by (default 10_000.0). Use e.g.
            ``10000.0`` to scale normalized outputs into typical intensity ranges.

        Returns
        -------
        list[dict]
            One dict per peptide with keys:

            * ``"intensities"`` – 2-D float32 array ``(n_positions, 8)``.
            * ``"ion_types"`` – list of 8 ion-type strings.
            * ``"ion_charges"`` – list of 8 fragment charge integers.
            * ``"b_ordinals"`` – 1-D int array ``[1, …, n_positions]``.
            * ``"y_ordinals"`` – 1-D int array ``[n_positions, …, 1]``.
        """
        peptides, exp_charges, exp_nces, exp_inst = _expand_inputs(
            peptides, charges=charges, nces=nces, instruments=instruments
        )
        results = self._inner.predict(
            peptides, exp_charges, exp_nces, instruments=exp_inst
        )

        # Optionally scale predicted intensities by a multiplier before returning.
        if multiplier is not None and multiplier != 1.0:
            try:
                for r in results:
                    # r["intensities"] is a numpy.ndarray; multiply in-place for efficiency
                    r_int = r.get("intensities")
                    if r_int is not None:
                        r_int *= multiplier
            except Exception:
                # If in-place scaling fails for some reason, fall back to non-mutating scaling
                scaled = []
                for r in results:
                    r2 = dict(r)
                    arr = r2.get("intensities")
                    if arr is not None:
                        r2["intensities"] = arr * multiplier
                    scaled.append(r2)
                results = scaled

        return results

    def param_count(self) -> int:
        """Return total number of parameters in the loaded model (if available)."""
        if hasattr(self._inner, "param_count"):
            try:
                return self._inner.param_count()
            except Exception as e:
                raise RuntimeError(f"failed to get param_count from inner model: {e}")
        raise AttributeError("underlying model does not expose 'param_count'")

    def summary(self) -> str:
        """Return a model summary string delegated to the Rust extension.

        Prefers the pretty hierarchical summary when available.
        """
        # Prefer pretty hierarchical summary if available
        if hasattr(self._inner, "summary_pretty"):
            try:
                return self._inner.summary_pretty()
            except Exception:
                pass
        if hasattr(self._inner, "summary"):
            try:
                return self._inner.summary()
            except Exception:
                pass
        # Fallback: compact repr using arch/requested name
        arch = getattr(self, "_arch", None) or getattr(self, "_requested_name", None)
        try:
            pc = self.param_count() if hasattr(self._inner, "param_count") else None
            if pc is not None:
                return f"{arch} params={pc}"
        except Exception:
            pass
        return f"{arch}"

    def predict_df(
        self,
        peptides: list[str],
        charges: int | list[int],
        nces: int | float | list[int] | list[float],
        instruments: str | list[Optional[str]] | None = None,
        multiplier: float = 10_000.0,
        exclude_zeros: bool = True,
        annotate_mz: bool = False,
        framework: str = "pandas",
    ):
        """Predict MS2 fragment intensities and return a long-format DataFrame.

        Each row represents one (peptide, ion_type, fragment_charge, ordinal) combination.

        Parameters
        ----------
        peptides:
            List of peptide sequences (inline modifications supported).
        charges:
            Precursor charge state per peptide. If a single integer is provided,
            it is broadcast to all peptides. If a list of charges is provided
            and its length differs from the number of peptides, a Cartesian
            product is performed (predicting each peptide at each charge state).
        nces:
            Normalized collision energy per peptide. Can be a single value
            (broadcast to all) or a list matching the expanded length.
        instruments:
            Instrument name per peptide (optional). Can be a single string
            (broadcast to all) or a list matching the expanded length.
        multiplier:
            Scalar to multiply predicted intensities by (default 10_000.0). Use e.g.
            ``10000.0`` to scale normalized outputs into typical intensity ranges.
        exclude_zeros:
            If True, exclude rows where all predicted intensities are zero.
        annotate_mz:
            If ``True``, append a ``mz`` column with the theoretical
            monoisotopic m/z for each fragment ion (computed via *rustyms*).
            Neutral-loss ions (``b_nl``, ``y_nl``) receive ``NaN``.
            Default ``False``.
        framework:
            ``'pandas'`` (default) or ``'polars'``.

        Returns
        -------
        pandas.DataFrame or polars.DataFrame
            Columns: ``peptide``, ``ion_type``, ``fragment_charge``,
            ``ordinal``, ``intensity``, and optionally ``mz``.

        Example
        -------
        >>> df = ms2_model.predict_df(
        ...     ["AGHCEWQMKYR"],
        ...     charges=[2], nces=[20], instruments=["QE"],
        ... )
        >>> df.head()
           peptide ion_type  fragment_charge  ordinal  intensity
        0  AGHCEWQMKYR       b                1        1      0.123
        1  AGHCEWQMKYR       b                2        1      0.045
        ...
        """
        peptides, exp_charges, exp_nces, exp_inst = _expand_inputs(
            peptides, charges=charges, nces=nces, instruments=instruments
        )
        results = self.predict(
            peptides,
            exp_charges,
            exp_nces,
            instruments=exp_inst,
            multiplier=multiplier,  # type: ignore
        )

        b_ion_types = {"b", "b_nl"}
        pep_col: list[str] = []
        ion_type_col: list[str] = []
        frag_charge_col: list[int] = []
        ordinal_col: list[int] = []
        intensity_col: list[float] = []
        mz_col: list[float] = []
        precursor_mz_col: list[float] = []

        for pep, charge, res in zip(peptides, exp_charges, results):  # type: ignore
            intensities = res["intensities"]
            ion_types = res["ion_types"]
            frag_charges = res["ion_charges"]
            b_ords = res["b_ordinals"]
            y_ords = res["y_ordinals"]
            n_pos, n_types = intensities.shape

            # Pre-compute theoretical fragment m/z for this peptide (if needed)
            frag_mz_lookup: dict[tuple[str, int, int], float] | None = None
            _precursor_mz = float("nan")
            if annotate_mz:
                max_frag_charge = max(int(fc) for fc in frag_charges)
                try:
                    _precursor_mz = compute_precursor_mz(pep, charge)
                    frag_info = compute_fragment_mzs(pep, max_frag_charge)
                    frag_mz_lookup = {
                        (t, c, o): m
                        for t, c, o, m in zip(
                            frag_info["ion_types"],
                            frag_info["charges"],
                            frag_info["ordinals"],
                            frag_info["mzs"],
                        )
                    }
                except Exception:
                    frag_mz_lookup = {}

            for r in range(n_pos):
                for c in range(n_types):
                    t = ion_types[c]
                    ordinal = int(b_ords[r]) if t in b_ion_types else int(y_ords[r])
                    val = float(intensities[r, c])
                    # If requested, skip individual ion rows with zero intensity.
                    if exclude_zeros and val == 0.0:
                        continue
                    pep_col.append(pep)
                    ion_type_col.append(t)
                    frag_charge_col.append(int(frag_charges[c]))
                    ordinal_col.append(ordinal)
                    intensity_col.append(val)
                    if annotate_mz:
                        precursor_mz_col.append(_precursor_mz)
                        if frag_mz_lookup is not None:
                            # Strip _nl suffix for lookup; NL ions won't match → NaN
                            base_type = t.replace("_nl", "")
                            mz_col.append(
                                frag_mz_lookup.get(
                                    (base_type, int(frag_charges[c]), ordinal),
                                    float("nan"),
                                )
                            )
                        else:
                            mz_col.append(float("nan"))

        data: dict = {
            "peptide": pep_col,
            "ion_type": ion_type_col,
            "fragment_charge": frag_charge_col,
            "ordinal": ordinal_col,
            "intensity": intensity_col,
        }
        if annotate_mz:
            data["precursor_mz"] = precursor_mz_col
            data["mz"] = mz_col

        return _make_df(data, framework)


# ---------------------------------------------------------------------------
# PropertyPrediction  – unified RT + CCS + MS2 predictor
# ---------------------------------------------------------------------------


class PropertyPrediction:
    """Unified peptide property predictor combining RT, CCS, and MS2 models.

    Each model is **optional**.  When a model is ``None`` its columns are
    omitted from the output.  By default the constructor loads the shipped
    pretrained weights for all three models; pass ``predict_rt=False``,
    ``predict_ccs=False``, or ``predict_ms2=False`` to skip a model entirely.

    Parameters
    ----------
    rt_model:
        An :class:`RTModel` instance, or ``None`` to skip RT prediction.
        Ignored when *predict_rt* is ``False``.
    ccs_model:
        A :class:`CCSModel` instance, or ``None`` to skip CCS prediction.
        Ignored when *predict_ccs* is ``False``.
    ms2_model:
        An :class:`MS2Model` instance, or ``None`` to skip MS2 prediction.
        Ignored when *predict_ms2* is ``False``.
    predict_rt:
        Whether to include retention-time predictions. Default ``True``.
    predict_ccs:
        Whether to include CCS predictions. Default ``True``.
    predict_ms2:
        Whether to include MS2 fragment-intensity predictions. Default ``True``.
    use_cuda:
        Forwarded to ``from_pretrained`` when constructing default models.
        Default ``False``.

    Examples
    --------
    >>> import redeem_properties as rp
    >>> prop = rp.PropertyPrediction()          # all three pretrained models
    >>> df = prop.predict_df(
    ...     ["PEPTIDE", "AGHCEWQMKYR"],
    ...     charges=[2, 2], nces=[20, 20], instruments=["QE", "QE"],
    ... )
    >>> df.columns.tolist()
    ['peptide', 'charge', 'nce', 'instrument', 'rt', 'ccs',
     'ion_type', 'fragment_charge', 'ordinal', 'intensity']

    Only RT + CCS (skip MS2):

    >>> prop = rp.PropertyPrediction(predict_ms2=False)
    >>> df = prop.predict_df(["PEPTIDE"], charges=[2])
    """

    def __init__(
        self,
        rt_model: Optional[RTModel] = None,
        ccs_model: Optional[CCSModel] = None,
        ms2_model: Optional[MS2Model] = None,
        *,
        predict_rt: bool = True,
        predict_ccs: bool = True,
        predict_ms2: bool = True,
        use_cuda: bool = False,
    ) -> None:
        # ------ RT ------
        if predict_rt:
            if rt_model is not None:
                self.rt_model: Optional[RTModel] = rt_model
            else:
                try:
                    self.rt_model = RTModel.from_pretrained("rt", use_cuda=use_cuda)
                except Exception:
                    self.rt_model = None
        else:
            self.rt_model = None

        # ------ CCS ------
        if predict_ccs:
            if ccs_model is not None:
                self.ccs_model: Optional[CCSModel] = ccs_model
            else:
                try:
                    self.ccs_model = CCSModel.from_pretrained("ccs", use_cuda=use_cuda)
                except Exception:
                    self.ccs_model = None
        else:
            self.ccs_model = None

        # ------ MS2 ------
        if predict_ms2:
            if ms2_model is not None:
                self.ms2_model: Optional[MS2Model] = ms2_model
            else:
                try:
                    self.ms2_model = MS2Model.from_pretrained("ms2", use_cuda=use_cuda)
                except Exception:
                    self.ms2_model = None
        else:
            self.ms2_model = None

    # -----------------------------------------------------------------
    def __repr__(self) -> str:
        parts = []
        if self.rt_model is not None:
            rt_arch = getattr(self.rt_model, "_arch", None) or getattr(
                self.rt_model, "_requested_name", None
            )
            rt_params = (
                self.rt_model.param_count()
                if hasattr(self.rt_model._inner, "param_count")
                else "unknown"
            )
            rt_path = getattr(self.rt_model, "_model_path", None)
            parts.append(f"\nrt={rt_arch!r} params={rt_params} path={rt_path!r}")
        if self.ccs_model is not None:
            ccs_arch = getattr(self.ccs_model, "_arch", None) or getattr(
                self.ccs_model, "_requested_name", None
            )
            ccs_params = (
                self.ccs_model.param_count()
                if hasattr(self.ccs_model._inner, "param_count")
                else "unknown"
            )
            ccs_path = getattr(self.ccs_model, "_model_path", None)
            parts.append(f"\nccs={ccs_arch!r} params={ccs_params} path={ccs_path!r}")
        if self.ms2_model is not None:
            ms2_arch = getattr(self.ms2_model, "_arch", None) or getattr(
                self.ms2_model, "_requested_name", None
            )
            ms2_params = (
                self.ms2_model.param_count()
                if hasattr(self.ms2_model._inner, "param_count")
                else "unknown"
            )
            ms2_path = getattr(self.ms2_model, "_model_path", None)
            parts.append(f"\nms2={ms2_arch!r} params={ms2_params} path={ms2_path!r}")
        return f"<PropertyPrediction {' '.join(parts)}\n>"

    def __str__(self) -> str:
        return self.__repr__()

    # -----------------------------------------------------------------
    def predict(
        self,
        peptides: list[str],
        charges: int | list[int] | None = None,
        nces: int | float | list[int] | list[float] | None = None,
        instruments: str | list[Optional[str]] | None = None,
        multiplier: float = 10_000.0,
    ) -> dict:
        """Run enabled models and return raw results in a dict.

        Parameters
        ----------
        peptides:
            List of peptide sequences (inline modifications supported).
        charges:
            Charge state per peptide (required for CCS and MS2). If a single
            integer is provided, it is broadcast to all peptides. If a list of
            charges is provided and its length differs from the number of peptides,
            a Cartesian product is performed.
        nces:
            Normalized collision energy per peptide (required for MS2). Can be
            a single value (broadcast to all) or a list matching the expanded length.
        instruments:
            Instrument name per peptide (optional, used by MS2). Can be a single
            string (broadcast to all) or a list matching the expanded length.
        multiplier:
            Scalar applied to MS2 predicted intensities (default 10 000).

        Returns
        -------
        dict
            Keys that may be present: ``"rt"`` (1-D ndarray), ``"ccs"``
            (list[dict]), ``"ms2"`` (list[dict]).
        """
        peptides, exp_charges, exp_nces, exp_inst = _expand_inputs(
            peptides, charges=charges, nces=nces, instruments=instruments
        )
        out: dict = {}

        if self.rt_model is not None:
            out["rt"] = self.rt_model.predict(peptides)

        if self.ccs_model is not None:
            if exp_charges is None:
                raise ValueError("charges are required for CCS prediction")
            out["ccs"] = self.ccs_model.predict(peptides, exp_charges)

        if self.ms2_model is not None:
            if exp_charges is None:
                raise ValueError("charges are required for MS2 prediction")
            if exp_nces is None:
                raise ValueError("nces are required for MS2 prediction")
            out["ms2"] = self.ms2_model.predict(
                peptides,
                exp_charges,
                exp_nces,
                instruments=exp_inst,
                multiplier=multiplier,
            )

        return out

    # -----------------------------------------------------------------
    def predict_df(
        self,
        peptides: list[str],
        charges: int | list[int] | None = None,
        nces: int | float | list[int] | list[float] | None = None,
        instruments: str | list[Optional[str]] | None = None,
        multiplier: float = 10_000.0,
        exclude_zeros: bool = True,
        annotate_mz: bool = True,
        annotate_mobility: bool = False,
        framework: str = "pandas",
    ):
        """Predict all enabled properties and return a single long-format DataFrame.

        When MS2 is enabled every fragment row is emitted; the scalar RT and CCS
        values are broadcast (repeated) across those rows so that each row is
        fully self-contained.

        When MS2 is **disabled** the DataFrame contains one row per peptide
        with only the scalar columns that are enabled.

        Parameters
        ----------
        peptides:
            List of peptide sequences (inline modifications supported).
        charges:
            Charge state per peptide (required for CCS and MS2). If a single
            integer is provided, it is broadcast to all peptides. If a list of
            charges is provided and its length differs from the number of peptides,
            a Cartesian product is performed.
        nces:
            Normalized collision energy per peptide (required for MS2). Can be
            a single value (broadcast to all) or a list matching the expanded length.
        instruments:
            Instrument name per peptide (optional, used by MS2). Can be a single
            string (broadcast to all) or a list matching the expanded length.
        multiplier:
            Scalar applied to MS2 predicted intensities (default 10 000).
        exclude_zeros:
            If ``True``, individual zero-intensity fragment rows are dropped.
        annotate_mz:
            If ``True`` (default), compute and add m/z columns.  When MS2 is
            enabled a ``precursor_mz`` column and a per-fragment ``mz``
            column are added.  When MS2 is disabled only ``precursor_mz``
            is added.  Requires *charges* to be provided.
        annotate_mobility:
            If ``True``, compute and append an ``ion_mobility`` column
            converted from the predicted CCS value. Requires *charges* and
            the CCS model to be enabled. Default ``False``.
        framework:
            ``'pandas'`` (default) or ``'polars'``.

        Returns
        -------
        pandas.DataFrame or polars.DataFrame
            Possible columns (depending on which models are enabled):
            ``peptide``, ``charge``, ``nce``, ``instrument``,
            ``rt``, ``ccs``, ``ion_mobility``, ``precursor_mz``, ``ion_type``,
            ``fragment_charge``, ``ordinal``, ``intensity``, ``mz``.
        """
        peptides, exp_charges, exp_nces, exp_inst = _expand_inputs(
            peptides, charges=charges, nces=nces, instruments=instruments
        )
        n = len(peptides)

        # -- scalar predictions (RT / CCS) ---------------------------------
        rt_values = None
        if self.rt_model is not None:
            rt_values = self.rt_model.predict(peptides)  # 1-D ndarray

        ccs_values = None
        if self.ccs_model is not None:
            if exp_charges is None:
                raise ValueError("charges are required for CCS prediction")
            ccs_results = self.ccs_model.predict(peptides, exp_charges)
            ccs_values = [r["ccs"] for r in ccs_results]  # list[float]

        # -- MS2 fragment predictions --------------------------------------
        ms2_results = None
        if self.ms2_model is not None:
            if exp_charges is None:
                raise ValueError("charges are required for MS2 prediction")
            if exp_nces is None:
                raise ValueError("nces are required for MS2 prediction")
            ms2_results = self.ms2_model.predict(
                peptides,
                exp_charges,
                exp_nces,
                instruments=exp_inst,
                multiplier=multiplier,
            )

        # -- Build output columns ------------------------------------------
        b_ion_types = {"b", "b_nl"}

        if ms2_results is not None:
            # Long-format: one row per fragment ion
            pep_col: list[str] = []
            charge_col: list[int] = []
            nce_col: list[int] = []
            instrument_col: list[Optional[str]] = []
            rt_col: list[float] = []
            ccs_col: list[float] = []
            mobility_col: list[float] = []
            precursor_mz_col: list[float] = []
            ion_type_col: list[str] = []
            frag_charge_col: list[int] = []
            ordinal_col: list[int] = []
            intensity_col: list[float] = []
            mz_col: list[float] = []

            for idx, (pep, res) in enumerate(zip(peptides, ms2_results)):
                intensities = res["intensities"]
                ion_types = res["ion_types"]
                frag_charges = res["ion_charges"]
                b_ords = res["b_ordinals"]
                y_ords = res["y_ordinals"]
                n_pos, n_types = intensities.shape

                # scalar values for this peptide
                _charge = exp_charges[idx] if exp_charges is not None else 0
                _nce = exp_nces[idx] if exp_nces is not None else 0
                _instrument = exp_inst[idx] if exp_inst is not None else None
                _rt = float(rt_values[idx]) if rt_values is not None else float("nan")
                _ccs = (
                    float(ccs_values[idx]) if ccs_values is not None else float("nan")
                )

                # Precompute m/z info for this peptide (if requested)
                _precursor_mz = float("nan")
                frag_mz_lookup: dict[tuple[str, int, int], float] | None = None
                if (annotate_mz or annotate_mobility) and exp_charges is not None:
                    try:
                        _precursor_mz = compute_precursor_mz(pep, _charge)
                    except Exception:
                        pass
                if annotate_mz and exp_charges is not None:
                    max_frag_charge = max(int(fc) for fc in frag_charges)
                    try:
                        frag_info = compute_fragment_mzs(pep, max_frag_charge)
                        frag_mz_lookup = {
                            (t, c, o): m
                            for t, c, o, m in zip(
                                frag_info["ion_types"],
                                frag_info["charges"],
                                frag_info["ordinals"],
                                frag_info["mzs"],
                            )
                        }
                    except Exception:
                        frag_mz_lookup = {}

                _ion_mobility = float("nan")
                if (
                    annotate_mobility
                    and ccs_values is not None
                    and exp_charges is not None
                ):
                    try:
                        _ion_mobility = ccs_to_mobility(
                            _ccs, float(_charge), _precursor_mz
                        )
                    except Exception:
                        pass

                for r in range(n_pos):
                    for c in range(n_types):
                        t = ion_types[c]
                        ordinal = int(b_ords[r]) if t in b_ion_types else int(y_ords[r])
                        val = float(intensities[r, c])
                        if exclude_zeros and val == 0.0:
                            continue
                        pep_col.append(pep)
                        charge_col.append(_charge)
                        nce_col.append(_nce)
                        instrument_col.append(_instrument)
                        rt_col.append(_rt)
                        ccs_col.append(_ccs)
                        if annotate_mobility:
                            mobility_col.append(_ion_mobility)
                        ion_type_col.append(t)
                        frag_charge_col.append(int(frag_charges[c]))
                        ordinal_col.append(ordinal)
                        intensity_col.append(val)
                        if annotate_mz:
                            precursor_mz_col.append(_precursor_mz)
                            if frag_mz_lookup is not None:
                                base_type = t.replace("_nl", "")
                                mz_col.append(
                                    frag_mz_lookup.get(
                                        (base_type, int(frag_charges[c]), ordinal),
                                        float("nan"),
                                    )
                                )
                            else:
                                mz_col.append(float("nan"))

            data: dict = {"peptide": pep_col}
            if exp_charges is not None:
                data["charge"] = charge_col
            if exp_nces is not None:
                data["nce"] = nce_col
            if exp_inst is not None:
                data["instrument"] = instrument_col
            if rt_values is not None:
                data["rt"] = rt_col
            if ccs_values is not None:
                data["ccs"] = ccs_col
            if annotate_mobility:
                data["ion_mobility"] = mobility_col
            if annotate_mz:
                data["precursor_mz"] = precursor_mz_col
            data["ion_type"] = ion_type_col
            data["fragment_charge"] = frag_charge_col
            data["ordinal"] = ordinal_col
            data["intensity"] = intensity_col
            if annotate_mz:
                data["mz"] = mz_col

        else:
            # No MS2 – one row per peptide with scalar columns only
            data = {"peptide": list(peptides)}
            if exp_charges is not None:
                data["charge"] = list(exp_charges)
            if rt_values is not None:
                data["rt"] = [float(v) for v in rt_values]
            if ccs_values is not None:
                data["ccs"] = ccs_values
            if annotate_mobility and ccs_values is not None and exp_charges is not None:
                mob_col: list[float] = []
                for pep, ch, ccs in zip(peptides, exp_charges, ccs_values):
                    try:
                        mz = compute_precursor_mz(pep, ch)
                        mob_col.append(ccs_to_mobility(ccs, float(ch), mz))
                    except Exception:
                        mob_col.append(float("nan"))
                data["ion_mobility"] = mob_col
            if annotate_mz and exp_charges is not None:
                prec_mz: list[float] = []
                for pep, ch in zip(peptides, exp_charges):
                    try:
                        prec_mz.append(compute_precursor_mz(pep, ch))
                    except Exception:
                        prec_mz.append(float("nan"))
                data["precursor_mz"] = prec_mz

        return _make_df(data, framework)
