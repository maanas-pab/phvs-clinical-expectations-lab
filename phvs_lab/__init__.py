"""PHVS Clinical Expectations Lab.

A research-grade, auditable application for testing a short thesis on
Pharvaris N.V. (NASDAQ: PHVS) by reverse-engineering the clinical preferences
and prescribing behaviour required to justify the observed market price.

Package layout
--------------
``phvs_lab.config``           Theme constants, paths, static copy.
``phvs_lab.modules.*``        Pure-Python analysis engines (no Streamlit).
``phvs_lab.utils``            Plotly figure builders and export helpers.
``phvs_lab.data``             Auditable CSV evidence layer.

All analysis functions are side-effect free so they can be unit tested
without launching the UI.
"""

__version__ = "1.0.0"
__app_name__ = "PHVS Clinical Expectations Lab"
