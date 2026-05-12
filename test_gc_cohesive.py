"""
Unit tests for Element.calculate_Gc_by_integration — cohesive traction mapping.

Verifies that the ELFORM 19/20 cohesive traction layout is respected:
    sig_zz  -> normal traction T_n  (Mode I)
    sig_xx  -> in-plane shear  T_s1 (Mode II component 1)
    sig_yy  -> in-plane shear  T_s2 (Mode II component 2)
    sig_xy, sig_yz, sig_zx  -> empty (zero)

Reference: LSTC dynasupport note "Cohesive material models".
"""

import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, ".")
from LS_Post_data_reader import Element


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_element(sig_xx, sig_yy, sig_zz, delta_n, delta_t_x, delta_t_y,
                  times=None):
    """
    Build a minimal synthetic cohesive Element for testing.

    The element lives in the x-y plane (bottom face z=0, top face z=1e-3),
    so its local-z normal is [0, 0, 1].  Bottom nodes are fixed; top nodes
    move to produce the prescribed separations.

    Parameters
    ----------
    sig_xx, sig_yy, sig_zz : array-like
        Traction time histories.
    delta_n : array-like
        Normal separation (top - bottom in z), i.e. Mode I opening.
    delta_t_x, delta_t_y : array-like
        Tangential separations in x and y (Mode II sliding).
    times : array-like, optional
        Time vector; defaults to 0..N-1.
    """
    n = len(sig_zz)
    if times is None:
        times = np.arange(n, dtype=float)

    # -- stress_data --
    stress_df = pd.DataFrame({
        "sig_xx": sig_xx,
        "sig_yy": sig_yy,
        "sig_zz": sig_zz,
        "sig_xy": np.zeros(n),
        "sig_yz": np.zeros(n),
        "sig_zx": np.zeros(n),
    }, index=pd.Index(times, name="time"))

    # -- node_data --
    # 8 nodes: bottom (1-4) at z=0, top (5-8) at z=1e-3
    # All bottom nodes fixed; top nodes move by (delta_t_x, delta_t_y, delta_n).
    node_ids_bottom = [1, 2, 3, 4]
    node_ids_top    = [5, 6, 7, 8]

    rows = []
    for nid in node_ids_bottom:
        for t in times:
            rows.append({"time": t, "id": nid,
                         "x_disp": 0.0, "y_disp": 0.0, "z_disp": 0.0})

    for nid in node_ids_top:
        for i, t in enumerate(times):
            rows.append({"time": t, "id": nid,
                         "x_disp": float(delta_t_x[i]),
                         "y_disp": float(delta_t_y[i]),
                         "z_disp": float(delta_n[i])})

    node_df = pd.DataFrame(rows).set_index(["time", "id"])

    # -- initial_node_coords (element in x-y plane, z=0 bottom / z=1e-3 top) --
    coords = {
        1: {"x": 0.0, "y": 0.0, "z": 0.0},
        2: {"x": 1.0, "y": 0.0, "z": 0.0},
        3: {"x": 1.0, "y": 1.0, "z": 0.0},
        4: {"x": 0.0, "y": 1.0, "z": 0.0},
        5: {"x": 0.0, "y": 0.0, "z": 1e-3},
        6: {"x": 1.0, "y": 0.0, "z": 1e-3},
        7: {"x": 1.0, "y": 1.0, "z": 1e-3},
        8: {"x": 0.0, "y": 1.0, "z": 1e-3},
    }

    el = Element(
        eid=1, pid=1,
        node_ids=[1, 2, 3, 4, 5, 6, 7, 8],
        initial_node_coords=coords,
        node_data=node_df,
        stress_data=stress_df,
    )
    return el


def _trapz_integral(traction, separation):
    """Trapezoidal integral of traction over |d(separation)|."""
    traction = np.asarray(traction, dtype=float)
    separation = np.asarray(separation, dtype=float)
    dsep = np.diff(np.abs(separation), prepend=np.abs(separation[0]))
    # match the code: use average traction over each increment
    trac_avg = (traction + np.concatenate([[0.0], traction[:-1]])) / 2.0
    return float(np.sum(trac_avg * np.abs(dsep)))


# ---------------------------------------------------------------------------
# Test 1: only sig_xx non-zero — Mode II G_c = trapz(sig_xx, delta_t)
# ---------------------------------------------------------------------------

def test_mode_II_only_sig_xx():
    n = 20
    times = np.linspace(0, 1, n)
    sig_xx = np.linspace(0, 10.0, n)       # linear ramp
    sig_yy = np.zeros(n)
    sig_zz = np.zeros(n)
    delta_n  = np.zeros(n)
    delta_t_x = np.linspace(0, 0.5, n)    # Mode II sliding in x
    delta_t_y = np.zeros(n)

    el = _make_element(sig_xx, sig_yy, sig_zz, delta_n, delta_t_x, delta_t_y, times)
    _, gc_II = el.calculate_Gc_by_integration(mode="II")

    # tangential separation magnitude = |delta_t_x| (delta_t_y = 0)
    expected = _trapz_integral(sig_xx, delta_t_x)
    assert abs(gc_II - expected) < 1e-6, \
        f"Mode II (sig_xx only): got {gc_II:.8f}, expected {expected:.8f}"


# ---------------------------------------------------------------------------
# Test 2: only sig_yy non-zero — Mode II G_c = trapz(sig_yy, delta_t)
# ---------------------------------------------------------------------------

def test_mode_II_only_sig_yy():
    n = 20
    times = np.linspace(0, 1, n)
    sig_xx = np.zeros(n)
    sig_yy = np.linspace(0, 8.0, n)
    sig_zz = np.zeros(n)
    delta_n  = np.zeros(n)
    delta_t_x = np.zeros(n)
    delta_t_y = np.linspace(0, 0.4, n)

    el = _make_element(sig_xx, sig_yy, sig_zz, delta_n, delta_t_x, delta_t_y, times)
    _, gc_II = el.calculate_Gc_by_integration(mode="II")

    expected = _trapz_integral(sig_yy, delta_t_y)
    assert abs(gc_II - expected) < 1e-6, \
        f"Mode II (sig_yy only): got {gc_II:.8f}, expected {expected:.8f}"


# ---------------------------------------------------------------------------
# Test 3: both sig_xx and sig_yy non-zero
#   shear_traction = sqrt(sig_xx^2 + sig_yy^2)
#   delta_t        = sqrt(delta_t_x^2 + delta_t_y^2)
# ---------------------------------------------------------------------------

def test_mode_II_combined_shear():
    n = 20
    times = np.linspace(0, 1, n)
    sig_xx = np.linspace(0, 6.0, n)
    sig_yy = np.linspace(0, 8.0, n)
    sig_zz = np.zeros(n)
    delta_n  = np.zeros(n)
    delta_t_x = np.linspace(0, 0.3, n)
    delta_t_y = np.linspace(0, 0.4, n)

    el = _make_element(sig_xx, sig_yy, sig_zz, delta_n, delta_t_x, delta_t_y, times)
    _, gc_II = el.calculate_Gc_by_integration(mode="II")

    shear_mag = np.sqrt(sig_xx**2 + sig_yy**2)
    delta_t_mag = np.sqrt(delta_t_x**2 + delta_t_y**2)
    expected = _trapz_integral(shear_mag, delta_t_mag)
    assert abs(gc_II - expected) < 1e-6, \
        f"Mode II (combined): got {gc_II:.8f}, expected {expected:.8f}"


# ---------------------------------------------------------------------------
# Test 4: Mode I — only sig_zz drives the normal traction
# ---------------------------------------------------------------------------

def test_mode_I_uses_sig_zz():
    n = 20
    times = np.linspace(0, 1, n)
    sig_xx = np.zeros(n)
    sig_yy = np.zeros(n)
    sig_zz = np.linspace(0, 50.0, n)
    delta_n  = np.linspace(0, 0.01, n)
    delta_t_x = np.zeros(n)
    delta_t_y = np.zeros(n)

    el = _make_element(sig_xx, sig_yy, sig_zz, delta_n, delta_t_x, delta_t_y, times)
    _, gc_I = el.calculate_Gc_by_integration(mode="I")

    expected = _trapz_integral(sig_zz, delta_n)
    assert abs(gc_I - expected) < 1e-6, \
        f"Mode I (sig_zz): got {gc_I:.8f}, expected {expected:.8f}"


# ---------------------------------------------------------------------------
# Test 5: mixed mode G_C = G_I + G_II
# ---------------------------------------------------------------------------

def test_mixed_mode_equals_sum():
    n = 20
    times = np.linspace(0, 1, n)
    sig_xx = np.linspace(0, 6.0, n)
    sig_yy = np.linspace(0, 4.0, n)
    sig_zz = np.linspace(0, 50.0, n)
    delta_n  = np.linspace(0, 0.01, n)
    delta_t_x = np.linspace(0, 0.3, n)
    delta_t_y = np.linspace(0, 0.2, n)

    el = _make_element(sig_xx, sig_yy, sig_zz, delta_n, delta_t_x, delta_t_y, times)
    _, gc_I  = el.calculate_Gc_by_integration(mode="I")
    _, gc_II = el.calculate_Gc_by_integration(mode="II")
    _, gc_C  = el.calculate_Gc_by_integration(mode="C")

    assert abs(gc_C - (gc_I + gc_II)) < 1e-6, \
        f"Mixed mode: G_C={gc_C:.8f} != G_I+G_II={gc_I+gc_II:.8f}"


# ---------------------------------------------------------------------------
# Test 6: sig_xy / sig_yz / sig_zx are NOT used (even if non-zero)
# ---------------------------------------------------------------------------

def test_off_diagonal_slots_ignored():
    """
    Populate sig_xy, sig_yz, sig_zx with large values.
    Results must be identical to the case where they are zero.
    """
    n = 10
    times = np.linspace(0, 1, n)
    sig_xx = np.linspace(0, 5.0, n)
    sig_yy = np.zeros(n)
    sig_zz = np.linspace(0, 40.0, n)
    delta_n  = np.linspace(0, 0.005, n)
    delta_t_x = np.linspace(0, 0.2, n)
    delta_t_y = np.zeros(n)

    el_clean = _make_element(sig_xx, sig_yy, sig_zz, delta_n, delta_t_x, delta_t_y, times)
    _, gc_I_clean  = el_clean.calculate_Gc_by_integration(mode="I")
    _, gc_II_clean = el_clean.calculate_Gc_by_integration(mode="II")

    # Inject large spurious values into off-diagonal slots
    el_noisy = _make_element(sig_xx, sig_yy, sig_zz, delta_n, delta_t_x, delta_t_y, times)
    el_noisy.stress_data['sig_xy'] = 999.0
    el_noisy.stress_data['sig_yz'] = 999.0
    el_noisy.stress_data['sig_zx'] = 999.0

    _, gc_I_noisy  = el_noisy.calculate_Gc_by_integration(mode="I")
    _, gc_II_noisy = el_noisy.calculate_Gc_by_integration(mode="II")

    assert abs(gc_I_clean - gc_I_noisy) < 1e-10, \
        "Mode I changed when off-diagonal slots were set to 999"
    assert abs(gc_II_clean - gc_II_noisy) < 1e-10, \
        "Mode II changed when off-diagonal slots were set to 999"


# ---------------------------------------------------------------------------
# Test 7: DCB benchmark regression — element 14501
# ---------------------------------------------------------------------------

def test_dcb_benchmark_regression():
    """G_Ic for the pure Mode I DCB element must stay at ~0.256 J/m2."""
    import os
    folder = r"C:\Users\nir\Desktop\Final_Project\analysis\single_element_mode_1_two_ways"
    if not os.path.isdir(folder):
        pytest.skip("DCB benchmark folder not available")

    from LS_Post_data_reader import Model
    model = Model(folder=folder, keyfile="simgle_element_mode_1.k",
                  load_nodout=True, load_elout=True, load_matsum=False)
    el = model.get_element(14501)

    _, gc_I  = el.calculate_Gc_by_integration(mode="I")
    _, gc_II = el.calculate_Gc_by_integration(mode="II")
    _, gc_C  = el.calculate_Gc_by_integration(mode="C")

    assert abs(gc_I - 0.256) < 0.002,  f"G_Ic regression: {gc_I:.6f}"
    assert gc_II < 1e-4,               f"G_IIc should be ~0 for pure Mode I DCB: {gc_II:.6f}"
    assert abs(gc_C - gc_I - gc_II) < 1e-8, "G_C != G_I + G_II"


if __name__ == "__main__":
    test_mode_II_only_sig_xx()
    test_mode_II_only_sig_yy()
    test_mode_II_combined_shear()
    test_mode_I_uses_sig_zz()
    test_mixed_mode_equals_sum()
    test_off_diagonal_slots_ignored()
    test_dcb_benchmark_regression()
    print("All tests passed.")
