from pathlib import Path
import traceback

OUT = Path("verification_outputs")
OUT.mkdir(exist_ok=True)

print("=== Package import check ===")
packages = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("pyvista", "pyvista"),
    ("vtk", "vtk"),
    ("openpnm", "openpnm"),
    ("porespy", "porespy"),
    ("skimage", "skimage"),
]

loaded = {}
for mod_name, label in packages:
    try:
        mod = __import__(mod_name)
        version = getattr(mod, "__version__", "unknown")
        loaded[label] = mod
        print(f"[OK] {label}: {version}")
    except Exception as exc:
        print(f"[FAIL] {label}: {exc}")

print("\n=== AutoPore package check ===")
try:
    from pore_network_model.autopore import llm
    from pore_network_model.autopore.model import THIS_DIR

    print(f"[OK] Unified package: {THIS_DIR}")
    print(f"[OK] LLM providers: {', '.join(llm.VALID_PROVIDERS)}")
except Exception:
    print("[FAIL] Unified AutoPore package import failed")
    traceback.print_exc()

print("\n=== Basic OpenPNM network check ===")
try:
    import openpnm as op
    pn = op.network.Cubic(shape=[5, 5, 5])
    print(f"[OK] Created test network with {pn.Np} pores and {pn.Nt} throats")
except Exception:
    print("[FAIL] OpenPNM network creation failed")
    traceback.print_exc()

print("\n=== PyVista off-screen rendering check ===")
try:
    import pyvista as pv
    pv.OFF_SCREEN = True
    pl = pv.Plotter(off_screen=True, window_size=(800, 600))
    pl.add_mesh(pv.Cube())
    img_path = OUT / "test_cube.png"
    pl.show(screenshot=str(img_path), auto_close=True)
    print(f"[OK] Saved screenshot to: {img_path.resolve()}")
except Exception:
    print("[WARN] PyVista screenshot test failed.")
    print("This usually means the Windows OpenGL/VTK backend is unavailable.")
    print("You can still use the environment for OpenPNM/PoreSpy, but PyVista may need a graphics driver update or software rendering setup.")
    traceback.print_exc()

print("\nVerification finished.")


try:
    import spyder  # noqa: F401
    print("[OK] spyder imported")
except Exception as e:
    print(f"[WARN] spyder import failed: {e}")
