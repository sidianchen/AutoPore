# AutoPore

AutoPore provides one GUI for image generation/import, pore-network
extraction, quasi-static or dynamic simulation, uncertainty quantification,
and result plotting.

## Project layout

```text
AutoPore.py                             # GUI launcher
pore_network_model/
  autopore/                             # Python package
    gui.py                              # unified GUI
    llm.py                              # Ollama/OpenAI/OpenRouter selection
    model.py                            # pore-network workflow backend
    utils.py                            # plotting and model utilities
    workflow.py                         # workflow tracker
    cli.py                              # no-LLM command-line runner
    simple_extract_pnm.py               # extraction-only runner
    image_utils.py                      # lightweight extraction helpers
    compare_dynamic.py                  # multi-case dynamic curve comparison
    crop_outputs.py                     # optional figure/video cropping utility
    paraview_script.py                  # ParaView rendering
  image/                                # imported image files
  input_pnm/                            # active model input
  *.exe / *.bat                         # simulation executables
installation/                           # environment setup and verification
```

Use `AutoPore.py` to launch the GUI.

## Installation

Prerequisites are Windows, Anaconda/Miniconda, and ParaView. Ollama is needed
only for local-LLM use.

Command Prompt:

```bat
cd installation
install_env_windows.bat
```

PowerShell:

```powershell
cd installation
.\install_env_windows.ps1
```

Manual verification:

```powershell
conda activate pore-network-viz
python installation/verify_install.py
```

## Run the GUI

```powershell
conda activate pore-network-viz
python AutoPore.py
```

The LLM bar in the GUI selects:

- `local`: an installed Ollama model discovered with **Detect**
- `openai`: a model accessed with `OPENAI_API_KEY`
- `openrouter`: a model accessed with `OPENROUTER_API_KEY`

The selected provider, model, and base URL are saved in
`pore_network_model/.autopore/llm_settings.json`. Changing the provider
fills its default model and endpoint; both remain editable.

Local Ollama setup example:

```powershell
ollama pull qwen2.5:7b
ollama serve
python AutoPore.py
```

The default local endpoint is `http://127.0.0.1:11434/v1`. Optional environment
variables are `OLLAMA_MODEL`, `OLLAMA_HOST`, and `OLLAMA_BASE_URL`.

OpenAI:

```powershell
$env:OPENAI_API_KEY="sk-your-key"
python AutoPore.py
```

OpenRouter:

```powershell
$env:OPENROUTER_API_KEY="sk-or-v1-your-key"
python AutoPore.py
```

## Run without an LLM

The package CLI keeps simulation and plotting as independent tasks:

```powershell
python -m pore_network_model.autopore.cli qs-single-run
python -m pore_network_model.autopore.cli qs-single-plot

python -m pore_network_model.autopore.cli dyn-single-run
python -m pore_network_model.autopore.cli dyn-single-plot

python -m pore_network_model.autopore.cli qs-uq-run --realizations 5
python -m pore_network_model.autopore.cli qs-uq-plot

python -m pore_network_model.autopore.cli dyn-uq-run --realizations 5
python -m pore_network_model.autopore.cli dyn-uq-plot
```

X-ray examples:

```powershell
python -m pore_network_model.autopore.cli xray-qs-single-run --image-file my_image.raw
python -m pore_network_model.autopore.cli xray-dyn-single-run --image-file my_image.raw
```

Pore/throat diameter options can be selected with
`--pore-diameter-type` and `--throat-diameter-type`.

## Extraction only

Edit the user settings near the top of `simple_extract_pnm.py`, then run:

```powershell
python -m pore_network_model.autopore.simple_extract_pnm
```

This path writes image and pore-network inputs without running a simulation.

Dynamic folders can be compared with:

```powershell
python -m pore_network_model.autopore.compare_dynamic --suffixes low mid high --labels "Low" "Middle" "High"
```

## Workflow

The guarded workflow proceeds from top to bottom:

1. Generate or import an image and extract the pore network
2. Run a quasi-static or dynamic simulation
3. Post-process and plot results

For UQ, image generation and simulation repeat for every realization before
aggregate and percentile plots are produced.

## Main parameters

| Group | Parameters |
|---|---|
| Modes | `task_mode`, `pnm_mode` |
| Image | `img_src`, dimensions, voxel size, porosity, blobiness |
| Extraction | image coarsening, pore diameter, throat diameter |
| Quasi-static | minimum pressure, maximum pressure, pressure step |
| Dynamic | wetting/nonwetting viscosity, capillary number, maximum capillary pressure |
| Plotting | metric and flow-pattern file mode |

X-ray files belong in `pore_network_model/image/`. Active simulation parameter
files are `pore_network_model/input_pnm/qs_param.txt` and `dyn_param.txt`.

## Outputs

Results are written under:

- `pore_network_model/single_run_qs_pnm/`
- `pore_network_model/single_run_dyn_pnm/`
- `pore_network_model/uq_qs_pnm/`
- `pore_network_model/uq_dyn_pnm/`

The GUI shows generated PNG figures in its figure window and records generated
file locations in the workflow panel.

## Example prompts

```text
Generate an image using default settings
Import an x-ray image from my folder
Run a quasi-static simulation
Run a dynamic simulation
Plot the results
Run uncertainty quantification with 5 quasi-static realizations
Set the wetting-phase viscosity to 0.01
Set the capillary number to 100
Reset everything
```

## Reference

The AutoPore code was developed by [Chen & Tchelepi (preprint)](https://egusphere.copernicus.org/preprints/2026/egusphere-2026-6229/). If you use this code (or any modified version), please cite the following publication:

S. Chen*, H. A. Tchelepi, An AI-assisted framework for image-based multiphase flow simulations. Preprint.

For questions, please contact the development team at sidianc@stanford.edu or sidianchen11@gmail.com.
