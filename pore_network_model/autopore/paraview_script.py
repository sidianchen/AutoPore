import sys
import re
import math
from pathlib import Path
from paraview.simple import *


DIAMETER_PROPERTY_NAMES = {
    "pore": {
        "inscribed": "inscribed_diameter",
        "equivalent": "equivalent_diameter",
        "extended": "extended_diameter",
    },
    "throat": {
        "inscribed": "inscribed_diameter",
        "equivalent": "equivalent_diameter",
    },
}

DEFAULT_PORE_DIAMETER_PROPERTY = DIAMETER_PROPERTY_NAMES["pore"]["equivalent"]
DEFAULT_THROAT_DIAMETER_PROPERTY = DIAMETER_PROPERTY_NAMES["throat"]["equivalent"]


def normalize_diameter_property(element, value):
    if value is None:
        if element == "pore":
            return DEFAULT_PORE_DIAMETER_PROPERTY
        if element == "throat":
            return DEFAULT_THROAT_DIAMETER_PROPERTY
        raise ValueError(f"Unknown diameter element: {element}")

    value = str(value).strip()
    if value in DIAMETER_PROPERTY_NAMES[element]:
        return DIAMETER_PROPERTY_NAMES[element][value]
    if value in DIAMETER_PROPERTY_NAMES[element].values():
        return value

    allowed = sorted(
        set(DIAMETER_PROPERTY_NAMES[element]) | set(DIAMETER_PROPERTY_NAMES[element].values())
    )
    raise ValueError(
        f"Invalid {element} diameter property '{value}'. Allowed values: {allowed}"
    )


def write_simple_pvd(files, time_values, output_pvd):
    output_pvd = Path(output_pvd)
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        '  <Collection>',
    ]

    for f, t in zip(files, time_values):
        rel_str = str(f).replace("\\", "/")
        lines.append(
            f'    <DataSet timestep="{t:.16g}" group="" part="0" file="{rel_str}"/>'
        )

    lines += [
        '  </Collection>',
        '</VTKFile>',
    ]
    output_pvd.write_text("\n".join(lines), encoding="utf-8")

def extract_time_from_path(path):
    path = Path(path)
    matches = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', path.stem)
    return float(matches[-1]) if matches else None


def get_sorted_vtp_files_and_times(input_folder):
    input_folder = Path(input_folder)
    files = list(input_folder.glob("*.vtp"))
    files = sorted(
        files,
        key=lambda p: extract_time_from_path(p) if extract_time_from_path(p) is not None else float("inf")
    )
    time_values = [extract_time_from_path(f) for f in files]
    return files, time_values

def get_sorted_vtp_files(input_folder):
    input_folder = Path(input_folder)

    def extract_number(path):
        # Use the LAST number in the filename stem, which is usually the timestep
        matches = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', path.stem)
        return float(matches[-1]) if matches else float("inf")

    files = sorted(input_folder.glob("*.vtp"), key=extract_number)

    print("Sorted VTP files:")
    for f in files[:20]:
        print(f"  {f.name}  ->  {extract_number(f)}")
    if len(files) > 20:
        print("  ...")

    return files


def reset_view_preserve_angle(render_view):
    camera_position = [float(value) for value in render_view.CameraPosition]
    camera_focal_point = [float(value) for value in render_view.CameraFocalPoint]
    camera_view_up = [float(value) for value in render_view.CameraViewUp]

    view_direction = [
        camera_focal_point[i] - camera_position[i]
        for i in range(3)
    ]
    view_distance = math.sqrt(sum(value * value for value in view_direction))
    if view_distance <= 0:
        render_view.ResetCamera()
        return

    view_direction = [value / view_distance for value in view_direction]

    render_view.ResetCamera()

    fitted_focal_point = [float(value) for value in render_view.CameraFocalPoint]
    fitted_position = [float(value) for value in render_view.CameraPosition]
    fitted_direction = [
        fitted_focal_point[i] - fitted_position[i]
        for i in range(3)
    ]
    fitted_distance = math.sqrt(sum(value * value for value in fitted_direction))
    if fitted_distance <= 0:
        fitted_distance = view_distance

    render_view.CameraFocalPoint = fitted_focal_point
    render_view.CameraPosition = [
        fitted_focal_point[i] - view_direction[i] * fitted_distance
        for i in range(3)
    ]
    render_view.CameraViewUp = camera_view_up
    if hasattr(render_view, "ResetCameraClippingRange"):
        render_view.ResetCameraClippingRange()
    render_view.Update()


def main():
    print("sys.argv =", sys.argv)

    # Expected:
    # pvbatch.exe paraview_script.py <input_vtp_or_folder> <output_name> <scalar_name> <output_mode> <vmin> <vmax>
    if len(sys.argv) not in (10, 12):
        print("Usage:")
        print("  pvbatch script.py <input_vtp_or_folder> <output_name> <scalar_name> <output_mode> <vmin> <vmax> <mode> <digits> <start> [pore_diameter_property throat_diameter_property]")
        print("")
        print("Examples:")
        print("  Single image:")
        print("    pvbatch script.py file.vtp out.png sat_nw png 0 1")
        print("  Animation:")
        print("    pvbatch script.py ./vtps out.avi sat_nw animation 0 1")
        sys.exit(1)

    input_path = sys.argv[1]
    output_name = sys.argv[2]
    scalar_name = sys.argv[3]
    output_mode = sys.argv[4]
    vmin = float(sys.argv[5])
    vmax = float(sys.argv[6])
    pore_diameter_property = normalize_diameter_property(
        "pore",
        sys.argv[10] if len(sys.argv) > 10 else None,
    )
    throat_diameter_property = normalize_diameter_property(
        "throat",
        sys.argv[11] if len(sys.argv) > 11 else None,
    )
    pore_size_array_name = f"network | properties | pore | {pore_diameter_property}"
    throat_size_array_name = f"network | properties | throat | {throat_diameter_property}"

    print("input_path   =", input_path)
    print("output_name  =", output_name)
    print("scalar_name  =", scalar_name)
    print("output_mode  =", output_mode)
    print("color limits =", [vmin, vmax])
    print("pore size array   =", pore_size_array_name)
    print("throat size array =", throat_size_array_name)
    
    
    if output_mode == "pnm":
        pore_array_name = pore_size_array_name
    else:
        pore_array_name = "network | properties | pore | " + scalar_name

    _DisableFirstRenderCameraReset()

    # -----------------------------
    # Read input
    # -----------------------------
    if output_mode == "animation":
        vtp_files, time_values = get_sorted_vtp_files_and_times(input_path)
        vtp_files = [str(f.resolve()) for f in get_sorted_vtp_files(input_path)]
        print("Found VTP files:", len(vtp_files))
    
        if len(vtp_files) == 0:
            print(f"Error: no .vtp files found in folder: {input_path}")
            sys.exit(2)
    
    
        output_pvd = Path(input_path).resolve().parent / "time_series.pvd"
        write_simple_pvd(vtp_files[int(sys.argv[9]):], time_values[int(sys.argv[9]):], output_pvd)
    
        #print("PVD file:", output_pvd)
        #print(output_pvd.read_text(encoding="utf-8"))
    
        reader = PVDReader(
            registrationName="time_series_reader",
            FileName=str(output_pvd.resolve())
        )
    
        reader.UpdatePipelineInformation()
    
        animationScene = GetAnimationScene()
        animationScene.UpdateAnimationUsingDataTimeSteps()
    
        timeKeeper = GetTimeKeeper()
        #print("Detected timesteps:", getattr(timeKeeper, "TimestepValues", None))

    else:
        vtp_files = None
        reader = XMLPolyDataReader(
            registrationName="single_reader",
            FileName=[str(Path(input_path).resolve())]
        )
        time_values = [extract_time_from_path(str(Path(input_path).resolve()))]

    # -----------------------------
    # Render view
    # -----------------------------
    renderView1 = GetActiveViewOrCreate("RenderView")
    readerDisplay = Show(reader, renderView1, "GeometryRepresentation")
    readerDisplay.Representation = "Surface"
    

    digit = int(sys.argv[8])

    
    if output_mode !="pnm":
        if output_mode == "animation":
            annotateTime1 = AnnotateTime(registrationName="AnnotateTime1")
            if sys.argv[7] == "quasi_static":
                annotateTime1.Format = 'Quasi static PNM: dP = {time:e} Pa'
            else:
                annotateTime1.Format = "Dynamic PNM: time = {time:e} s"
            annotateTime1Display = Show(annotateTime1, renderView1, "TextSourceRepresentation")
            annotateTime1Display.WindowLocation = "Upper Left Corner"
            annotateTime1Display.FontSize = 24
            annotateTime1Display.Bold = 0
            annotateTime1Display.Color = [0.1, 0.1, 0.1]
        else:
            reader.TimeArray = "None"
            annotateTime1 = AnnotateTime(registrationName="AnnotateTime1")
            if sys.argv[7] == "quasi_static":
                annotateTime1.Format = f"Quasi static PNM: dP = {time_values[0]:.{digit}e} Pa"
            else:
                annotateTime1.Format = f"Dynamic PNM: time = {time_values[0]:.{digit}e} s"
            annotateTime1Display = Show(annotateTime1, renderView1, "TextSourceRepresentation")
            annotateTime1Display.WindowLocation = "Upper Left Corner"
            annotateTime1Display.FontSize = 24
            annotateTime1Display.Bold = 0
            annotateTime1Display.Color = [0.1, 0.1, 0.1]
          
            
    if output_mode == "pnm" or (scalar_name != "sat_nw" and scalar_name != "sat_w"):  
        glyph1 = Glyph(registrationName="Glyph1", Input=reader, GlyphType="Arrow")
    else:
        threshold1 = Threshold(registrationName='Threshold1', Input=reader)
    
        threshold1.Scalars = ['POINTS', pore_array_name]
        threshold1.ThresholdMethod = 'Between'
        
        threshold1.LowerThreshold = 1e-8
        threshold1.UpperThreshold = 1.0
            
        # -----------------------------
        # Pore glyphs (boxes)
        # -----------------------------
        glyph1 = Glyph(registrationName="Glyph1", Input=threshold1, GlyphType="Arrow")
    
        

    glyph1.Set(
        GlyphType="Box",
        ScaleArray=["POINTS", pore_size_array_name],
        ScaleFactor= 1.0,
        GlyphMode="All Points",
    )

    glyph1Display = Show(glyph1, renderView1, "GeometryRepresentation")
    glyph1Display.Representation = "Surface"
    renderView1.Update()

    # -----------------------------
    # Color glyphs
    # -----------------------------
    if output_mode == "pnm":
        pore_array_name = pore_size_array_name
    else:
        pore_array_name = "network | properties | pore | " + scalar_name

    ColorBy(glyph1Display, ("POINTS", pore_array_name))
    glyph1Display.RescaleTransferFunctionToDataRange(True, False)
    glyph1Display.SetScalarBarVisibility(renderView1, True)

    # Transfer functions
    lut_name = "networkpropertiespore" + scalar_name
    networkLUT = GetColorTransferFunction(lut_name)
    networkPWF = GetOpacityTransferFunction(lut_name)
    networkTF2D = GetTransferFunction2D(lut_name)

    scalarBar = GetScalarBar(networkLUT, renderView1)
    scalarBar.Set(
        WindowLocation="Any Location",
        Position=[0.8570512820512821, 0.07051282051282051],
        ScalarBarLength=0.32999999999999996,
    )

    networkLUT.RescaleTransferFunction(vmin, vmax)
    print("Applied LUT range:", [vmin, vmax])

    # -----------------------------
    # Base reader coloring
    # -----------------------------
    SetActiveSource(reader)

    ColorBy(readerDisplay, ("POINTS", pore_array_name))
    readerDisplay.RescaleTransferFunctionToDataRange(True, False)
    readerDisplay.SetScalarBarVisibility(renderView1, True)

    SetActiveSource(glyph1)

    networkPWF.Points = [0.0, 0.267857164144516, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

    # Properties modified on networkpropertiesporesat_nwLUT
    networkLUT.EnableOpacityMapping = 0
    
    # get opacity transfer function/opacity map for 'networkpropertiesporesat_nw'
    networkPWF = GetOpacityTransferFunction('networkpropertiesporesat_nw')
    
    # get 2D transfer function for 'networkpropertiesporesat_nw'
    networkTF2D = GetTransferFunction2D('networkpropertiesporesat_nw')
    
    # Properties modified on networkpropertiesporesat_nwPWF
    networkPWF.Points = [0.0, 0.4419642984867096, 0.5, 0.0, 0.999934, 1.0, 0.5, 0.0]

    networkPWF.Points = [0.0, 0.0133928582072258, 0.5, 0.0, 0.999934, 1.0, 0.5, 0.0]
    
    

    # -----------------------------
    # Throat pipeline
    # -----------------------------
    SetActiveSource(reader)
    
    if output_mode == "pnm":
        shrink1 = Shrink(registrationName="Shrink1", Input=reader)
    else:
        shrink1 = Shrink(registrationName="Shrink1", Input=threshold1)
    shrink1.ShrinkFactor = 1.0

    shrink1Display = Show(shrink1, renderView1, "UnstructuredGridRepresentation")
    shrink1Display.Representation = "Surface"
    
    

    Hide(reader, renderView1)
    shrink1Display.SetScalarBarVisibility(renderView1, True)
    renderView1.Update()

    cellDatatoPointData1 = CellDatatoPointData(
        registrationName="CellDatatoPointData1",
        Input=shrink1
    )

    cellDatatoPointData1Display = Show(
        cellDatatoPointData1, renderView1, "UnstructuredGridRepresentation"
    )
    cellDatatoPointData1Display.Representation = "Surface"

    Hide(shrink1, renderView1)
    cellDatatoPointData1Display.SetScalarBarVisibility(renderView1, True)
    renderView1.Update()

    extractSurface1 = ExtractSurface(
        registrationName="ExtractSurface1",
        Input=cellDatatoPointData1
    )

    extractSurface1Display = Show(extractSurface1, renderView1, "GeometryRepresentation")
    extractSurface1Display.Representation = "Surface"

    Hide(cellDatatoPointData1, renderView1)
    extractSurface1Display.SetScalarBarVisibility(renderView1, True)
    renderView1.Update()

    tube1 = Tube(registrationName="Tube1", Input=extractSurface1)
    tube1.Set(
        Scalars=["POINTS", throat_size_array_name],
        Vectors=["POINTS", ""],
        NumberofSides=4,
        Radius=1.0,
    )

    tube1Display = Show(tube1, renderView1, "GeometryRepresentation")
    tube1Display.Representation = "Surface"
    
    SetActiveSource(tube1)
    ColorBy(tube1Display, ("POINTS", pore_array_name))

    Hide(extractSurface1, renderView1)
    tube1Display.SetScalarBarVisibility(renderView1, True)
    renderView1.Update()

    # -----------------------------
    # Layout and camera
    # -----------------------------
    layout1 = GetLayout()
    layout1.SetSize(1560, 780)

    renderView1.Set(
        CameraPosition=[-185.74120439328044, -187.2786417517627, 198.05598552046257],
        CameraFocalPoint=[86.67288378766136, 36.62500000000002, 36.571428571428584],
        CameraViewUp=[0.2767332887347608, 0.3174564433413025, 0.9069950901119224],
        CameraParallelScale=100.3802692639437,
    )

    networkLUT.RescaleTransferFunction(vmin, vmax)
    print("Re-applied LUT range:", [vmin, vmax])
    

    
    
##############################################################################################################################################################################
    
    # create a new 'XML PolyData Reader'
    inp = str(Path(input_path).resolve()) if output_mode != "animation" else str(Path(input_path) / vtp_files[0])
    ps_20000000vtp = XMLPolyDataReader(registrationName='ps_200.00000.vtp', FileName=[inp])
    
    # Properties modified on ps_10000000vtp
    ps_20000000vtp.TimeArray = 'None'
    
    # show data in view
    ps_20000000vtpDisplay = Show(ps_20000000vtp, renderView1, 'GeometryRepresentation')
    
    # trace defaults for the display properties.
    ps_20000000vtpDisplay.Representation = 'Outline'
    
    
    # create a new 'XML PolyData Reader'
    inp = str(Path(input_path).resolve()) if output_mode != "animation" else str(Path(input_path) / vtp_files[0]) 
    ps_10000000vtp = XMLPolyDataReader(registrationName='ps_100.00000.vtp', FileName=[inp])
    
    # Properties modified on ps_10000000vtp
    ps_10000000vtp.TimeArray = 'None'
    
    # show data in view
    ps_10000000vtpDisplay = Show(ps_10000000vtp, renderView1, 'GeometryRepresentation')
    
    # trace defaults for the display properties.
    ps_10000000vtpDisplay.Representation = 'Surface'
    
    # update the view to ensure updated data information
    renderView1.Update()
    
    # create a new 'Glyph'
    glyph2 = Glyph(registrationName='Glyph2', Input=ps_10000000vtp,
        GlyphType='Arrow')
    
    # Properties modified on glyph2
    glyph2.GlyphType = 'Box'
    
    # show data in view
    glyph2Display = Show(glyph2, renderView1, 'GeometryRepresentation')
    
    # trace defaults for the display properties.
    glyph2Display.Representation = 'Surface'
    
    # update the view to ensure updated data information
    renderView1.Update()
    
    # Properties modified on glyph2
    glyph2.Set(
        ScaleArray=['POINTS', pore_size_array_name],
        ScaleFactor=1.0,
        GlyphMode='All Points',
    )
    
    # update the view to ensure updated data information
    renderView1.Update()
    
    # Properties modified on glyph2
    glyph2.ScaleFactor = 0.999
    
    # update the view to ensure updated data information
    renderView1.Update()
    
    # set active source
    SetActiveSource(ps_10000000vtp)
    
    # create a new 'Shrink'
    shrink2 = Shrink(registrationName='Shrink2', Input=ps_10000000vtp)
    
    # Properties modified on shrink2
    shrink2.ShrinkFactor = 1.0
    
    # show data in view
    shrink2Display = Show(shrink2, renderView1, 'UnstructuredGridRepresentation')
    
    # trace defaults for the display properties.
    shrink2Display.Representation = 'Surface'
    
    # hide data in view
    Hide(ps_10000000vtp, renderView1)
    
    # update the view to ensure updated data information
    renderView1.Update()
    
    # create a new 'Cell Data to Point Data'
    cellDatatoPointData2 = CellDatatoPointData(registrationName='CellDatatoPointData2', Input=shrink2)
    
    # show data in view
    cellDatatoPointData2Display = Show(cellDatatoPointData2, renderView1, 'UnstructuredGridRepresentation')
    
    # trace defaults for the display properties.
    cellDatatoPointData2Display.Representation = 'Surface'
    
    # hide data in view
    Hide(shrink2, renderView1)
    
    # update the view to ensure updated data information
    renderView1.Update()
    
    # create a new 'Extract Surface'
    extractSurface2 = ExtractSurface(registrationName='ExtractSurface2', Input=cellDatatoPointData2)
    
    # show data in view
    extractSurface2Display = Show(extractSurface2, renderView1, 'GeometryRepresentation')
    
    # trace defaults for the display properties.
    extractSurface2Display.Representation = 'Surface'
    
    # hide data in view
    Hide(cellDatatoPointData2, renderView1)
    
    # update the view to ensure updated data information
    renderView1.Update()
    
    # create a new 'Tube'
    tube2 = Tube(registrationName='Tube2', Input=extractSurface2)
    
    # Properties modified on tube2
    tube2.Set(
        Scalars=['POINTS', throat_size_array_name],
        Vectors=['POINTS', ''],
        NumberofSides=4,
        Radius=0.999,
    )
    
    # show data in view
    tube2Display = Show(tube2, renderView1, 'GeometryRepresentation')
    
    # trace defaults for the display properties.
    tube2Display.Representation = 'Surface'
    
    # hide data in view
    Hide(extractSurface2, renderView1)
    
    # update the view to ensure updated data information
    renderView1.Update()
    
    # set active source
    SetActiveSource(tube2)
    

    # Properties modified on tube2Display
    tube2Display.Opacity = 0.01
    
    # set active source
    SetActiveSource(glyph2)
    
    # Properties modified on glyph2Display
    glyph2Display.Opacity = 0.05
    

######################################################################################################################################################################

    # Force update before save
    renderView1.Update()

    # -----------------------------
    # Save output
    # -----------------------------
    if output_mode == "png":
        print("Saving screenshot to:", output_name)
        reset_view_preserve_angle(renderView1)
        SaveScreenshot(
            filename=str(output_name),
            viewOrLayout=renderView1,
            location=16,
            ImageResolution=[1560, 780],
            OverrideColorPalette="WhiteBackground",
        )

    elif output_mode == "animation":
        
        output_path = Path(output_name)
        if output_path.exists():
            try:
                print("Removing existing file:", output_path)
                output_path.unlink()
            except Exception as e:
                print(f"Warning: could not remove existing file: {e}")
        
        
        animationScene = GetAnimationScene()
        animationScene.UpdateAnimationUsingDataTimeSteps()

        timeKeeper = GetTimeKeeper()
        timesteps = list(getattr(timeKeeper, "TimestepValues", []))

        if len(timesteps) > 0:
            nframes = len(timesteps)
        else:
            print("Warning: ParaView did not detect timesteps, using file count instead.")
            nframes = len(vtp_files)

        print("Saving animation to:", output_name)
        #print("Frame count:", nframes)
        #print("Timesteps:", timesteps)

        reset_view_preserve_angle(renderView1)
        SaveAnimation(
            filename=str(output_name),
            viewOrLayout=renderView1,
            location=16,
            ImageResolution=[1560, 780],
            OverrideColorPalette="WhiteBackground",
            FrameRate=5,
            FrameWindow=[0, nframes - 1],
        )

    elif output_mode =="pnm":
        print("Saving screenshot to:", output_name)
        reset_view_preserve_angle(renderView1)
        SaveScreenshot(
            filename=str(output_name),
            viewOrLayout=renderView1,
            location=16,
            ImageResolution=[1560, 780],
            OverrideColorPalette="WhiteBackground",
        )

    else:
        print(f"Error: unsupported output_mode = {output_mode}")
        sys.exit(3)

    print("Done.")


if __name__ == "__main__":
    main()
