from win32com.client.gencache import EnsureDispatch, EnsureModule
from win32com.client import CastTo, constants
from win32com.client import gencache

import os
from shutil import copyfile

# Notes
#
# The python project and script was tested with the following tools:
#       Python 3.4.3 for Windows (32-bit) (https://www.python.org/downloads/) - Python interpreter
#       Python for Windows Extensions (32-bit, Python 3.4) (http://sourceforge.net/projects/pywin32/) - for COM support
#       Microsoft Visual Studio Express 2013 for Windows Desktop (https://www.visualstudio.com/en-us/products/visual-studio-express-vs.aspx) - easy-to-use IDE
#       Python Tools for Visual Studio (https://pytools.codeplex.com/) - integration into Visual Studio
#
# Note that Visual Studio and Python Tools make development easier, however this python script should should run without either installed.

class PythonStandaloneApplication(object):
    class LicenseException(Exception):
        pass

    class ConnectionException(Exception):
        pass

    class InitializationException(Exception):
        pass

    class SystemNotPresentException(Exception):
        pass

    def __init__(self):
        # make sure the Python wrappers are available for the COM client and
        # interfaces
        gencache.EnsureModule('{EA433010-2BAC-43C4-857C-7AEAC4A8CCE0}', 0, 1, 0)
        gencache.EnsureModule('{F66684D7-AAFE-4A62-9156-FF7A7853F764}', 0, 1, 0)
        # Note - the above can also be accomplished using 'makepy.py' in the
        # following directory:
        #      {PythonEnv}\Lib\site-packages\wind32com\client\
        # Also note that the generate wrappers do not get refreshed when the
        # COM library changes.
        # To refresh the wrappers, you can manually delete everything in the
        # cache directory:
        #	   {PythonEnv}\Lib\site-packages\win32com\gen_py\*.*

        self.TheConnection = EnsureDispatch("ZOSAPI.ZOSAPI_Connection")
        if self.TheConnection is None:
            raise PythonStandaloneApplication.ConnectionException("Unable to intialize COM connection to ZOSAPI")

        self.TheApplication = self.TheConnection.CreateNewApplication()
        if self.TheApplication is None:
            raise PythonStandaloneApplication.InitializationException("Unable to acquire ZOSAPI application")

        if self.TheApplication.IsValidLicenseForAPI == False:
            raise PythonStandaloneApplication.LicenseException("License is not valid for ZOSAPI use")

        self.TheSystem = self.TheApplication.PrimarySystem
        if self.TheSystem is None:
            raise PythonStandaloneApplication.SystemNotPresentException("Unable to acquire Primary system")

    def __del__(self):
        if self.TheApplication is not None:
            self.TheApplication.CloseApplication()
            self.TheApplication = None

        self.TheConnection = None

    def OpenFile(self, filepath, saveIfNeeded):
        if self.TheSystem is None:
            raise PythonStandaloneApplication.SystemNotPresentException("Unable to acquire Primary system")
        self.TheSystem.LoadFile(filepath, saveIfNeeded)

    def CloseFile(self, save):
        if self.TheSystem is None:
            raise PythonStandaloneApplication.SystemNotPresentException("Unable to acquire Primary system")
        self.TheSystem.Close(save)

    def SamplesDir(self):
        if self.TheApplication is None:
            raise PythonStandaloneApplication.InitializationException("Unable to acquire ZOSAPI application")

        return self.TheApplication.SamplesDir

    def ExampleConstants(self):
        if self.TheApplication.LicenseStatus is constants.LicenseStatusType_PremiumEdition:
            return "Premium"
        elif self.TheApplication.LicenseStatus is constants.LicenseStatusType_ProfessionalEdition:
            return "Professional"
        elif self.TheApplication.LicenseStatus is constants.LicenseStatusType_StandardEdition:
            return "Standard"
        else:
            return "Invalid"


if __name__ == '__main__':
    zosapi = PythonStandaloneApplication()
    value = zosapi.ExampleConstants()

    if not os.path.exists(zosapi.TheApplication.SamplesDir + "\\API\\Python"):
        os.makedirs(zosapi.TheApplication.SamplesDir + "\\API\\Python")

    #! [e11s01_py]
    # Create New Sequential File
    TheSystem = zosapi.TheSystem
    TheSystem.New(False)
    # Name File
    fileName = zosapi.TheApplication.SamplesDir + "\\API\\Python\\e11_basic_seq.zmx"
    TheSystem.SaveAs(fileName)
    #! [e11s01_py]

    #! [e11s02_py]
    # Changing System Explorer Settings
    # Set Aperture
    TheSystemData = TheSystem.SystemData
    TheSystemData.Aperture.ApertureValue = 20
    TheSystem.SystemData.MaterialCatalogs.AddCatalog('SCHOTT')

    # Set Apodization Type to Gaussian, and set apodization factor to 1
    TheSystemData.Aperture.ApodizationType = 1  # ApodizationType 0 = uniform; 1 = gaussian; 2 = Cosine Cubed
    TheSystemData.Aperture.ApodizationFactor = 1
    # Set Wavelength
    TheSystemData.Wavelengths.SelectWavelengthPreset(constants.WavelengthPreset_FdC_Visible)
    #! [e11s02_py]


    #! [e11s03_py]
    # Insert and Track New Surfaces, set STOP to surface 1
    TheLDE = TheSystem.LDE
    TheLDE.InsertNewSurfaceAt(1)
    TheLDE.InsertNewSurfaceAt(1)
    TheLDE.InsertNewSurfaceAt(1)
    Surf_1 = TheLDE.GetSurfaceAt(1)
    Surf_2 = TheLDE.GetSurfaceAt(2)
    Surf_3 = TheLDE.GetSurfaceAt(3)
    Surf_4 = TheLDE.GetSurfaceAt(4)
    Surf_1.IsStop = 1
    # Set some baseline parameters
    Surf_1.Thickness = 5
    Surf_2.Thickness = 5
    Surf_2.Radius = 100
    Surf_2.Material = "N-BK7"
    Surf_3.Thickness = 3
    Surf_3.Radius = -30
    Surf_3.Material = "F2"
    Surf_4.Radius = -80
    #! [e11s03_py]

    #! [e11s04_py]
    # Set system lens units to inches, scale all values with Scale Lens tool
    unit = TheSystemData.Units.LensUnits  # For demonstration only. This file is new, so it has default units mm.
    ScaleLens = TheSystem.Tools.OpenScale()  # Open Scale Lens tool
    # Apply Tool Settings
    ScaleLens.ScaleByUnits = True
    ScaleLens.ScaleToUnit = 2  # 0=millimeters; 1=centimeters; 2=inches; 3=meters
    # Cast to ISystemTool interface to gain access to Run
    ScaleTool = CastTo(ScaleLens, "ISystemTool")
    ScaleTool.RunAndWaitForCompletion()
    ScaleTool.Close()
    #! [e11s04_py]

    #! [e11s05_py]
    # Add Rectangular Aperture to Surface 1
    # Get surface 1, create aperture settings
    Surf_1 = zosapi.TheSystem.LDE.GetSurfaceAt(1)
    rAperture = Surf_1.ApertureData.CreateApertureTypeSettings(constants.SurfaceApertureTypes_RectangularAperture)
    # Set aperture size
    rAperture._S_RectangularAperture.XHalfWidth = .1
    rAperture._S_RectangularAperture.YHalfWidth = .1
    # Apply aperture settings to surface 1
    Surf_1.ApertureData.ChangeApertureTypeSettings(rAperture)
    #! [e11s05_py]

    #! [e11s06_py]
    # Run Quick Focus
    QuickFocus = TheSystem.Tools.OpenQuickFocus()
    FocusTool = CastTo(QuickFocus, "ISystemTool")
    FocusTool.RunAndWaitForCompletion()
    FocusTool.Close()
    #! [e11s06_py]

    #! [e11s07_py]
    # Open Universal Plot of RMS Spot Size vs Surface3 Thickness
    UnivPlot = TheSystem.Analyses.New_Analysis(constants.AnalysisIDM_UniversalPlot1D)
    UnivPlot_Settings = UnivPlot.GetSettings()
    UnivPlot_Set = CastTo(UnivPlot_Settings, "IAS_")  # Cast settings to IAS_ interface
    print("Universal Plot has analysis specific settings? ", UnivPlot.HasAnalysisSpecificSettings)
    # Above is False; Universal Plot Settings must be changed via ModifySettings (changing a config (.cfg) file)
    cfg = zosapi.TheApplication.ZemaxDataDir + "\\Configs\\UNI.CFG"
    UnivPlot_Set.Save()  # Create new .cfg file, named "UNI.CFG" in \Configs\ folder
    UnivPlot_Set.ModifySettings(cfg, 'UN1_SURFACE', TheSystem.LDE.NumberOfSurfaces - 2)
    UnivPlot_Set.ModifySettings(cfg, 'UN1_STARTVAL', Surf_4.Thickness - 0.4 / 25.4)  # Change universal plot settings
    UnivPlot_Set.ModifySettings(cfg, 'UN1_STOPVAL', Surf_4.Thickness + 0.1 / 25.4)
    UnivPlot_Set.ModifySettings(cfg, 'UN1_STEPS', 20)
    UnivPlot_Set.ModifySettings(cfg, 'UN1_PAR1', 10)
    UnivPlot_Set.ModifySettings(cfg, 'UN1_OPERAND', "RSRE")
    # For ModifySettings keycodes (UN1_STARTVAL, UN1_STEPS, etc.), see MODIFYSETTINGS page in ZPL>keywords help files
    # LoadFrom allows you to load any CFG file, not just default; not available via GUI
    UnivPlot_Set.LoadFrom(cfg)
    #! [e11s07_py]

    #! [e11s08_py]
    # Open Spot Diagram to See Result!
    newSpot = zosapi.TheSystem.Analyses.New_StandardSpot()
    print("Spot has analysis specific settings? ", newSpot.HasAnalysisSpecificSettings)  # True; no ModifySettings
    newSettings = newSpot.GetSettings()
    spotSet = CastTo(newSettings, "IAS_Spot")  # Cast to IAS_Spot interface; enables access to Spot Diagram properties
    spotSet.RayDensity = 15
    newSpot.ApplyAndWaitForCompletion()
    #! [e11s08_py]

    # save!
    TheSystem.Save()

    # This will clean up the connection to OpticStudio.
    # Note that it closes down the server instance of OpticStudio, so you for maximum performance do not do
    # this until you need to.
    del zosapi
    zosapi = None
