from win32com.client.gencache import EnsureDispatch, EnsureModule
from win32com.client import CastTo, constants
from win32com.client import gencache
import os, time, sys

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

    TheSystem = zosapi.TheSystem
    TheApplication = zosapi.TheApplication
    #! [e21s01_py]
    # Create a new non-sequential file
    TheSystem.New(False)
    TheSystem.MakeNonSequential()
    # Add new catalog MISC
    TheSystem.SystemData.MaterialCatalogs.AddCatalog('MISC')
    # Set Wave #1 to 0.47 micron
    TheSystem.SystemData.Wavelengths.GetWavelength(1).Wavelength = 0.47
    # Use lumens as the source unit
    TheSystem.SystemData.Units.SourceUnits = constants.ZemaxSourceUnits_Lumens
    #! [e21s01_py]

    #! [e21s02_py]
    # Add 4 more objects
    TheNCE = TheSystem.NCE
    TheNCE.AddObject()
    TheNCE.AddObject()
    TheNCE.AddObject()
    TheNCE.AddObject()
    #! [e21s02_py]

    #! [e21s03_py]
    # Set 1st object as a Source File
    Object_1 = TheNCE.GetObjectAt(1)
    Typeset_SourceFile = Object_1.GetObjectTypeSettings(constants.ObjectType_SourceFile)
    Typeset_SourceFile.FileName1 = 'RAYFILE_LB_T67C_100K_190608_ZEMAX.DAT'
    Object_1.ChangeType(Typeset_SourceFile)
    Object_1.GetObjectCell(constants.ObjectColumn_Par1).IntegerValue = 5
    Object_1.GetObjectCell(constants.ObjectColumn_Par2).IntegerValue = 1000
    Object_1.GetObjectCell(constants.ObjectColumn_Par3).DoubleValue = 2.485572
    Object_1.GetObjectCell(constants.ObjectColumn_Par8).DoubleValue = 0.47
    Object_1.GetObjectCell(constants.ObjectColumn_Par9).DoubleValue = 0.47
    #! [e21s03_py]

    #! [e21s04_py]
    # Edit source data of object 1
    # SourcesData includes all the settings in Object Properties > Sources
    Object_1.SourcesData.PrePropagation = -0.2
    Object_1.SourcesData.ArrayType = constants.ArrayMode_Rectangular
    Object_1.SourcesData.ArrayNumberX = 5
    Object_1.SourcesData.ArrayNumberY = 5
    #! [e21s04_py]

    #! [e21s05_py]
    # Set 2nd object as CAD Part: STEP/IGES/SAT
    Object_2 = TheNCE.GetObjectAt(2)
    Typeset_CADPartSTEPIGESSAT = Object_1.GetObjectTypeSettings(constants.ObjectType_CADPartSTEPIGESSAT)
    Typeset_CADPartSTEPIGESSAT.FileName1 = 'LB_T67C_190608_GEOMETRY.STEP'
    Object_2.ChangeType(Typeset_CADPartSTEPIGESSAT)
    #! [e21s05_py]

    #! [e21s06_py]
    # Set Rays Ignore Object = Always for object 2
    # TypeData includes all settings in Object Properties > Type
    Object_2.TypeData.RaysIgnoreObject = constants.RaysIgnoreObjectType_Always
    #! [e21s06_py]

    #! [e21s07_py]
    # Set 3rd object as Cylinder Volume
    Object_3 = TheNCE.GetObjectAt(3)
    Typeset_CylinderVolume = Object_3.GetObjectTypeSettings(constants.ObjectType_CylinderVolume)
    Object_3.ChangeType(Typeset_CylinderVolume)
    # Set positions, material and parameters
    Object_3.GetObjectCell(constants.ObjectColumn_ZPosition).DoubleValue = 0.8
    Object_3.GetObjectCell(constants.ObjectColumn_Material).Value = 'PMMA'
    Object_3.GetObjectCell(constants.ObjectColumn_Par1).DoubleValue = 1.2
    Object_3.GetObjectCell(constants.ObjectColumn_Par2).DoubleValue = 0.1
    Object_3.GetObjectCell(constants.ObjectColumn_Par3).DoubleValue = 1.2
    #! [e21s07_py]

    #! [e21s08_py]
    # Make Face 1 of object 3 has Lambertian scattering properties
    # To set scatter properties, you need to first create "ScatteringSettings" by "CreateScatterModelSettings()" method.
    # And then assign is to object 3 by ChangeScatterModelSettings().
    ScatType_Lam = Object_3.CoatScatterData.GetFaceData(1).CreateScatterModelSettings(constants.ObjectScatteringTypes_Lambertian)
    ScatType_Lam._S_Lambertian.ScatterFraction = 1
    Object_3.CoatScatterData.GetFaceData(1).ChangeScatterModelSettings(ScatType_Lam)
    Object_3.CoatScatterData.GetFaceData(1).NumberOfRays = 1
    #! [e21s08_py]

    #! [e21s09_py]
    # Make object 3 a volume scattering material
    # VolumePhysicsData includes all settings in Object Properties > VolumePhysics.
    # Use Photoluminescence model
    Object_3.VolumePhysicsData.Model = constants.VolumePhysicsModelType_PhotoluminescenceModel
    Photo_setting = Object_3.VolumePhysicsData.ModelSettings._S_PhotoluminescenceModel
    # Use Standard Algorithm
    Photo_setting.BasicAlgorithm = False
    # Set absorption, emission and quantum yield files
    Photo_setting.AbsorptionFile = '_sample_3.ZAS'
    Photo_setting.EmissionFile = '_sample_3.ZES'
    Photo_setting.QuantumYield = '_sample_3.ZQE'
    # set efficiency spectrum to quantum yield
    Photo_setting.EfficiencySpectrum = constants.EfficiencySpectrumType_QuantumYield
    # set photoluminescence parameters
    Photo_setting.ExtinctionCoefficient = 1E+05
    Photo_setting.ExtinctionWavelength = 0.47
    Photo_setting.PLDensity = 3.1E+017
    # Set Model to Ignore Mie Scattering
    Photo_setting.ConsiderMieScattering = False
    #! [e21s09_py]

    #! [e21s10_py]
    # Set 4th object as Standard Lens
    Object_4 = TheNCE.GetObjectAt(4)
    Typeset_StandardLens = Object_4.GetObjectTypeSettings(constants.ObjectType_StandardLens)
    Object_4.ChangeType(Typeset_StandardLens)
    # Set positions
    Object_4.GetObjectCell(constants.ObjectColumn_ZPosition).DoubleValue = 0.9
    # To set solve for any cell, you need to first create a "ISolveData" by "CreateSolveType()" method.
    # And then assign it to the cell.
    Solve_ObjPick = Object_4.GetObjectCell(constants.ObjectColumn_Material).CreateSolveType(constants.SolveType_ObjectPickup)
    Solve_ObjPick._S_ObjectPickup.Object = 3
    # Set parameters
    Object_4.GetObjectCell(constants.ObjectColumn_Material).SetSolveData(Solve_ObjPick)
    Object_4.GetObjectCell(constants.ObjectColumn_Par3).DoubleValue = 1.2
    Object_4.GetObjectCell(constants.ObjectColumn_Par4).DoubleValue = 1.2
    Object_4.GetObjectCell(constants.ObjectColumn_Par5).DoubleValue = 1.2
    Object_4.GetObjectCell(constants.ObjectColumn_Par6).DoubleValue = -1.2
    Object_4.GetObjectCell(constants.ObjectColumn_Par8).DoubleValue = 1.2
    Object_4.GetObjectCell(constants.ObjectColumn_Par9).DoubleValue = 1.2
    #! [e21s10_py]

    #! [e21s11_py]
    # Set 5th object as Detector Color
    Object_5 = TheNCE.GetObjectAt(5)
    Typeset_DetectorColor = Object_5.GetObjectTypeSettings(constants.ObjectType_DetectorColor)
    Object_5.ChangeType(Typeset_DetectorColor)
    # Set positions, material and parameters
    Object_5.GetObjectCell(constants.ObjectColumn_ZPosition).DoubleValue = 7
    Object_5.GetObjectCell(constants.ObjectColumn_Material).Value = 'ABSORB'
    Object_5.GetObjectCell(constants.ObjectColumn_Par1).DoubleValue = 5
    Object_5.GetObjectCell(constants.ObjectColumn_Par2).DoubleValue = 5
    Object_5.GetObjectCell(constants.ObjectColumn_Par3).IntegerValue = 150
    Object_5.GetObjectCell(constants.ObjectColumn_Par4).IntegerValue = 150
    Object_5.GetObjectCell(constants.ObjectColumn_Par6).IntegerValue = 4
    Object_5.GetObjectCell(constants.ObjectColumn_Par7).IntegerValue = 3
    #! [e21s11_py]

    #! [e21s12_py]
    # Open NSC Ray Trace tool and turn on Scatter NSC Rays and Ignore Errors
    RayTraceControl = TheSystem.Tools.OpenNSCRayTrace()
    RayTraceControl.SplitNSCRays = False
    RayTraceControl.ScatterNSCRays = True
    RayTraceControl.UsePolarization = False
    RayTraceControl.IgnoreErrors = True
    RayTraceControl.SaveRays = False

    # Trace rays and report the progress when it's running.
    # Note that, instead an RunAndWaitCompletion(), Run() is used so that
    # the code will just go on without waiting the tracing finishs.
    # We will check the progress of tracing by a while loop.
    # You can check the properties "Progress", which is percentage integer data (1-100)

    print('Starting Tracing...')
    RayTraceControl.ClearDetectors(0)
    baseTool = CastTo(RayTraceControl, 'ISystemTool')
    baseTool.Run()
    while (baseTool.Progress != 100):
        sys.stdout.write("\r" + str(baseTool.Progress) + '%')
        sys.stdout.flush()
        time.sleep(1)

    baseTool.Close()
    print('\nFinished!')
    #! [e21s12_py]

    #! [e21s13_py]
    # Open two detector viewers for showing results in angle space and position space
    # Detector Viewer has its own settings interface: IAS_DetectorViewer.
    # Note that not all analyses have a specific settings interface.
    TheAnalysis = TheSystem.Analyses
    Det1 = TheAnalysis.New_DetectorViewer()
    Det_Set1 = Det1.GetSettings()
    baseSettings = CastTo(Det_Set1, 'IAS_DetectorViewer')
    baseSettings.ShowAs = constants.DetectorViewerShowAsTypes_TrueColor
    baseSettings.Smoothing = 3
    Det1.ApplyAndWaitForCompletion()

    Det2 = TheAnalysis.New_DetectorViewer()
    Det_Set2 = Det2.GetSettings()
    baseSettings = CastTo(Det_Set2, 'IAS_DetectorViewer')
    baseSettings.ShowAs = constants.DetectorViewerShowAsTypes_TrueColor
    baseSettings.Smoothing = 3
    baseSettings.DataType = constants.DetectorViewerShowDataTypes_AngleSpace
    Det2.ApplyAndWaitForCompletion()
    #! [e21s13_py]

    TheSystem.SaveAs(TheApplication.SamplesDir + '\\API\\Python\\e21_White_LED_Phosphor.ZMX')


    # This will clean up the connection to OpticStudio.
    # Note that it closes down the server instance of OpticStudio, so you for maximum performance do not do
    # this until you need to.
    del zosapi
    zosapi = None