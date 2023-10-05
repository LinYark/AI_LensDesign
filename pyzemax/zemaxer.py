from win32com.client.gencache import EnsureDispatch, EnsureModule
from win32com.client import CastTo, constants
from win32com.client import gencache
import os

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
        # Using precompiled regular expressions
        gencache.EnsureModule('{EA433010-2BAC-43C4-857C-7AEAC4A8CCE0}', 0, 1, 0)
        gencache.EnsureModule('{F66684D7-AAFE-4A62-9156-FF7A7853F764}', 0, 1, 0)
        
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

    # creates a new API directory

    # Set up primary optical system
    sampleDir = os.path.abspath("./samples/api/python")
    os.makedirs(sampleDir,exist_ok=True)
    TheSystem = zosapi.TheSystem
    TheApplication = zosapi.TheApplication

    
    #! [e01s01_py]
    # Make new file
    testFile = sampleDir + '\\e01_new_file_and_quickfocus.zmx'
    TheSystem.New(False)
    TheSystem.SaveAs(testFile)
    #! [e01s01_py]

    TheSystem.SystemData.MaterialCatalogs.AddCatalog('SCHOTT')

    #! [e01s02_py]
    # Aperture
    TheSystemData = TheSystem.SystemData
    TheSystemData.Aperture.ApertureValue = 40
    #! [e01s02_py]
    
    #! [e01s03_py]
    # Fields
    Field_1 = TheSystemData.Fields.GetField(1)
    NewField_2 = TheSystemData.Fields.AddField(0, 5.0, 1.0)
    #! [e01s03_py]
    
    #! [e01s04_py]
    # Wavelength preset
    slPreset = TheSystemData.Wavelengths.SelectWavelengthPreset(constants.WavelengthPreset_d_0p587)
    #! [e01s04_py]
    
    #! [e01s05_py]
    # Lens data
    TheLDE = TheSystem.LDE
    TheLDE.InsertNewSurfaceAt(2)
    TheLDE.InsertNewSurfaceAt(2)
    Surface_1 = TheLDE.GetSurfaceAt(1)
    Surface_2 = TheLDE.GetSurfaceAt(2)
    Surface_3 = TheLDE.GetSurfaceAt(3)
    #! [e01s05_py]

    #! [e01s06_py]
    # Changes surface cells in LDE
    Surface_1.Thickness = 50.0
    Surface_1.Comment = 'Stop is free to move'
    Surface_2.Radius = 100.0
    Surface_2.Thickness = 10.0
    Surface_2.Comment = 'front of lens'
    Surface_2.Material = 'N-BK7'
    Surface_3.Comment = 'rear of lens'
    #! [e01s06_py]
    
    #! [e01s07_py]
    # Solver
    Solver = Surface_3.RadiusCell.CreateSolveType(constants.SolveType_FNumber)
    SolverFNumber = Solver._S_FNumber
    SolverFNumber.FNumber = 10
    Surface_3.RadiusCell.SetSolveData(Solver)
    #! [e01s07_py]
    
    #! [e01s08_py]
    # QuickFocus
    quickFocus = TheSystem.Tools.OpenQuickFocus()
    quickFocus.Criterion = constants. QuickFocusCriterion_SpotSizeRadial
    quickFocus.UseCentroid = True
    quickFocusCast = CastTo(quickFocus,'ISystemTool')
    quickFocusCast.RunAndWaitForCompletion()
    quickFocusCast.Close()
    #! [e01s08_py]
    
    #! [e01s09_py]
    # Save and close
    TheSystem.Save()
    #! [e01s09_py]
    
    # This will clean up the connection to OpticStudio.
    # Note that it closes down the server instance of OpticStudio, so you for maximum performance do not do
    # this until you need to.
    del zosapi
    zosapi = None