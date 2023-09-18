from win32com.client.gencache import EnsureDispatch, EnsureModule
from win32com.client import CastTo, constants
from win32com.client import gencache

import os

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

    #! [e09s01_py]
    # Open new NS system and save
    TheSystem = TheApplication.CreateNewSystem(constants.SystemType_NonSequential)    # Create New NSC File
    filename = TheApplication.SamplesDir + "\\API\\Python\\e09_NSC_CAD.zmx"   # Define file path and name
    TheSystem.SaveAs(filename)  # Save New NSC File
    #! [e09s01_py]
    
    #! [e09s02_py]
    # Insert CAD object
    NSCE = TheSystem.NCE
    Obj1 = NSCE.GetObjectAt(1)
    Obj1.ZPosition = -5
    Obj1_Type = Obj1.GetObjectTypeSettings(constants.ObjectType_CADPartSTEPIGESSAT)  # create CAD object type
    # print(Obj1_Type.GetFileNames1())  # return names of valid solidworks part files in proper directory
    # Set object 1 as CAD object if CAD file exists in proper directory
    if os.path.isfile(TheApplication.ObjectsDir + "\\CAD Files\\ExtPoly.stp"):  # Check if the CAD part exists in correct directory
        Obj1_Type.FileName1 = 'ExtPoly.stp'  # set CAD file to be used (file must be in valid directory)
        Obj1.ChangeType(Obj1_Type)  # Set Object 1 as the previously specified CAD file
    else:
        raise ImportError("CAD file not found")
    #! [e09s02_py]

    #! [e09s03_py]
    # Expose CAD parts in NS Component Editor
    Obj1_CAD = Obj1.CADData  # Retrieve CAD data
    if Obj1_CAD.IsCADAvailable:
        Obj1_CAD.SurfaceTolerance = 2  # Set surface tolerance (2 corresponds to 10^-6 lens units)
        Obj1_CAD.SetAllPartsExposed(True)  # expose all CAD parameters in NS Component Editor
        print(Obj1.AvailableParameters())  # Show all available parameters (should show exposed CAD parameters)
    #! [e09s03_py]

    #! [e09s04_py]
    #Change Face Values of CAD part
    if Obj1_CAD.HasFaceData:
        Obj1_CAD.FaceMode = 1  # Set face mode to "Use Angles of Normal Vectors"
        Obj1_CAD.FaceAngle = 3.1415  # Change setting for face mode
        if Obj1_CAD.NumberOfSurfaces > 1:
            Obj1_CAD.SetSurfaceFace(0, Obj1_CAD.NumberOfSurfaces-1)  # set face 0 to the last surface (combine 2 faces)
    #! [e09s04_py]

    #! [e09s05_py]
    POBfile = open(TheApplication.ObjectsDir + "\\Polygon Objects\\API_cube_demo.POB", "w")  # Open new POB file
    # Append new POB file with polygon definitions (see help files for syntax information)
    POBfile.write("! A cube" '\n')
    POBfile.write("! front face vertices" '\n' "V 1 -1 -1 0" '\n' "V 2 1 -1 0" '\n' "V 3 1 1 0" '\n' "V 4 -1 1 0" '\n')
    POBfile.write("! back face vertices" '\n' "V 5 -1 -1 2" '\n' "V 6 1 -1 2" '\n' "V 7 1 1 2" '\n' "V 8 -1 1 2" '\n')
    POBfile.write("! Front" '\n' "R 1 2 3 4 0 0" '\n' "! Back" '\n' "R 5 6 7 8 0 0" '\n')
    POBfile.write("! Top" '\n' "R 4 3 7 8 0 0" '\n' "! Bottom" '\n' "R 1 2 6 5 0 0" '\n')
    POBfile.write("! Left Side" '\n' "R 1 4 8 5 0 0" '\n' "! Right Side" '\n' "R 2 3 7 6 0 0")
    POBfile.close()
    Obj2 = NSCE.InsertNewObjectAt(2)  # Add new line to NSCE
    Obj2_Type = Obj2.GetObjectTypeSettings(constants.ObjectType_PolygonObject)
    Obj2_Type.FileName1 = "API_cube_demo.POB"
    Obj2.ChangeType(Obj2_Type)
    #! [e09s05_py]

    TheSystem.Analyses.New_Analysis(constants.AnalysisIDM_NSC3DLayout)
    TheSystem.Save()  # Save New NSC File

    # This will clean up the connection to OpticStudio.
    # Note that it closes down the server instance of OpticStudio, so you for maximum performance do not do
    # this until you need to.
    del zosapi
    zosapi = None
