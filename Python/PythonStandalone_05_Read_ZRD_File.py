from win32com.client.gencache import EnsureDispatch, EnsureModule
from win32com.client import CastTo, constants
from win32com.client import gencache

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
    
    # Insert Code Here

    TheSystem = zosapi.TheSystem
    TheApplication = zosapi.TheApplication
    Logic = ['False', 'True']
    
    # Open file and set Analysis Ryas to only 10
    testFile = TheApplication.SamplesDir + r'\Non-sequential\Miscellaneous\Digital_projector_flys_eye_homogenizer.zmx'
    TheSystem.LoadFile(testFile, False)
    print(TheSystem.SystemFile)
    TheSystem.NCE.GetObjectAt(1).GetObjectCell(constants.ObjectColumn_Par2).IntegerValue = 1

    #! [e27s01_py]
    # Trace and save a ZRD file for test later
    NSCRayTrace = TheSystem.Tools.OpenNSCRayTrace()
    NSCRayTrace.SplitNSCRays = True
    NSCRayTrace.ScatterNSCRays = False
    NSCRayTrace.UsePolarization = True
    NSCRayTrace.IgnoreErrors = True
    NSCRayTrace.SaveRays = True
    NSCRayTrace.SaveRaysFile = 'Digital_projector_flys_eye_homogenizer.ZRD'
    NSCRayTrace.ClearDetectors(0)
    
    NSCRayTraceCast = CastTo(NSCRayTrace,'ISystemTool')
    NSCRayTraceCast.RunAndWaitForCompletion()
    NSCRayTraceCast.Close()
    #! [e27s01_py]

    #! [e27s02_py]
    ZRDReader = TheSystem.Tools.OpenRayDatabaseReader()
    ZRDReader.ZRDFile = TheApplication.SamplesDir + r'\Non-sequential\Miscellaneous\Digital_projector_flys_eye_homogenizer.ZRD'
    print(TheApplication.SamplesDir + r'\Non-sequential\Miscellaneous\Digital_projector_flys_eye_homogenizer.ZRD')

    baseTool = CastTo(ZRDReader, 'ISystemTool')
    baseTool.RunAndWaitForCompletion()
    if baseTool.Succeeded == 0:
        print('Raytracing failed!\n')
        print(baseTool.ErrorMessage)
    else:
        print('Raytracing completed!\n')
    #! [e27s02_py]


    #! [e27s03_py]
    ZRDResult = ZRDReader.GetResults()
    # ReadNExtResult() returns data ray by ray
    success_NextResult, rayNumber, waveIndex, wlUM, numSegments = ZRDResult.ReadNextResult()
    while success_NextResult == True:
        print('\n\n\nsuccess_NextResult: %s, rayNumber: %d, waveIndex: %d, wlUM: %f, numSegments: %d\n\n' %
              (Logic[success_NextResult], rayNumber, waveIndex, wlUM, numSegments))
        segdata = ZRDResult.ReadNextSegmentFull()
        while segdata[0] == True:
            print('''success_NextSegmentFull: %s, segmentLevel: %d, segmentParent: %d,
hitObj: %d, hitFace: %d, insideOf: %d, status: %s,
x: %f, y: %f, z: %f,l: %f, m: %f, n: %f,
exr: %f, exi: %f, eyr: %f, eyi: %f, ezr: %f, ezi: %f,
intensity: %f, pathLength: %f, xybin: %d, lmbin: %d,
xNorm: %f, yNorm: %f, zNorm: %f,
index: %f, startingPhase: %f, phaseOf: %f, phaseAt: %f\n''' %
                  (Logic[segdata[0]], segdata[1], segdata[2], segdata[3], segdata[4], segdata[5], segdata[6],
                   segdata[7], segdata[8], segdata[9], segdata[10], segdata[11], segdata[12], segdata[13],
                   segdata[14], segdata[15], segdata[16], segdata[17], segdata[18], segdata[19], segdata[20],
                   segdata[21], segdata[22], segdata[23], segdata[24], segdata[25], segdata[26], segdata[27],
                   segdata[28], segdata[29]))
            segdata = ZRDResult.ReadNextSegmentFull()
        success_NextResult, rayNumber, waveIndex, wlUM, numSegments = ZRDResult.ReadNextResult()
        
    baseTool.Close()
    #! [e27s03_py]

    
    
    # This will clean up the connection to OpticStudio.
    # Note that it closes down the server instance of OpticStudio, so you for maximum performance do not do
    # this until you need to.
    del zosapi
    zosapi = None



