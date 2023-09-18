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
        
        # Define a Preferences File.
        # Preferences file is defined on the IZOSAPIConnection interface (prior to connecting to the API)
        # If no PreferencesFile is defined it will use the default OpticStudio.CFG file however changes will not persist between sessions. 
        # If a PreferencesFile is defined, then any changes will save automatically. 
        #! [e26s05_py]
        print('===PreferenceFile===')
        cfgFile = r'C:\Users\Documents\Zemax\Configs\OpticStudio.CFG'
        if os.path.exists(cfgFile):
            self.TheConnection.PreferencesFile = cfgFile
            print('PreferencesFile: ' + self.TheConnection.PreferencesFile)
        else:
            print('Default OpticStudio.CFG preferences used')
        #! [e26s05_py]
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
    
    # Insert Code Here
    TheApplication = zosapi.TheApplication
    TheSystem = zosapi.TheSystem
    
    #! [e26s01_py]
    # Define variables for Project Preferences 
    Preference = TheApplication.Preferences
    PrefG = Preference.General
    Logic = ['False', 'True']

    #Define the enums to a map/dictionary 
    DateTimeTypeLookup = {constants.DateTimeType_None:'None', constants.DateTimeType_DateTime:'DateTime', constants.DateTimeType_Date:'Date'}
    LanguageLookup = {constants.LanguageType_Chinese:'Chinese', constants.LanguageType_English:'English', constants.LanguageType_Japanese:'Japanese'}
    EncodingTypeLookup= {constants.EncodingType_ANSI:'ANSI', constants.EncodingType_Unicode:'Unicode'}
    #! [e26s01_py]
	
    #! [e26s02_py]
    # Read and print the initial settings
    print('\n===Check Settings===')
    print('DateTimeFormat: %s' %DateTimeTypeLookup[PrefG.DateTimeFormat])
    print('Language: %s' %LanguageLookup[PrefG.Language])
    print('ZMXFileEncoding: %s' %EncodingTypeLookup[PrefG.ZMXFileEncoding])
    print('TXTFileEncoding: %s' %EncodingTypeLookup[PrefG.TXTFileEncoding])
    print('UseSessionFiles: %s' %(Logic[PrefG.UseSessionFiles]))
    print('IncludeCalculatedDataInsession: %s' %(Logic[PrefG.IncludeCalculatedDataInSession]))
    print('UpdateMostRecentlyUsedList: %s' %(Logic[PrefG.UpdateMostRecentlyUsedList]))
    print('UserPreferences: %s' %(PrefG.UserPreferences))
    #! [e26s02_py]
	
    #! [e26s03_py]
    # Reset the settings to default
    Preference.ResetToDefaults()
    #! [e26s03_py]
	
    #! [e26s04_py]
    # Set the settings
    PrefG.DateTimeFormat = constants.DateTimeType_None;
    PrefG.ZMXFileEncoding = constants.EncodingType_ANSI;
    PrefG.TXTFileEncoding = constants.EncodingType_ANSI;
    PrefG.UseSessionFiles = False;
    PrefG.IncludeCalculatedDataInSession = False;
    PrefG.UpdateMostRecentlyUsedList = False;
    PrefG.UserPreferences = 'Never gonna tell a lie and hurt you';
    #! [e26s04_py]  
    
    print('\n===Final Settings===')
    print('DateTimeFormat: %s' %DateTimeTypeLookup[PrefG.DateTimeFormat])
    print('Language: %s' %LanguageLookup[PrefG.Language])
    print('ZMXFileEncoding: %s' %EncodingTypeLookup[PrefG.ZMXFileEncoding])
    print('TXTFileEncoding: %s' %EncodingTypeLookup[PrefG.TXTFileEncoding])
    print('UseSessionFiles: %s' %(Logic[PrefG.UseSessionFiles]))
    print('IncludeCalculatedDataInsession: %s' %(Logic[PrefG.IncludeCalculatedDataInSession]))
    print('UpdateMostRecentlyUsedList: %s' %(Logic[PrefG.UpdateMostRecentlyUsedList]))
    print('UserPreferences: %s' %(PrefG.UserPreferences))
      
        
    # This will clean up the connection to OpticStudio.
    # Note that it closes down the server instance of OpticStudio, so you for maximum performance do not do
    # this until you need to.
    del zosapi
    zosapi = None



