; Zero2YoloYard - Standard Clean Inno Setup Installer Script
; Author: BlueDarkUP (FTC Team 27570)

#define MyAppName "Zero2YoloYard"
#define MyAppVersion "2.0.0"
#define MyAppPublisher "BlueDarkUP (FTC Team 27570)"
#define MyAppURL "https://github.com/BlueDarkUP/Zero2YoloYard"
#define MyAppExeName "Zero2YoloYard.exe"

[Setup]
AppId={{5D9D2B7B-B4B2-4A7B-9D82-2C5E87D913A4}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} v{#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
OutputDir=..\installer_output
OutputBaseFilename=Zero2YoloYard_Installer_v2.0
SetupIconFile=app_icon.ico
UninstallDisplayIcon={app}\app_icon.ico
Compression=lzma2/fast
SolidCompression=no
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=commandline dialog
WizardStyle=modern
DisableWelcomePage=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: checkedonce

[Dirs]
Name: "{app}\checkpoints"; Permissions: users-full
Name: "{app}\checkpoints\sam3"; Permissions: users-full
Name: "{app}\checkpoints\clip"; Permissions: users-full
Name: "{app}\local_storage"; Permissions: users-full
Name: "{app}\local_storage\datasets"; Permissions: users-full
Name: "{app}\local_storage\frames"; Permissions: users-full
Name: "{app}\local_storage\videos"; Permissions: users-full
Name: "{app}\local_storage\models"; Permissions: users-full
Name: "{app}\gkdt_engine"; Permissions: users-full
Name: "{app}\gkdt_engine\output"; Permissions: users-full
Name: "{app}\gkdt_engine\test_real_world"; Permissions: users-full
Name: "{app}\gkdt_engine\test_real_world\object_detector_lib"; Permissions: users-full
Name: "{app}\gkdt_engine\test_real_world\object_detector_lib\weights"; Permissions: users-full

[Files]
Source: "..\dist\Zero2YoloYard.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\dist\settings.json"; DestDir: "{app}"; Flags: ignoreversion onlyifdoesntexist; Permissions: users-full
Source: "app_icon.ico"; DestDir: "{app}"; Flags: ignoreversion
Source: "使用说明.txt"; DestDir: "{app}"; Flags: ignoreversion isreadme

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\app_icon.ico"; WorkingDir: "{app}"
Name: "{group}\使用说明"; Filename: "{app}\使用说明.txt"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\app_icon.ico"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
