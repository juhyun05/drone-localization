[CmdletBinding()]
param(
    [ValidatePattern('^COM[0-9]+$')]
    [string]$Port,
    [ValidateRange(1, 600)]
    [int]$IpTimeoutSeconds = 45,
    [string]$CameraUrl,
    [switch]$Localize,
    [switch]$GlobalOnly,
    [ValidateSet('map', 'apriltag', 'fusion')]
    [string]$LocalizationMode,
    [ValidateSet('original_map', 'apriltag_map')]
    [string]$MapVersion,
    [string]$CameraConfig,
    [switch]$HybridMatching,
    [ValidateSet('building', 'road', 'fixed', 'adaptive')]
    [string]$MatchingMode,
    [switch]$RoadDebug,
    [ValidateScript({ $_ -gt 0 -and ![double]::IsInfinity($_) })]
    [double]$PrintWidthM,
    [ValidateRange(1, [int]::MaxValue)]
    [int]$OpenCvThreads = 1,
    [ValidateRange(1, [int]::MaxValue)]
    [int]$GuiIntervalMs = 33,
    [ValidateRange(-90, 90)]
    [int]$HueShift = 0,
    [ValidateRange(0, 90)]
    [int]$HueBandWidth = 0,
    [ValidateRange(0, 255)]
    [int]$SaturationTolerance = 55,
    [switch]$BuildOnly,
    [switch]$NoPreview
)

$ErrorActionPreference = 'Stop'

function Find-XiaoPort {
    param([string]$RequestedPort)

    $listing = pnputil.exe /enum-devices /connected /class Ports | Out-String
    $devices = [regex]::Matches(
        $listing, '(?i)USB\\VID_303A&PID_1001[^\r\n]*\r?\n[^\r\n]*\((COM[0-9]+)\)')
    $ports = @($devices | ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique)
    if ($RequestedPort) {
        if ($ports -notcontains $RequestedPort) {
            throw "An Espressif USB-Serial/JTAG device was not found on $RequestedPort."
        }
        return $RequestedPort
    }
    if ($ports.Count -eq 1) { return $ports[0] }
    if ($ports.Count -eq 0) { throw 'Connect the XIAO with a USB data cable and try again.' }
    throw "More than one Espressif port was found ($($ports -join ', ')). Specify -Port COMn."
}

function Wait-XiaoAddress {
    param([string]$SerialPort, [int]$TimeoutSeconds)

    $timer = [Diagnostics.Stopwatch]::StartNew()
    $serial = [IO.Ports.SerialPort]::new($SerialPort, 115200)
    $serial.DtrEnable = $true
    $buffer = ''
    $opened = $false
    $waitingForWifi = $false
    try {
        while ($timer.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
            try {
                if (!$serial.IsOpen) {
                    $serial.Open()
                    if (!$opened) {
                        Write-Host "Reading the camera IP on $SerialPort. If no boot text appears, tap RESET once."
                        $opened = $true
                    }
                }
                $buffer += $serial.ReadExisting()
            } catch [UnauthorizedAccessException], [IO.IOException], [InvalidOperationException] {
                if (!$opened -and $timer.Elapsed.TotalSeconds -ge [Math]::Min(8, $TimeoutSeconds)) {
                    throw "Could not read $SerialPort. Close any serial monitor and check the USB cable."
                }
                # The native USB serial device can briefly disappear during RESET.
                $serial.Dispose()
                $serial = [IO.Ports.SerialPort]::new($SerialPort, 115200)
                $serial.DtrEnable = $true
                Start-Sleep -Milliseconds 150
                continue
            }

            if ($buffer -match 'Camera init failed with error 0x[0-9a-f]+') {
                throw "The XIAO reported a camera initialization failure: $($Matches[0])"
            }
            if ($buffer -match 'WiFi connecting') { $waitingForWifi = $true }
            if ($buffer -match "Camera Ready!\s+Use 'http://([0-9.]+)'") {
                return [Net.IPAddress]::Parse($Matches[1]).IPAddressToString
            }
            if ($buffer.Length -gt 4096) { $buffer = $buffer.Substring($buffer.Length - 4096) }
            Start-Sleep -Milliseconds 100
        }
        if ($waitingForWifi) {
            throw 'No camera IP reported. Check the 2.4 GHz network and Wi-Fi settings, then retry.'
        }
        throw "No Camera Ready message on $SerialPort. Tap RESET while the script is reading the port."
    } finally {
        $serial.Dispose()
    }
}

function Start-CameraViewer {
    param([string]$StreamUrl)

    Write-Output $StreamUrl
    if ($NoPreview) { return }
    if ($Localize) {
        $viewerArgs = @('--live-localization', '--auto-scale', '--illumination-method', 'combined', '--shadow-aware')
        $viewerArgs += @('--hue-shift', $HueShift, '--hue-band-width', $HueBandWidth,
                         '--saturation-tolerance', $SaturationTolerance)
        if ($GlobalOnly) { $viewerArgs += '--global-only' }
        if ($LocalizationMode) { $viewerArgs += @('--localization-mode', $LocalizationMode.ToLowerInvariant()) }
        if ($MapVersion) { $viewerArgs += @('--map-version', $MapVersion.ToLowerInvariant()) }
        if ($CameraConfig) { $viewerArgs += @('--camera-config', $CameraConfig) }
        if ($HybridMatching) { $viewerArgs += '--hybrid-matching' }
        if ($MatchingMode) { $viewerArgs += @('--matching-mode', $MatchingMode.ToLowerInvariant()) }
        if ($RoadDebug) { $viewerArgs += '--road-debug' }
        if ($PrintWidthM -gt 0) {
            $viewerArgs += @('--print-width-m', $PrintWidthM.ToString([Globalization.CultureInfo]::InvariantCulture))
        }
    } else {
        $viewerArgs = @('--camera-test')
    }
    & $python (Join-Path $project 'main.py') @viewerArgs --camera-url $StreamUrl `
        --opencv-threads $OpenCvThreads --gui-interval-ms $GuiIntervalMs
    if ($LASTEXITCODE -ne 0) { throw "The Python camera viewer exited with code $LASTEXITCODE." }
}

function Update-LocalFile {
    param([string]$Path, [string]$Content)

    if (!(Test-Path -LiteralPath $Path -PathType Leaf) -or
        [IO.File]::ReadAllText($Path) -cne $Content) {
        [IO.File]::WriteAllText($Path, $Content)
    }
}

function Initialize-CameraWebServer {
    param([string]$Source, [string]$Sketch, [string]$Settings)

    $sourceIno = Join-Path $Source 'CameraWebServer.ino'
    if (!(Test-Path -LiteralPath $sourceIno -PathType Leaf)) {
        throw 'Install Arduino-ESP32 3.3.5 with Arduino CLI first. See README.md for the installation commands.'
    }
    $ino = [IO.File]::ReadAllText($sourceIno)
    $boardConfig = [IO.File]::ReadAllText((Join-Path $Source 'board_config.h'))
    $ino = $ino.Replace('const char *ssid = "**********";',
                        ('#include "wifi_secrets.h"' + "`n" + 'const char *ssid = WIFI_SSID;'))
    $ino = $ino.Replace('const char *password = "**********";', 'const char *password = WIFI_PASSWORD;')
    $boardConfig = $boardConfig.Replace('#define CAMERA_MODEL_ESP_EYE  // Has PSRAM',
                                        '//#define CAMERA_MODEL_ESP_EYE  // Has PSRAM')
    $boardConfig = $boardConfig.Replace('//#define CAMERA_MODEL_XIAO_ESP32S3', '#define CAMERA_MODEL_XIAO_ESP32S3')

    $null = New-Item -ItemType Directory -Path $Sketch -Force
    foreach ($name in @('app_httpd.cpp', 'camera_index.h', 'camera_pins.h', 'partitions.csv')) {
        Update-LocalFile -Path (Join-Path $Sketch $name) -Content ([IO.File]::ReadAllText((Join-Path $Source $name)))
    }
    Update-LocalFile -Path (Join-Path $Sketch 'CameraWebServer.ino') -Content $ino
    Update-LocalFile -Path (Join-Path $Sketch 'board_config.h') -Content $boardConfig
    Update-LocalFile -Path (Join-Path $Sketch 'wifi_secrets.h') -Content ([IO.File]::ReadAllText($Settings))
}

$project = Split-Path -Parent $PSScriptRoot
$workspace = Split-Path -Parent $project
$python = Join-Path $workspace '.localization-venv/Scripts/python.exe'
if ($env:VIRTUAL_ENV) {
    $activePython = Join-Path $env:VIRTUAL_ENV 'Scripts/python.exe'
    if (Test-Path -LiteralPath $activePython -PathType Leaf) { $python = $activePython }
}
if ($HybridMatching -and $MatchingMode) { throw 'Choose -HybridMatching or -MatchingMode.' }
if ($RoadDebug -and !($HybridMatching -or ($MatchingMode -and $MatchingMode -ne 'building'))) {
    throw '-RoadDebug requires a road matching mode.'
}
if (!$Localize -and ($GlobalOnly -or $HybridMatching -or $MatchingMode -or $RoadDebug -or $PrintWidthM -gt 0 -or
                    $PSBoundParameters.ContainsKey('HueShift') -or
                    $PSBoundParameters.ContainsKey('HueBandWidth') -or
                    $PSBoundParameters.ContainsKey('SaturationTolerance'))) {
    throw '-GlobalOnly, -PrintWidthM, and the Hue and Saturation options require -Localize.'
}
if (!$BuildOnly -and !$NoPreview -and !(Test-Path -LiteralPath $python -PathType Leaf)) {
    throw 'Create the Python virtual environment and install requirements.txt first. See README.md.'
}
if ($CameraUrl) {
    if ($BuildOnly) { throw '-CameraUrl uses an existing stream and cannot be combined with -BuildOnly.' }
    [uri]$parsedUrl = $null
    if (![uri]::TryCreate($CameraUrl, [UriKind]::Absolute, [ref]$parsedUrl) -or
        $parsedUrl.Scheme -ne 'http' -or !$parsedUrl.Host -or $parsedUrl.UserInfo) {
        throw '-CameraUrl requires an HTTP stream URL such as http://BOARD_IP:81/stream.'
    }
    # The official CameraWebServer prints the control page URL on the serial port.
    if ($parsedUrl.HostNameType -eq [UriHostNameType]::IPv4 -and
        $parsedUrl.IsDefaultPort -and $parsedUrl.PathAndQuery -eq '/' -and !$parsedUrl.Fragment) {
        $CameraUrl = "http://$($parsedUrl.Host):81/stream"
    }
    Start-CameraViewer -StreamUrl $CameraUrl
    return
}

$localBuild = Join-Path $workspace '.xiao-build'
$settings = Join-Path $project 'firmware/xiao_camera_stream/wifi_secrets.h'
if (!(Test-Path -LiteralPath $settings -PathType Leaf)) {
    throw 'Copy wifi_secrets.example.h to wifi_secrets.h and enter your local Wi-Fi settings.'
}
$useLocalCli = Test-Path -LiteralPath "$localBuild/arduino-cli.exe" -PathType Leaf
if (!$useLocalCli) {
    $cli = (Get-Command arduino-cli.exe -CommandType Application -ErrorAction SilentlyContinue).Source
    if (!$cli) { throw 'Install Arduino CLI and add it to PATH. See README.md.' }
}
$null = New-Item -ItemType Directory -Path $localBuild -Force

# A shared sketch and build directory must not be used by two runs at once.
try {
    $lock = [IO.File]::Open("$localBuild/start-xiao.lock", [IO.FileMode]::OpenOrCreate,
                           [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
} catch [IO.IOException] {
    throw 'Another XIAO camera build or upload is already running in this workspace.'
}
$drive = $null
$mapped = $false
$arduinoDirs = @{
    ARDUINO_DIRECTORIES_DATA = 'data'
    ARDUINO_DIRECTORIES_DOWNLOADS = 'downloads'
    ARDUINO_DIRECTORIES_USER = 'user'
}
$previousDirs = @{}
try {
    $drive = @('Z:', 'Y:', 'X:', 'W:') |
        Where-Object { !(Get-PSDrive -Name $_.TrimEnd(':') -ErrorAction SilentlyContinue) } |
        Select-Object -First 1
    if (!$drive) { throw 'No free build drive among Z:, Y:, X:, and W:.' }
    $null = subst.exe $drive $workspace
    if ($LASTEXITCODE -ne 0) { throw "Could not map the workspace to $drive." }
    $mapped = $true

    $d = "$drive/.xiao-build"
    if ($useLocalCli) {
        $cli = "$d/arduino-cli.exe"
        foreach ($variable in $arduinoDirs.Keys) {
            $previousDirs[$variable] = [Environment]::GetEnvironmentVariable($variable, 'Process')
            [Environment]::SetEnvironmentVariable($variable, "$d/$($arduinoDirs[$variable])", 'Process')
        }
    }
    $dataDir = (& $cli config get directories.data | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or !$dataDir) { throw 'Arduino CLI data directory could not be determined.' }
    $board = 'esp32:esp32:XIAO_ESP32S3:PSRAM=opi,CDCOnBoot=default'
    $sketch = "$d/examples/CameraWebServer"
    $example = Join-Path $dataDir 'packages/esp32/hardware/esp32/3.3.5/libraries/ESP32/examples/Camera/CameraWebServer'
    Initialize-CameraWebServer -Source $example -Sketch $sketch -Settings $settings
    $output = "$d/official-build"
    Write-Host 'Building the CameraWebServer example...'
    # A compiler error can quote the credential initializer. Keep its output off disk.
    $report = & $cli compile --jobs 8 --fqbn $board --build-path $output $sketch 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "CameraWebServer build failed (exit $LASTEXITCODE). Check the ESP32 board package and wifi_secrets.h syntax."
    }
    $report | Where-Object { $_ -match '^Sketch uses|^Global variables use' } | ForEach-Object { Write-Host $_ }
    if ($BuildOnly) { return }

    $selectedPort = Find-XiaoPort -RequestedPort $Port
    Write-Host "Uploading CameraWebServer to $selectedPort..."
    & $cli upload --port $selectedPort --fqbn "${board},UploadSpeed=115200" --input-dir $output $sketch | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "Upload to $selectedPort failed. Close any serial monitor and retry." }
    $address = Wait-XiaoAddress -SerialPort $selectedPort -TimeoutSeconds $IpTimeoutSeconds
} finally {
    foreach ($variable in $previousDirs.Keys) {
        $previous = $previousDirs[$variable]
        if ($null -eq $previous) {
            Remove-Item -LiteralPath "Env:$variable" -ErrorAction SilentlyContinue
        } else {
            [Environment]::SetEnvironmentVariable($variable, $previous, 'Process')
        }
    }
    if ($mapped) { $null = subst.exe $drive /D }
    $lock.Dispose()
}

Start-CameraViewer -StreamUrl "http://${address}:81/stream"
