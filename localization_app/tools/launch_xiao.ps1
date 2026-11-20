$ErrorActionPreference = 'Stop'

try {
    $project = Split-Path -Parent $PSScriptRoot
    $startScript = Join-Path $PSScriptRoot 'start_xiao_camera.ps1'
    $options = @{
        Localize = $true
        PrintWidthM = 2.5
        HybridMatching = $true
        HueShift = 3
        HueBandWidth = 15
        SaturationTolerance = 78
    }

    Write-Host 'XIAO 카메라 위치 추정'
    Write-Host '1. 새 Wi-Fi 설정 및 업로드'
    Write-Host '2. 기존 카메라 주소로 바로 실행'
    Write-Host '0. 종료'
    do {
        $choice = (Read-Host '번호를 입력하세요').Trim()
    } until ($choice -in @('1', '2', '0', ''))

    switch ($choice) {
        '1' {
            $wifi = Join-Path $project 'firmware/xiao_camera_stream/wifi_secrets.h'
            if (!(Test-Path -LiteralPath $wifi -PathType Leaf)) {
                $template = Join-Path (Split-Path -Parent $wifi) 'wifi_secrets.example.h'
                Copy-Item -LiteralPath $template -Destination $wifi
            }
            & notepad.exe $wifi
            Write-Host '메모장에서 WIFI_SSID와 WIFI_PASSWORD를 수정하고 저장하세요.'
            Write-Host 'PC를 같은 네트워크에 연결하고 XIAO를 데이터 USB 케이블로 꽂으세요.'
            Write-Host '기존 카메라 창, 브라우저 스트림, 시리얼 모니터를 닫으세요.'
            if ((Read-Host '준비가 끝났으면 Enter, 취소하려면 q').Trim() -ne '') { return }
        }
        '2' {
            $address = (Read-Host 'XIAO IPv4, 카메라 페이지 주소 또는 스트림 주소를 입력하세요').Trim()
            if (!$address) { return }
            $ip = $null
            if ([Net.IPAddress]::TryParse($address, [ref]$ip) -and $ip.AddressFamily -eq 'InterNetwork') {
                $address = "http://${ip}:81/stream"
            }
            Write-Host '다른 카메라 창과 브라우저 스트림을 닫은 상태에서 연결합니다.'
            $options.CameraUrl = $address
        }
        default { return }
    }

    & $startScript @options
} catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
