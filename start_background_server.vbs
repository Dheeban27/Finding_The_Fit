' Launch StyleFinder and Named Cloudflare Tunnel silently in the background
Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

' 1. Start StyleFinder Python Server
WshShell.Run "cmd /c \"cd /d E:\\AI\\Clothing_Retrieval && .venv\\Scripts\\activate.bat && python cir_server.py --host 0.0.0.0 --port 8000\"", 0, False

WScript.Sleep 4000

' 2. Start Named Cloudflare Tunnel (assumes tunnel "stylefinder" already exists)
WshShell.Run "cmd /c \"cd /d E:\\AI\\Clothing_Retrieval && cloudflared.exe tunnel run stylefinder\"", 0, False

' Wait a few seconds for the tunnel to become active
WScript.Sleep 7000

' Create a shortcut to the static, permanent URL (edit the line below with your real domain)
Dim staticUrl
staticUrl = "https://drj.today"  ' <-- replace with your actual DNS name
If staticUrl <> "" Then
    Set linkFile = fso.CreateTextFile("Open_StyleFinder.url", True)
    linkFile.WriteLine "[InternetShortcut]"
    linkFile.WriteLine "URL=" & staticUrl
    linkFile.Close
    WshShell.Popup "StyleFinder is Live!\nLink: " & staticUrl, 10, "StyleFinder Ready", 64
Else
    WshShell.Popup "StyleFinder started! Update the VBS with your permanent URL.", 8, "StyleFinder Started", 64
End If
