mkdir served
START "frpc" cmd /c ".\bin\frpc.exe -c .\bin\frpc.ini"
START "uploadserver" cmd /c "cd served && python -m uploadserver --bind 127.0.0.1"
START "FridgeAI Training Server" cmd /c "python main.py"
