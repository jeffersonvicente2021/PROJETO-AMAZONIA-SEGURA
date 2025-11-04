@echo off
cd /d C:\Users\Administrador\Documents\PROJETO-AMAZONIA-SEGURA
call .venv\Scripts\activate.bat
cd src
python export_events.py --tipo mensal
