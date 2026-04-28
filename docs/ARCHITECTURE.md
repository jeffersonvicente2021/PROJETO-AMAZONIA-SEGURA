# Arquitetura

O projeto agora segue uma organizacao modular:

- `src/main.py`: ponto de entrada do monitoramento em tempo real.
- `src/config.py`: configuracao, leitura do `.env` e paths padrao.
- `src/camera.py`: streams RTSP e snapshot HQ.
- `src/detector.py`: carregamento YOLO, ROI, deteccao e desenho na tela.
- `src/recorder.py`: gravacao de eventos em video.
- `src/database.py`: schema SQLite, metadados e inserts de eventos.
- `src/reports.py`: exportacao CSV/PDF.

Dados, modelos e saidas ficam sob `data/`. Automacoes manuais ficam em `scripts/`.
