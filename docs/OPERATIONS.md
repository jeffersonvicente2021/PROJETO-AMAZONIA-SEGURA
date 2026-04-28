# Operacao

## Monitoramento

```powershell
python -m src.main
```

## Relatorios

```powershell
python -m src.reports --tipo diario
python -m src.reports --tipo semanal
python -m src.reports --tipo mensal
```

Os `.bat` em `scripts/` executam os mesmos comandos a partir da raiz do projeto.

## Validacao

```powershell
python scripts/validate_mvp.py
python -m compileall src scripts tests
```

## Dados

- Eventos: `data/events/`
- Relatorios gerados: `data/reports/`
- Dataset: `data/datasets/barcos_canoas_tefe/`
- Treinos YOLO: `data/runs/`
- Modelos avulsos: `data/models/`
