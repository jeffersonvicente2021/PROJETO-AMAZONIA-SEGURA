# Projeto Amazonia Segura

Pipeline local de visao computacional com YOLOv8, OpenCV e SQLite para detectar embarcacoes em camera IP, registrar eventos e gerar relatorios.

## Estrutura

```text
src/      codigo principal
scripts/  automacoes e utilitarios
docs/     documentacao
data/     dados, eventos, modelos, datasets e resultados
tests/    testes automatizados
```

## Instalar

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Crie `.env` na raiz usando `.env.example` como base.

## Rodar

```powershell
python -m src.main
```

## Relatorios

```powershell
python -m src.reports --tipo diario
python -m src.reports --tipo semanal
python -m src.reports --tipo mensal
```

## Validar

```powershell
python scripts/validate_mvp.py
```
