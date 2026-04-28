import datetime
import glob
import json
import os
import sqlite3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "events"
today = datetime.datetime.now().strftime("%Y-%m-%d")
DAY = BASE / today


def fail(msg):
    print(f"[FAIL] {msg}")
    sys.exit(2)


def warn(msg):
    print(f"[WARN] {msg}")


def ok(msg):
    print(f"[OK] {msg}")


def latest(pattern):
    files = sorted(glob.glob(str(pattern)), key=os.path.getmtime)
    return files[-1] if files else None


if not BASE.exists():
    fail(f"Pasta {BASE} nao existe. Rode o script principal primeiro.")
ok(f"{BASE} existe")

events_db = BASE / "events.db"
if not events_db.exists():
    warn("events.db nao encontrado ainda (sem eventos?). Sera criado ao primeiro evento.")
else:
    ok("events.db existe")

if not DAY.exists():
    warn(f"Pasta do dia {today} nao existe ainda (sem eventos hoje?).")
else:
    ok(f"Pasta do dia {today} existe")

for sub in ["videos", "images", "meta"]:
    path = DAY / sub
    if DAY.exists():
        if not path.exists():
            warn(f"Subpasta {sub}/ nao existe em {DAY}. O script deveria criar on-demand.")
        else:
            ok(f"Subpasta {sub}/ existe ({len(list(path.glob('*')))} arquivos)")

if DAY.exists():
    last_mp4 = latest(DAY / "videos" / "event_*.mp4")
    last_img = latest(DAY / "images" / "snapshot_*_HQ.jpg")
    last_json = latest(DAY / "meta" / "meta_*.json")

    if last_mp4:
        size = os.path.getsize(last_mp4)
        ok(f"Ultimo MP4: {os.path.basename(last_mp4)} ({size} bytes)")
        if size < 100 * 1024:
            warn("MP4 muito pequeno (<100KB). Pode ter cortado cedo demais.")
    else:
        warn("Nenhum MP4 encontrado hoje ainda.")

    if last_img:
        ok(f"Ultima imagem HQ: {os.path.basename(last_img)}")
    else:
        warn("Nenhuma imagem HQ encontrada hoje ainda.")

    if last_json:
        ok(f"Ultimo meta JSON: {os.path.basename(last_json)}")
        try:
            with open(last_json, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
            ok(f"Meta JSON legivel. Chaves: {list(meta.keys())}")
        except Exception as exc:
            fail(f"Erro lendo JSON: {exc}")
    else:
        warn("Nenhum meta JSON encontrado hoje ainda.")

if events_db.exists():
    try:
        con = sqlite3.connect(events_db)
        cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row[0] for row in cur.fetchall()}

        if "events" not in tables:
            warn("Tabela 'events' nao encontrada. Sera criada no primeiro commit do script.")
        else:
            ok("Tabela 'events' encontrada.")
            cur.execute("SELECT COUNT(*) FROM events;")
            ok(f"Total de eventos: {cur.fetchone()[0]}")

            cur.execute("PRAGMA table_info(events);")
            cols = [row[1] for row in cur.fetchall()]
            ok(f"Colunas da tabela events: {cols}")

            expected_cols = ["id", "classes", "conf_max", "ts_start", "ts_end"]
            missing = [col for col in expected_cols if col not in cols]
            if missing:
                warn(f"Colunas esperadas ausentes: {missing}")
            else:
                cur.execute(
                    """
                    SELECT id, classes, conf_max, ts_start, ts_end
                    FROM events
                    ORDER BY id DESC
                    LIMIT 3;
                    """
                )
                rows = cur.fetchall()
                if rows:
                    ok("Ultimos 3 eventos:")
                    for row in rows:
                        print("  -", row)
                else:
                    warn("Tabela 'events' existe, mas ainda esta vazia.")
    except Exception as exc:
        fail(f"Erro no SQLite: {exc}")
    finally:
        try:
            con.close()
        except Exception:
            pass

print("\nResumo:")
print("Se voce viu MP4, imagem HQ, meta JSON e events.db/tabela 'events', esta pronto para coletar.")
