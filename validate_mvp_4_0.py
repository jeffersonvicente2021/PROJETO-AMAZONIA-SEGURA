# validate_mvp_4_0.py
import os, sys, glob, json, sqlite3, datetime
from pathlib import Path

BASE = Path("data_events")
today = datetime.datetime.now().strftime("%Y-%m-%d")
DAY = BASE / today

def fail(msg):
    print(f"[FAIL] {msg}")
    sys.exit(2)

def warn(msg):
    print(f"[WARN] {msg}")

def ok(msg):
    print(f"[OK] {msg}")

# 1) Estrutura base
if not BASE.exists():
    fail("Pasta data_events/ não existe. Rode o script principal primeiro.")
ok("data_events/ existe")

events_db = BASE / "events.db"
if not events_db.exists():
    warn("events.db não encontrado ainda (sem eventos?). Será criado ao primeiro evento.")
else:
    ok("events.db existe")

if not DAY.exists():
    warn(f"Pasta do dia {today} não existe ainda (sem eventos hoje?).")
else:
    ok(f"Pasta do dia {today} existe")

# 2) Subpastas
for sub in ["videos", "images", "meta"]:
    p = DAY / sub
    if DAY.exists():
        if not p.exists():
            warn(f"Subpasta {sub}/ não existe em {DAY}. O script deveria criar on-demand.")
        else:
            ok(f"Subpasta {sub}/ existe ({len(list(p.glob('*')))} arquivos)")

# 3) Amostras de arquivos
def latest(pattern):
    files = sorted(glob.glob(str(pattern)), key=os.path.getmtime)
    return files[-1] if files else None

if DAY.exists():
    last_mp4 = latest(DAY / "videos" / "event_*.mp4")
    last_img = latest(DAY / "images" / "snapshot_*_HQ.jpg")
    last_json = latest(DAY / "meta" / "meta_*.json")

    if last_mp4:
        size = os.path.getsize(last_mp4)
        ok(f"Último MP4: {os.path.basename(last_mp4)} ({size} bytes)")
        if size < 100*1024:
            warn("MP4 muito pequeno (<100KB). Pode ter cortado cedo demais.")
    else:
        warn("Nenhum MP4 encontrado hoje ainda.")

    if last_img:
        ok(f"Última imagem HQ: {os.path.basename(last_img)}")
    else:
        warn("Nenhuma imagem HQ encontrada hoje ainda.")

    if last_json:
        ok(f"Último meta JSON: {os.path.basename(last_json)}")
        try:
            with open(last_json, "r", encoding="utf-8") as f:
                meta = json.load(f)
            keys = list(meta.keys())[:6]
            ok(f"Meta JSON legível. Chaves: {keys}")
        except Exception as e:
            fail(f"Erro lendo JSON: {e}")
    else:
        warn("Nenhum meta JSON encontrado hoje ainda.")

# 4) Banco: sanity check
if events_db.exists():
    try:
        con = sqlite3.connect(events_db)
        cur = con.cursor()
        # Confere tabela
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {r[0] for r in cur.fetchall()}
        if "events" not in tables:
            warn("Tabela 'events' não encontrada. Será criada no primeiro commit do script.")
        else:
            ok("Tabela 'events' encontrada.")
            cur.execute("SELECT COUNT(*) FROM events;")
            total = cur.fetchone()[0]
            ok(f"Total de eventos: {total}")
            cur.execute("SELECT id, class, conf, started_at, ended_at FROM events ORDER BY id DESC LIMIT 3;")
            rows = cur.fetchall()
            for r in rows:
                print("  -", r)
    except Exception as e:
        fail(f"Erro no SQLite: {e}")
    finally:
        try: con.close()
        except: pass

print("\nResumo:")
print("Se você viu pelo menos 1 MP4, 1 imagem HQ e 1 meta JSON do dia + events.db presente/tabela 'events', tá pronto pra coletar.")
