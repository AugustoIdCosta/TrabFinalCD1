from pathlib import Path
import runpy

# Arquivo de compatibilidade: redireciona para o script adaptado da base2.
script_dir = Path(__file__).resolve().parent
runpy.run_path(script_dir / "3_knn.py", run_name="__main__")
