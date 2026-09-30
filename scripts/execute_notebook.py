"""Execute and export the notebook using the active project environment."""

import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]
os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
path = ROOT / "notebooks/01_datathon.ipynb"
notebook = nbformat.read(path, as_version=4)
client = NotebookClient(
    notebook, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}
)
client.execute()
nbformat.write(notebook, path)
exporter = HTMLExporter()
exporter.embed_images = True
html, _ = exporter.from_notebook_node(notebook)
path.with_suffix(".html").write_text(html)
print(f"Executed {len(notebook.cells)} cells; exported {path.with_suffix('.html').name}")
