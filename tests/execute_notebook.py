"""Execute notebook cells sequentially using IPython, without network sockets.

Useful in restricted runners that cannot start a Jupyter ZeroMQ kernel.
Stores text/table outputs; figure outputs are regenerated interactively and are
also available in results/. This does not simulate a hosted Colab session.
"""
from pathlib import Path
import os
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

root = Path(__file__).resolve().parents[1]
path = root / 'notebooks/temperature_air_pollution_analysis.ipynb'
nb = nbformat.read(path, as_version=4)
os.chdir(path.parent)
shell = InteractiveShell.instance()
count = 0
for cell in nb.cells:
    if cell.cell_type != 'code':
        continue
    count += 1
    with capture_output() as captured:
        execution = shell.run_cell(cell.source, store_history=True)
    if execution.error_before_exec or execution.error_in_exec:
        raise RuntimeError(f'Notebook code cell {count} failed') from (execution.error_before_exec or execution.error_in_exec)
    cell.execution_count = count
    cell.outputs = []
    for name in ['stdout', 'stderr']:
        content = getattr(captured, name)
        if content:
            cell.outputs.append(nbformat.v4.new_output('stream', name=name, text=content))
    for output in captured.outputs:
        if 'image/png' not in output.data:
            cell.outputs.append(nbformat.v4.new_output('display_data', data=output.data, metadata=output.metadata))
nbformat.validate(nb)
nbformat.write(nb, path)
print(f'PASS: all {count} code cells executed sequentially in IPython; text/table outputs saved.')
