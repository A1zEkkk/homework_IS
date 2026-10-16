"""Выполнить ноутбук и сохранить текстовые результаты и графики в ячейках."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
import base64
import contextlib
import io
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

path = next(Path('.').glob('*.ipynb'))
nb = json.loads(path.read_text(encoding='utf-8'))
namespace = {'__name__': '__main__'}
outputs = []
def show(*args, **kwargs):
    for num in plt.get_fignums():
        buffer = io.BytesIO()
        plt.figure(num).savefig(buffer, format='png', dpi=110, bbox_inches='tight')
        outputs.append({'output_type': 'display_data', 'metadata': {}, 'data': {
            'image/png': base64.b64encode(buffer.getvalue()).decode('ascii'),
            'text/plain': ['<Figure>']}})
    plt.close('all')
plt.show = show
count = 0
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    count += 1
    print(f'Running cell {count}/14', flush=True)
    outputs = []
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(''.join(cell['source']), f'cell_{count}', 'exec'), namespace)
    if stdout.getvalue():
        outputs.append({'output_type': 'stream', 'name': 'stdout', 'text': stdout.getvalue().splitlines(keepends=True)})
    cell.update(execution_count=count, outputs=outputs)
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')

summary = {key: namespace[key] for key in ['accuracy_my', 'accuracy_sk', 'best_k', 'best_components', 'n95', 'final_accuracy']}
summary['variance15'] = float(namespace['cumulative'][14])
summary['grid_best'] = [int(namespace['final_n']), int(namespace['final_k']), float(namespace['accuracy_matrix'][namespace['best']])]
summary['grid_worst'] = [int(namespace['n_components_grid'][namespace['worst'][0]]), int(namespace['k_grid'][namespace['worst'][1]]), float(namespace['accuracy_matrix'][namespace['worst']])]
Path('lab_results.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary), flush=True)
