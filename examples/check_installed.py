"""Check the wheel outside the checkout, including every test and CLI report."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import venv

ROOT = Path(__file__).resolve().parents[1]
ENV = os.environ.copy()
ENV.pop('PYTHONPATH', None)
MODULES = ['transportlab', 'transportlab.__main__', 'transportlab.cli',
           'transportlab.core', 'transportlab.demo', 'transportlab.audit',
           'transportlab.sensitivity', 'transportlab.marginal']
COMMANDS = [['demo']]
COMMANDS += [[name, 'examples/points.json'] for name in
             ['solve', 'audit', 'assignment', 'push']]
COMMANDS += [['sensitivity', 'examples/direction.json', '--iterations', '3']]


def call(args, cwd, env=ENV):
    return subprocess.check_output(args, cwd=cwd, env=env, text=True)


with tempfile.TemporaryDirectory() as directory:
    work = Path(directory)
    wheels = work / 'wheels'
    wheels.mkdir()
    call([sys.executable, '-m', 'pip', 'wheel', '--no-deps', '-w', str(wheels),
          str(ROOT)], work)
    built = list(wheels.glob('*.whl'))
    assert len(built) == 1, built
    venv.create(work / 'env', with_pip=True)
    py = str(work / 'env/bin/python')
    call([py, '-m', 'pip', 'install', '--no-deps', str(built[0])], work)
    outside = work / 'outside'
    outside.mkdir()
    for name in ['tests', 'examples', 'assets']:
        shutil.copytree(ROOT / name, outside / name)
    # Data-only fixture directory: no __init__.py or module source to shadow installs.
    shutil.copytree(ROOT / 'transportlab/artifacts', outside / 'transportlab/artifacts')
    suite = '''
import importlib,importlib.util,json,pathlib,sys,unittest
outside=pathlib.Path.cwd().resolve()
sys.path.insert(0,str(outside))
names=json.loads(sys.argv[1])
source=pathlib.Path(sys.argv[2]).resolve()
prefix=pathlib.Path(sys.prefix).resolve()
for name in names:
    if name != 'transportlab.__main__':importlib.import_module(name)
def check():
    for name in names:
        origin=pathlib.Path(importlib.util.find_spec(name).origin).resolve()
        assert origin.is_relative_to(prefix),(name,str(origin))
        assert not origin.is_relative_to(source),(name,str(origin))
        assert 'site-packages' in origin.parts,(name,str(origin))
check()
tests=unittest.defaultTestLoader.discover(str(outside/'tests'))
result=unittest.TextTestRunner(verbosity=2).run(tests)
assert result.testsRun>0 and result.wasSuccessful()
check()
print(json.dumps(dict(tests=result.testsRun,installed_origins=len(names),
                     checked_before_and_after=True)))
'''
    print(call([py, '-I', '-c', suite, json.dumps(MODULES), str(ROOT)], outside))
    reference = work / 'reference'
    reference.mkdir()
    shutil.copytree(ROOT / 'examples', reference / 'examples')
    source_env = ENV.copy()
    source_env['PYTHONPATH'] = str(ROOT)
    expected = [json.loads(call([sys.executable, '-m', 'transportlab'] + args,
                                reference, source_env)) for args in COMMANDS]
    for entry in [[py, '-I', '-m', 'transportlab'], [str(work / 'env/bin/transportlab')]]:
        for args, report in zip(COMMANDS, expected):
            # Compare every report leaf, including the full training history.
            actual_text = call(entry + args, outside)
            assert json.loads(actual_text) == report, (entry, args)
            if args[0] == 'demo':
                (outside / 'demo.json').write_text(actual_text)
    for name in ['demo', 'pushforward']:
        print(call([py, '-I', str(outside / 'examples' / ('render_' + name + '.py')),
                    'demo.json', '--check'], outside))
    # Retain every installed marginal control and CLI refusal from the old gate.
    controls = sorted((outside / 'transportlab/artifacts').glob('marginal-*.json'))
    assert len(controls) == 12
    for file in controls:
        control = json.loads(file.read_text())
        (outside / 'marginal-input.json').write_text(json.dumps(control['input']))
        actual = json.loads(call([py, '-I', '-m', 'transportlab.marginal',
                                  'marginal-input.json'], outside))
        assert actual == control['report'], file.name
        bad = dict(control['input'], iterations=501)
        rejected = subprocess.run([py, '-I', '-m', 'transportlab.marginal', '-'],
                                  input=json.dumps(bad), capture_output=True,
                                  text=True, cwd=outside, env=ENV)
        assert rejected.returncode == 2 and not rejected.stdout
    print(json.dumps(dict(marginal_controls=12, cli_refusals=12)))
    print(json.dumps(dict(complete_reports=len(COMMANDS), entrypoints=2,
                          reproduced_figures=2)))
