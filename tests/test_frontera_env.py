"""Interrupted Conda transactions, using fake installers and no network access."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FronteraEnvTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root/'scripts').mkdir()
        self.script = self.root/'scripts/frontera_env.sh'
        self.script.write_bytes((ROOT/'scripts/frontera_env.sh').read_bytes())
        for name in ('requirements-p2.txt', 'requirements-p3.txt', 'requirements-plots.txt'):
            (self.root/name).touch()
        self.prefix = self.root/'carbon'
        self.calls = self.root/'calls'
        self.fake_python = self.root/'python-template'
        self.fake_python.write_text(f'#!{sys.executable}\n'+'''import os, sys
from pathlib import Path
with Path(os.environ['CARBON_TEST_CALLS']).open('a') as stream:
    stream.write('python '+repr(sys.argv[1:])+'\\n')
if sys.argv[1:] == ['-m', 'pip', '--version'] and (Path(sys.argv[0]).parent.parent/'broken-pip').exists():
    sys.exit(3)
if sys.argv[1:] == ['-m', 'pip', 'freeze']:
    print('test-package==1.0')
''')
        self.fake_python.chmod(0o755)
        self.conda = self.root/'conda'
        self.conda.write_text(f'#!{sys.executable}\n'+'''import os, shutil, sys
from pathlib import Path
with Path(os.environ['CARBON_TEST_CALLS']).open('a') as stream:
    stream.write('conda '+repr(sys.argv[1:])+'\\n')
if sys.argv[1] == 'create':
    prefix = Path(sys.argv[sys.argv.index('--prefix')+1])
    (prefix/'conda-meta').mkdir(parents=True)
    (prefix/'bin').mkdir()
    shutil.copy2(os.environ['CARBON_TEST_PYTHON'], prefix/'bin/python')
else:
    print('@EXPLICIT')
''')
        self.conda.chmod(0o755)
        self.env = dict(os.environ, CARBON_ENV=str(self.prefix), CONDA_EXE=str(self.conda),
                        CARBON_TEST_CALLS=str(self.calls), CARBON_TEST_PYTHON=str(self.fake_python))

    def run_script(self):
        return subprocess.run(['bash', str(self.script)], env=self.env, cwd=self.root,
                              capture_output=True, text=True)

    def make_prefix(self, python=False, frozen=False):
        (self.prefix/'conda-meta').mkdir(parents=True)
        (self.prefix/'preserve-me').write_text('partial transaction or user content')
        if python:
            (self.prefix/'bin').mkdir()
            shutil.copy2(self.fake_python, self.prefix/'bin/python')
        if frozen:
            (self.prefix/'carbon-pip-freeze.txt').write_text('test-package==1.0\n')

    def test_missing_python_is_backed_up_and_recreated(self):
        self.make_prefix()
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        backup, = self.root.glob('carbon.incomplete.*/prefix')
        self.assertEqual((backup/'preserve-me').read_text(), 'partial transaction or user content')
        self.assertTrue((self.prefix/'bin/python').is_file())
        self.assertIn("conda ['create'", self.calls.read_text())
        self.assertIn('Ready. Python:', result.stdout)

    def test_python_with_missing_pip_is_also_recreated(self):
        self.make_prefix(python=True)
        (self.prefix/'broken-pip').touch()
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        backup, = self.root.glob('carbon.incomplete.*/prefix')
        self.assertTrue((backup/'broken-pip').is_file())
        self.assertFalse((self.prefix/'broken-pip').exists())

    def test_frozen_broken_environment_is_preserved_in_place(self):
        self.make_prefix(frozen=True)
        result = self.run_script()
        self.assertEqual(result.returncode, 2)
        self.assertIn('refusing automatic replacement', result.stdout)
        self.assertTrue((self.prefix/'preserve-me').is_file())
        self.assertEqual(list(self.root.glob('carbon.incomplete.*')), [])
        self.assertFalse(self.calls.exists())

    def test_healthy_frozen_environment_is_not_reinstalled(self):
        self.make_prefix(python=True, frozen=True)
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        calls = self.calls.read_text()
        self.assertNotIn("'install'", calls)
        self.assertNotIn('conda ', calls)
        self.assertEqual(list(self.root.glob('carbon.incomplete.*')), [])
        self.assertTrue((self.prefix/'preserve-me').is_file())

    def test_slurm_spooled_script_uses_submit_directory(self):
        spool = self.root/'spool'; spool.mkdir()
        script = spool/'slurm_script'; script.write_bytes(self.script.read_bytes())
        self.env.update(SLURM_JOB_ID='456', SLURM_SUBMIT_DIR=str(self.root))
        result = subprocess.run(['bash', str(script)], env=self.env, cwd=spool,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertIn('Slurm job: 456', result.stdout)
        log, = (self.root/'out').glob('frontera-env.456.*.log')
        self.assertIn('Ready. Python:', log.read_text())
        self.assertFalse((spool/'out').exists())

    def test_foreign_directory_is_not_modified(self):
        self.prefix.mkdir(); (self.prefix/'keep').write_text('unrelated')
        result = self.run_script()
        self.assertEqual(result.returncode, 2)
        self.assertEqual((self.prefix/'keep').read_text(), 'unrelated')
        self.assertEqual(list(self.root.glob('carbon.incomplete.*')), [])


if __name__ == '__main__':
    unittest.main()
