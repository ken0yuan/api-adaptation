"""Offline checks for background GPU polling and resumable bridge calls."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
# This draft overlays the existing skill. Its unchanged relay supplies SSH helpers.
sys.path.append(str(Path(__file__).resolve().parents[4] / 'work/api-adaptation/scripts'))
import step1_wsl_gpu as gpu
import step1_wsl_bridge as bridge


class GPUPollingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name)
        (self.output / 'submission.json').write_text(json.dumps({'run_id': 123}), encoding='utf-8')
        (self.output / 'manifest.json').write_text('{}', encoding='utf-8')
        (self.output / 'evidence/artifacts-1').mkdir(parents=True)

    def run_state(self, status):
        return {'id': 123, 'status': status, 'run_attempt': 1,
                'conclusion': 'success' if status == 'completed' else None}

    def test_watch_polls_one_run_until_completion_and_persists_progress(self):
        states = [self.run_state(status) for status in ['queued', 'in_progress', 'completed']]
        progress = []
        def observe_sleep(seconds):
            self.assertEqual(seconds, 10)
            progress.append(json.loads((self.output / 'run.json').read_text())['status'])
        with (
            patch.object(gpu, 'api', side_effect=states) as api,
            patch.object(gpu.time, 'sleep', side_effect=observe_sleep),
            patch.object(gpu, 'analyze', return_value={'execution': 'pass'}),
            redirect_stdout(io.StringIO()),
        ):
            result = gpu.fetch(self.output, None)
        self.assertEqual(result['execution'], 'pass')
        self.assertEqual(progress, ['queued', 'in_progress'])
        self.assertEqual([call.args[0] for call in api.call_args_list], ['actions/runs/123'] * 3)
        self.assertEqual(json.loads((self.output / 'run.json').read_text())['status'], 'completed')
        self.assertTrue((self.output / 'evidence/report.json').is_file())

    def test_bounded_fetch_remains_available(self):
        with (
            patch.object(gpu, 'api', return_value=self.run_state('in_progress')),
            patch.object(gpu.time, 'sleep') as sleep,
            redirect_stdout(io.StringIO()),
        ):
            self.assertIsNone(gpu.fetch(self.output, 0))
        sleep.assert_not_called()
        self.assertEqual(json.loads((self.output / 'run.json').read_text())['status'], 'in_progress')

    def test_bridge_cli_forwards_watch_for_the_existing_request(self):
        argv = ['bridge', 'assist', '--source-ssh', 'source-alias',
                '--request', '/run/step1-request.json', '--local-output', str(self.output), '--watch']
        with patch.object(sys, 'argv', argv), patch.object(bridge, 'assist', return_value=0) as assist:
            self.assertEqual(bridge.main(), 0)
        self.assertEqual(assist.call_args.args[-1], None)
        self.assertEqual(assist.call_args.args[1], '/run/step1-request.json')

    def test_gpu_cli_forwards_watch_for_the_existing_run_directory(self):
        argv = ['gpu', 'fetch', '--output', str(self.output), '--watch']
        with (
            patch.object(sys, 'argv', argv),
            patch.object(gpu, 'fetch', return_value={'execution': 'pass'}) as fetch,
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(gpu.main(), 0)
        fetch.assert_called_once_with(self.output.resolve(), None)


if __name__ == '__main__':
    unittest.main()
