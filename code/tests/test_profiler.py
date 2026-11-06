"""Stage profiler (runs/profiler.py) with a fake nvidia-smi; no GPU needed."""
import sys
import time
import unittest
from types import SimpleNamespace

from assemblyvision.runs.profiler import GpuSampler, Profiler, gpu_processes, parse_sample, sampler_command

# prints one sample every 20 ms: utilisation rises 10, 20, ...; memory 1000 + 10 * i MB
FAKE_SMI = [sys.executable, "-u", "-c",
            "import time\nfor i in range(1, 400):\n    print(f'{10 * (i % 10)}, {1000 + 10 * i}, 24564', flush=True)\n"
            "    time.sleep(0.02)"]


class ProfilerTest(unittest.TestCase):
    def test_parse_sample(self):
        self.assertEqual(parse_sample("37, 5120, 24564"), (37.0, 5120.0))
        self.assertIsNone(parse_sample("[N/A], 5120"))
        self.assertIsNone(parse_sample(""))

    def test_sampler_command_uses_the_gpu_index(self):
        cmd = sampler_command(0, 0.5)
        self.assertIn("-i", cmd)
        self.assertEqual(cmd[cmd.index("-i") + 1], "0")
        self.assertIn("--loop-ms=500", cmd)

    def test_samples_grouped_per_stage(self):
        prof = Profiler(True, command=FAKE_SMI).start()
        with prof.stage("ingest"):
            time.sleep(0.3)
        with prof.stage("pose_body"):
            time.sleep(0.4)
        prof.stop()
        manifest = {"pose_body": {"breakdown_s": {"decode": 1.0, "inference": 2.0}},
                    "overlay": {"breakdown_s": {"render": 0.5, "encode": 0.5, "concat": 0.0}}}
        s = prof.summary(manifest)
        self.assertEqual(list(s["stages"]), ["ingest", "pose_body"])
        self.assertGreater(s["stages"]["pose_body"]["samples"], 3)
        self.assertGreaterEqual(s["stages"]["pose_body"]["gpu_mem_peak_mb"], s["stages"]["ingest"]["gpu_mem_peak_mb"])
        self.assertIsNotNone(s["run"]["gpu_util_mean"])
        self.assertEqual(s["breakdown"]["pose_body"]["inference"], 2.0)
        self.assertIn("overlay", s["breakdown"])
        self.assertAlmostEqual(s["stages"]["pose_body"]["seconds"], 0.4, delta=0.15)
        self.assertTrue(s["timeline"])

    def test_disabled_profiler_samples_nothing(self):
        prof = Profiler(False, gpu_index=0).start()
        with prof.stage("ingest"):
            pass
        prof.stop()
        s = prof.summary({})
        self.assertIsNone(prof.sampler)
        self.assertEqual(s["run"]["samples"], 0)
        self.assertEqual(s["breakdown"], {})

    def test_cpu_run_has_no_sampler(self):
        self.assertIsNone(Profiler(True, gpu_index=-1).sampler)

    def test_missing_command_is_recorded_not_raised(self):
        g = GpuSampler(["/nonexistent/nvidia-smi"]).start()
        g.stop()
        self.assertIn("FileNotFoundError", g.error)

    def test_gpu_processes_filters_by_bus(self):
        def run(cmd, **kw):
            if "--query-compute-apps=gpu_bus_id,pid,process_name,used_memory" in cmd:
                out = ("00000000:01:00.0, 4242, SRBMiner-MULTI, 3000\n"
                       "00000000:03:00.0, 99, python, 324\n")
            else:
                out = "NVIDIA RTX A5000, 00000000:01:00.0, 560.35.05, 24564\n"
            return SimpleNamespace(stdout=out, returncode=0)
        procs = gpu_processes(0, run=run)
        self.assertEqual(procs, [{"pid": 4242, "name": "SRBMiner-MULTI", "used_mb": 3000.0}])

    def test_gpu_processes_raises_without_nvidia_smi(self):
        def run(cmd, **kw):
            raise OSError("no nvidia-smi")
        with self.assertRaises(OSError):
            gpu_processes(0, run=run)


if __name__ == "__main__":
    unittest.main()
