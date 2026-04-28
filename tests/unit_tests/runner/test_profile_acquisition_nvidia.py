from subprocess import CompletedProcess
from unittest.mock import patch

import pytest

from flagscale.runner.auto_tuner.profile_acquisition.backends.nvidia import NvidiaProfileBackend


def test_nvidia_backend_collect_device_memory_reads_homogeneous_gpus():
    backend = NvidiaProfileBackend()
    with patch(
        'flagscale.runner.auto_tuner.profile_acquisition.backends.nvidia.subprocess.run',
        return_value=CompletedProcess(
            args=['nvidia-smi'],
            returncode=0,
            stdout='46068\n46068\n',
            stderr='',
        ),
    ):
        measurement = backend.collect_device_memory()

    assert measurement.kind == 'device_memory'
    assert measurement.source == 'nvidia'
    assert measurement.metrics['total_memory_mb'] == 46068
    assert measurement.metadata['device_count'] == 2


def test_nvidia_backend_collect_collectives_parses_key_value_outputs():
    backend = NvidiaProfileBackend()
    outputs = [
        CompletedProcess(
            args=['p2p'],
            returncode=0,
            stdout='bandwidth_gbps=71.5\nlatency_us=2.8\n',
            stderr='',
        ),
        CompletedProcess(
            args=['allreduce'],
            returncode=0,
            stdout='bandwidth_gbps=49.3\nlatency_us=7.2\n',
            stderr='',
        ),
    ]
    with patch(
        'flagscale.runner.auto_tuner.profile_acquisition.backends.nvidia.subprocess.run',
        side_effect=outputs,
    ):
        measurements = backend.collect_collectives(
            p2p_command=['p2p'],
            all_reduce_command=['allreduce'],
            runner='nccl_tests',
        )

    assert measurements['p2p'].metrics['bandwidth_gbps'] == 71.5
    assert measurements['p2p'].metrics['latency_us'] == 2.8
    assert measurements['all_reduce'].metrics['bandwidth_gbps'] == 49.3
    assert measurements['all_reduce'].metrics['latency_us'] == 7.2


def test_nvidia_backend_collect_collectives_requires_explicit_commands():
    backend = NvidiaProfileBackend()

    with pytest.raises(ValueError, match='requires an explicit command'):
        backend.collect_collectives(
            p2p_command=None,
            all_reduce_command=['allreduce'],
            runner='nccl_tests',
        )


def test_nvidia_backend_collect_collectives_can_build_nccl_test_commands():
    backend = NvidiaProfileBackend()
    outputs = [
        CompletedProcess(
            args=['p2p_bw'],
            returncode=0,
            stdout=(
                "# size count time algbw busbw error\n"
                "8 2 float sum 2.8 1.0 1.1 0\n"
                "1048576 262144 float sum 18.5 70.2 71.5 0\n"
            ),
            stderr='',
        ),
        CompletedProcess(
            args=['all_reduce_perf'],
            returncode=0,
            stdout=(
                "# size count type redop root time algbw busbw error time algbw busbw error\n"
                "8 2 float sum 0 7.2 0.5 0.6 0 7.0 0.5 0.7 0\n"
                "1048576 262144 float sum 0 31.0 48.7 49.3 0 30.8 48.9 49.1 0\n"
            ),
            stderr='',
        ),
    ]
    with patch(
        'flagscale.runner.auto_tuner.profile_acquisition.backends.nvidia.subprocess.run',
        side_effect=outputs,
    ) as run_mock:
        measurements = backend.collect_collectives(
            p2p_command=None,
            all_reduce_command=None,
            runner='nccl_tests',
            nccl_tests_bin_dir='/opt/nccl-tests/build',
            ngpus=2,
        )

    first_call = run_mock.call_args_list[0].args[0]
    second_call = run_mock.call_args_list[1].args[0]
    assert first_call[:2] == ['/opt/nccl-tests/build/p2p_bw', '-b']
    assert second_call[:2] == ['/opt/nccl-tests/build/all_reduce_perf', '-b']
    assert measurements['p2p'].metrics['bandwidth_gbps'] == 71.5
    assert measurements['p2p'].metrics['latency_us'] == 2.8
    assert measurements['all_reduce'].metrics['bandwidth_gbps'] == 49.3
    assert measurements['all_reduce'].metrics['latency_us'] == 7.2


def test_nvidia_backend_collect_collectives_can_build_torch_runner_commands():
    backend = NvidiaProfileBackend()
    outputs = [
        CompletedProcess(
            args=['torchrun-p2p'],
            returncode=0,
            stdout='bandwidth_gbps=71.5\nlatency_us=2.8\n',
            stderr='',
        ),
        CompletedProcess(
            args=['torchrun-allreduce'],
            returncode=0,
            stdout='bandwidth_gbps=49.3\nlatency_us=7.2\n',
            stderr='',
        ),
    ]
    with patch(
        'flagscale.runner.auto_tuner.profile_acquisition.backends.nvidia.subprocess.run',
        side_effect=outputs,
    ) as run_mock:
        measurements = backend.collect_collectives(
            p2p_command=None,
            all_reduce_command=None,
            runner='torch_nccl',
            ngpus=2,
        )

    first_call = run_mock.call_args_list[0].args[0]
    second_call = run_mock.call_args_list[1].args[0]
    assert first_call[:6] == [
        first_call[0],
        '-m',
        'torch.distributed.run',
        '--standalone',
        '--nnodes=1',
        '--nproc_per_node=2',
    ]
    assert '--collective' in first_call
    assert '--collective' in second_call
    assert measurements['p2p'].metrics['bandwidth_gbps'] == 71.5
    assert measurements['all_reduce'].metrics['latency_us'] == 7.2


def _topology_aware_collective_outputs():
    return [
        CompletedProcess(args=['topo'], returncode=0, stdout=_topology_output(), stderr=''),
        _collective_process('p2p-pix', bandwidth_gbps=210.0, latency_us=3.1),
        _collective_process('p2p-sys', bandwidth_gbps=95.0, latency_us=11.4),
        _collective_process('ar2', bandwidth_gbps=154.0, latency_us=8.5),
        _collective_process('ar4', bandwidth_gbps=121.0, latency_us=13.2),
    ]


def _topology_output():
    return (
        '        GPU0 GPU1 GPU2 GPU3 CPU Affinity NUMA Affinity\n'
        'GPU0     X   PIX  SYS  SYS  0-47         0\n'
        'GPU1    PIX   X   SYS  SYS  0-47         0\n'
        'GPU2    SYS  SYS   X   PIX  48-95        1\n'
        'GPU3    SYS  SYS  PIX   X   48-95        1\n'
    )


def _collective_process(args, bandwidth_gbps, latency_us):
    output = f'bandwidth_gbps={bandwidth_gbps}\nlatency_us={latency_us}\n'
    return CompletedProcess(args=[args], returncode=0, stdout=output, stderr='')


def test_nvidia_backend_collect_collectives_builds_pair_class_and_group_size_runs():
    backend = NvidiaProfileBackend()

    with patch(
        'flagscale.runner.auto_tuner.profile_acquisition.backends.nvidia.subprocess.run',
        side_effect=_topology_aware_collective_outputs(),
    ) as run_mock:
        measurements = backend.collect_collectives(
            p2p_command=None,
            all_reduce_command=None,
            runner='torch_nccl',
            p2p_pair_classes='auto',
            all_reduce_group_sizes=(2, 4),
        )

    pix_call = run_mock.call_args_list[1]
    sys_call = run_mock.call_args_list[2]
    ar2_call = run_mock.call_args_list[3]
    ar4_call = run_mock.call_args_list[4]
    assert '--nproc_per_node=2' in pix_call.args[0]
    assert '--nproc_per_node=2' in ar2_call.args[0]
    assert '--nproc_per_node=4' in ar4_call.args[0]
    assert pix_call.kwargs['env']['CUDA_VISIBLE_DEVICES'] == '0,1'
    assert sys_call.kwargs['env']['CUDA_VISIBLE_DEVICES'] == '0,2'
    assert measurements['p2p_classes']['pix'].metrics['bandwidth_gbps'] == 210.0
    assert measurements['p2p_classes']['sys'].metadata['gpu_pair'] == [0, 2]
    assert measurements['all_reduce_profiles'][2].metrics['latency_us'] == 8.5
    assert measurements['all_reduce_profiles'][4].metrics['bandwidth_gbps'] == 121.0
    assert measurements['p2p'].metrics['bandwidth_gbps'] == 95.0
    assert measurements['all_reduce'].metrics['latency_us'] == 13.2


def test_nvidia_backend_collect_collectives_rejects_unknown_runner():
    backend = NvidiaProfileBackend()

    with pytest.raises(ValueError, match='Unknown collective runner'):
        backend.collect_collectives(
            p2p_command=None,
            all_reduce_command=None,
            runner='unknown',
        )
