from flagscale.runner.auto_tuner.search.pp_first_partition import (
    POLICY_PROFILE_TIME_BALANCED,
    build_param_balanced_partition,
    build_profile_time_balanced_partition,
    generate_partition_candidates,
)


def test_profile_time_balanced_partition_skews_toward_output_heavier_stage():
    param_partition = build_param_balanced_partition(
        num_layers=28,
        pp_degree=2,
        world_size=8,
        hidden_size=64,
        padded_vocab_size=32000,
    )

    profile_partition = build_profile_time_balanced_partition(
        num_layers=28,
        pp_degree=2,
        world_size=8,
        hidden_size=64,
        padded_vocab_size=32000,
        seq_length=2048,
    )

    assert param_partition.stage_ranges == ((0, 13), (14, 27))
    assert profile_partition.partition_policy == POLICY_PROFILE_TIME_BALANCED
    assert profile_partition.stage_ranges == ((0, 15), (16, 27))


def test_generate_partition_candidates_supports_profile_time_balanced_policy():
    partitions = generate_partition_candidates(
        num_layers=28,
        pp_degree=2,
        world_size=8,
        partition_policy=[POLICY_PROFILE_TIME_BALANCED],
        max_partitions=1,
        hidden_size=64,
        padded_vocab_size=32000,
        seq_length=2048,
    )

    assert len(partitions) == 1
    assert partitions[0].partition_policy == POLICY_PROFILE_TIME_BALANCED
    assert "heuristic:profile_time_balanced" in partitions[0].legality_flags
