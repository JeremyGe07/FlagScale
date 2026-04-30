class NoSegmentStageCandidatesError(ValueError):
    """Raised when a stage cannot form any executable segment-local candidate."""


class NoSegmentAssignmentCandidatesError(ValueError):
    """Raised when a partition cannot form any segment DP assignment candidate."""
