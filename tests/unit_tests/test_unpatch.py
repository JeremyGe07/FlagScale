import os
import sys
from pathlib import Path


PATCH_DIR = Path(__file__).resolve().parents[2] / "tools" / "patch"
if str(PATCH_DIR) not in sys.path:
    sys.path.insert(0, str(PATCH_DIR))

import unpatch


def test_init_submodule_uses_worktree_git_modules_path(mocker, tmp_path):
    main_path = tmp_path / "worktree"
    target_path = main_path / "third_party" / "Megatron-LM"
    worktree_git_modules = (
        tmp_path / "common.git" / "worktrees" / "legacy" / "modules" / "third_party" / "Megatron-LM"
    )

    repo = mocker.Mock()
    repo.working_tree_dir = str(main_path)
    repo.git.rev_parse.return_value = str(worktree_git_modules)
    submodule = mocker.Mock()
    repo.submodule.return_value = submodule

    mocker.patch.object(unpatch, "Repo", return_value=repo)
    mocker.patch.object(
        unpatch.os.path,
        "exists",
        side_effect=lambda path: path in {str(worktree_git_modules), str(target_path)},
    )
    remove_tree = mocker.patch.object(unpatch.shutil, "rmtree")

    unpatch.init_submodule(str(main_path), str(target_path), "Megatron-LM", force=True)

    repo.git.rev_parse.assert_called_once_with("--git-path", os.path.join("modules", "third_party", "Megatron-LM"))
    remove_tree.assert_any_call(str(worktree_git_modules))
    remove_tree.assert_any_call(str(target_path))
    submodule.update.assert_called_once_with(init=True, force=True)
