"""Tests for the Repo git-revision wrapper (Bandit/git_version.py).

These build throwaway git repositories in temporary directories so the real
``git`` plumbing is exercised (no mocking of subprocess). Each test repo is
created with a deterministic identity and an initial commit.
"""

import subprocess

import pytest

from Bandit.exceptions import BanditError
from Bandit.git_version import Repo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git(repo, *args):
    """Run a git command in *repo*, raising on failure."""
    subprocess.run(['git', *args], cwd=repo, check=True,
                   capture_output=True, text=True)


def _rev_parse(repo, ref='HEAD'):
    """Return the full commit hash for *ref* in *repo*."""
    return subprocess.run(['git', 'rev-parse', ref], cwd=repo,
                          capture_output=True, text=True).stdout.strip()


@pytest.fixture
def git_repo(tmp_path):
    """Create a git repo on branch ``main`` with one commit and a ``v1.0`` tag.

    :returns: Path to the repository working tree
    """
    repo = tmp_path / 'repo'
    repo.mkdir()
    _git(repo, 'init', '-b', 'main')
    _git(repo, 'config', 'user.email', 'test@example.com')
    _git(repo, 'config', 'user.name', 'Test User')
    (repo / 'file.txt').write_text('first\n')
    _git(repo, 'add', '.')
    _git(repo, 'commit', '-m', 'first commit')
    _git(repo, 'tag', 'v1.0')
    return repo


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestRepoConstruction:
    """Repo() validates that the path exists and is a git work tree."""

    def test_valid_repo(self, git_repo):
        """A valid git work tree is accepted and resolved to an absolute path."""
        repo = Repo(git_repo)
        assert repo.path_repo == git_repo.resolve()

    def test_nonexistent_path_raises(self, tmp_path):
        """A path that does not exist raises BanditError."""
        with pytest.raises(BanditError):
            Repo(tmp_path / 'does_not_exist')

    def test_non_git_directory_raises(self, tmp_path):
        """An existing directory that is not a git work tree raises BanditError."""
        plain = tmp_path / 'plain'
        plain.mkdir()
        with pytest.raises(BanditError):
            Repo(plain)


# ---------------------------------------------------------------------------
# branch / commit properties
# ---------------------------------------------------------------------------


class TestRepoBranchAndCommit:
    """The branch and commit properties reflect the checked-out state."""

    def test_branch_on_named_branch(self, git_repo):
        """branch returns the current branch name."""
        assert Repo(git_repo).branch == 'main'

    def test_branch_is_none_when_detached(self, git_repo):
        """branch returns None when HEAD is detached."""
        commit = _rev_parse(git_repo)
        _git(git_repo, 'checkout', commit)
        assert Repo(git_repo).branch is None

    def test_commit_matches_git(self, git_repo):
        """commit returns the full HEAD commit hash."""
        assert Repo(git_repo).commit == _rev_parse(git_repo)


# ---------------------------------------------------------------------------
# resolve_commit
# ---------------------------------------------------------------------------


class TestResolveCommit:
    """resolve_commit turns a branch, tag, or (partial) hash into a full hash."""

    def test_resolve_branch(self, git_repo):
        assert Repo(git_repo).resolve_commit('main') == _rev_parse(git_repo)

    def test_resolve_tag(self, git_repo):
        assert Repo(git_repo).resolve_commit('v1.0') == _rev_parse(git_repo)

    def test_resolve_full_hash(self, git_repo):
        full = _rev_parse(git_repo)
        assert Repo(git_repo).resolve_commit(full) == full

    def test_resolve_partial_hash(self, git_repo):
        full = _rev_parse(git_repo)
        assert Repo(git_repo).resolve_commit(full[:8]) == full

    def test_unknown_revision_returns_none(self, git_repo):
        assert Repo(git_repo).resolve_commit('no-such-ref') is None


# ---------------------------------------------------------------------------
# matches_revision
# ---------------------------------------------------------------------------


class TestMatchesRevision:
    """matches_revision compares a branch/tag/commit against HEAD."""

    @pytest.mark.parametrize('revision', [None, '', '   '])
    def test_empty_revision_always_matches(self, git_repo, revision):
        """An empty or None revision means 'no check' and always matches."""
        assert Repo(git_repo).matches_revision(revision) is True

    def test_matches_current_branch(self, git_repo):
        assert Repo(git_repo).matches_revision('main') is True

    def test_matches_tag(self, git_repo):
        assert Repo(git_repo).matches_revision('v1.0') is True

    def test_matches_full_commit(self, git_repo):
        assert Repo(git_repo).matches_revision(_rev_parse(git_repo)) is True

    def test_matches_partial_commit(self, git_repo):
        assert Repo(git_repo).matches_revision(_rev_parse(git_repo)[:8]) is True

    def test_revision_is_stripped(self, git_repo):
        """Surrounding whitespace on the revision is ignored."""
        assert Repo(git_repo).matches_revision('  main  ') is True

    def test_wrong_branch_does_not_match(self, git_repo):
        assert Repo(git_repo).matches_revision('nonexistent-branch') is False

    def test_wrong_partial_commit_does_not_match(self, git_repo):
        """A hex prefix that is not a prefix of HEAD does not match."""
        assert Repo(git_repo).matches_revision('deadbeef') is False

    def test_detached_head_matches_commit(self, git_repo):
        """A detached checkout matches its full and partial commit, and tag."""
        first = _rev_parse(git_repo)

        # Add a second commit, then detach back onto the first.
        (git_repo / 'file.txt').write_text('second\n')
        _git(git_repo, 'commit', '-am', 'second commit')
        second = _rev_parse(git_repo)
        _git(git_repo, 'checkout', first)

        repo = Repo(git_repo)
        assert repo.branch is None
        assert repo.matches_revision(first) is True
        assert repo.matches_revision(first[:10]) is True
        assert repo.matches_revision('v1.0') is True
        # It must NOT match the branch it is no longer on or the newer commit.
        assert repo.matches_revision(second) is False
        assert repo.matches_revision('main') is False


# ---------------------------------------------------------------------------
# checkout
# ---------------------------------------------------------------------------


class TestCheckout:
    """checkout switches HEAD and rejects invalid revisions."""

    def test_checkout_tag_detaches(self, git_repo):
        """Checking out a tag detaches HEAD at the tagged commit."""
        repo = Repo(git_repo)
        repo.checkout('v1.0')
        assert repo.branch is None
        assert repo.commit == _rev_parse(git_repo, 'v1.0')

    def test_checkout_invalid_raises(self, git_repo):
        with pytest.raises(BanditError):
            Repo(git_repo).checkout('no-such-ref')
