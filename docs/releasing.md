# Releasing

Scribnic uses [Semantic Versioning](https://semver.org/). While the version is
`0.x`, bump the **minor** version for new features or breaking changes and the
**patch** version for fixes only. The version lives only in `pyproject.toml`;
`scribnic.__version__` and `scribnic --version` read it from the installed
package metadata.

## Milestones

Each planned release has a GitHub milestone named after its tag (`v0.2.0`).
Issues without a target go to the `Backlog` milestone. Close the milestone when the
release is published.

## Steps

1. Start from an up-to-date `main` with CI passing.

2. Bump the version:

   ```bash
   uv version --bump minor   # or: patch
   ```

3. Draft the notes and fold them into `CHANGELOG.md`:

   ```bash
   uvx git-cliff --unreleased --tag "v$(uv version --short)"
   ```

   Rename `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD`, curate the draft into
   it (merge related commits, drop internal noise, add **Breaking changes**), and
   add a new empty `## [Unreleased]` section above it. Update the link references
   at the bottom of the file.

4. Commit, tag, and push:

   ```bash
   git commit -am "chore(release): vX.Y.Z"
   git tag -a vX.Y.Z -m "vX.Y.Z"
   git push origin main vX.Y.Z
   ```

5. The [release workflow](../.github/workflows/release.yml) then checks that the
   tag matches the package version, builds the sdist and wheel, and creates the
   GitHub release with the matching `CHANGELOG.md` section as notes. To rerun it
   for an existing tag, start it manually with that tag as input.

Publishing to PyPI is not configured yet. When it is added, use
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/) rather than API
tokens.
