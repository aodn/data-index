# GitHub Release artifacts instead of PyPI for package distribution

This is an internal pipeline tool that embeds AWS-specific configuration (ECR registry,
S3 Tables ARNs, Fargate cluster settings) and has no general-purpose public audience.
Publishing to PyPI would expose internal infrastructure details publicly and require
maintaining a PyPI account and trusted-publisher setup for no practical benefit.
Instead, the wheel and sdist are attached to versioned GitHub Releases, installable
via a direct URL (`pip install https://github.com/aodn/data-index/releases/download/vX.Y.Z/...`).
This keeps distribution within the existing GitHub access control boundary.

## Considered Options

- **PyPI** — rejected: public exposure of internal AWS config; requires a separate
  PyPI trusted-publisher OIDC setup; no external consumers expected.
- **AWS CodeArtifact** — rejected: additional AWS resource to manage; GitHub Releases
  achieves the same access control with less operational overhead.
