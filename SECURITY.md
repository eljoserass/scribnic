# Security policy

## Supported versions

Scribnic is pre-1.0. Security fixes go to the latest release and `main` only.

## Reporting a vulnerability

Report vulnerabilities privately through
[GitHub security advisories](https://github.com/eljoserass/scribnic/security/advisories/new).
Do not open a public issue. Include the affected version or commit, the workflow
or component involved, and steps to reproduce.

You should receive an acknowledgement within seven days. Once a fix is available,
the advisory and the release notes will describe the issue.

## Health data

Never include patient audio, transcripts, notes, or any other health information
in issues, pull requests, discussions, or advisories. Reproduce problems with
synthetic or public recordings.

## Scope

Relevant reports include:

- Leaking audio, transcripts, or notes outside the configured run directory or
  inference endpoint.
- Downloading or running assets that do not match their pinned revision or
  checksum.
- Command injection or unsafe process handling in the runtime managers.

Scribnic runs local inference processes and writes run artifacts to disk. Securing
the host, its storage, and any external endpoint you configure is the deployer's
responsibility.
