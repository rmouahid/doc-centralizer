# Contributing to doc-centralizer

Thanks for your interest in improving doc-centralizer! This guide explains how to
set up the project, the conventions used for branches and commits, and what to
check before opening a pull request.

By participating, you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).

## Reporting bugs and suggesting features

Open an [issue](https://github.com/rmouahid/doc-centralizer/issues) first, for
bugs as well as feature ideas, so the approach can be discussed before you spend
time on code. For a bug, include the steps to reproduce it, what you expected,
what happened instead (with the error message or traceback), and your OS and
Python version.

## Development setup

Requirements: Python 3.12+, [Poetry](https://python-poetry.org/) and, for image
files, [Tesseract OCR](https://github.com/tesseract-ocr/tesseract).

```bash
git clone https://github.com/<your-username>/doc-centralizer.git
cd doc-centralizer
poetry install
```

The Phi-3 weights (`bash scripts/download_model.sh`, ~2.4 GB) are only needed to
run the application, not the tests.

## Branches

`main` is the integration branch: it must always pass the tests. Never commit to
it directly; create a branch from an up-to-date `main` instead, named after the
kind of change and the issue it addresses:

| Prefix | Use | Example |
|---|---|---|
| `feature/` | New functionality | `feature/12-pdf-table-extraction` |
| `fix/` | Bug fix | `fix/18-empty-index-crash` |
| `docs/` | Documentation only | `docs/3-contributing-guide` |
| `refactor/` | Code change without behaviour change | `refactor/21-split-chunker` |
| `test/` | Tests only | `test/9-history-edge-cases` |
| `chore/` | Build, CI, dependencies, tooling | `chore/7-update-faiss` |

```bash
git switch main
git pull
git switch -c feature/12-pdf-table-extraction
```

Before opening the pull request, rebase your branch on the latest `main` rather
than merging `main` into it, to keep the history linear:

```bash
git fetch origin
git rebase origin/main
```

## Commit messages

Commits follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<optional scope>): <description>

<optional body: why the change, and the notable choices>

<optional footer: Closes #12>
```

- **type**: `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `build`, `ci`,
  `style`, `chore` or `revert`.
- **description**: imperative mood, lowercase, no final period, 72 characters at
  most, e.g. `feat(chunker): extract text from excalidraw files`.
- **body**: explain *why* rather than *what*, wrapped at 72 characters.
- A breaking change adds `!` after the type (`feat!: ...`) and a
  `BREAKING CHANGE:` footer.
- Reference the issue in the footer: `Closes #12`, or `Refs #12` if it stays open.

Keep each commit to one coherent change; squash "wip" or "fix typo" commits
before asking for a review.

## Running the tests locally

```bash
poetry run pytest
```

The suite covers chunking, embeddings/FAISS search and conversation history,
and does not need the model file. GitHub Actions runs the same command on every
push and pull request (`.github/workflows/tests.yml`), so make sure it passes
locally first. Add or update tests for every bug fix and new feature.

To check the application itself:

```bash
poetry run streamlit run streamlit_app.py
```

## Pull requests

1. Push your branch and open a pull request against `main`.
2. Give it a Conventional Commits title and describe the context, the solution,
   how you tested it, and anything left out of scope. Link the issue with
   `Closes #<number>`.
3. Make sure the CI is green and the README is updated if behaviour or setup
   changed.
4. Address review comments with new commits, then tidy the history before the
   merge. Pull requests are merged with a rebase, so every commit lands on
   `main` as is.

## License

By contributing, you agree that your contributions will be licensed under the
[MIT License](LICENSE) of this project.
