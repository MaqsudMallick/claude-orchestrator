# claude-orchestrator

Run a DAG of Claude tasks in parallel, with tool approvals handled
automatically — **without** losing Claude's ability to ask you clarifying
questions.

That combination is the whole point. `claude -p` is headless, so
`AskUserQuestion` is unavailable. `permission_mode="bypassPermissions"` skips
the permission callback entirely, so Claude never gets to ask. This runs each
task as its own `ClaudeSDKClient` session in the **default** permission mode and
routes every tool call through one `can_use_tool` callback that auto-approves
ordinary work, denies dangerous work, and hands `AskUserQuestion` to you.

## Install

```bash
uv sync
```

Requires the `claude` CLI on PATH and logged in (the SDK drives it and reuses
your Claude Code credentials).

## Use

```bash
uv run orchestrator examples/example.yaml --dry-run   # validate + show the plan
uv run orchestrator examples/example.yaml -j 4        # run it
uv run orchestrator spec.yaml --json-report run.json  # machine-readable report
```

Exit code is `0` only if every task succeeded; `1` if anything failed or was
skipped; `2` if the spec itself is invalid.

## Spec format

```yaml
version: 1

project_dir: ./repo       # the agents' place of work (see below)

defaults:
  cwd: .                  # task cwds resolve under project_dir
  difficulty: medium
  max_turns: 40
  max_context_chars: 4000 # cap on upstream text injected downstream
  allowed_write_dirs: []  # extra writable roots beyond each task's cwd
  allow_push: false       # `git push` is denied unless this is true

agents:
  db-architect:
    description: Designs data models.
    system_prompt: |
      You are a database architect...
      If requirements are ambiguous, use the AskUserQuestion tool.
    tools: [Read, Write, Edit, Bash]   # omit for "all tools"

tasks:
  - id: schema
    agent: db-architect
    difficulty: hard
    prompt: Design the schema for ...
  - id: api
    agent: backend-dev
    difficulty: medium
    depends_on: [schema]
    isolate: worktree     # optional: run in its own git worktree
    prompt: Implement the API ...
```

Per-task fields: `id`, `agent`, `prompt`, `difficulty`, `depends_on`, `cwd`,
`isolate`, `effort`, `max_turns`, `max_budget_usd`. Unknown fields are a hard
error, so typos surface at load time rather than mid-run.

Validation happens before anything is spent: unknown agents, dangling
dependencies, duplicate ids, reserved ids (`START`/`END`) and dependency cycles
all fail with a readable message.

## Where the agents work

`project_dir` is the directory the agents work in. A relative path resolves
against the spec file, so a spec can sit next to the project it drives; an
absolute path is used as-is.

```yaml
project_dir: ~/projects/my-app
defaults:
  cwd: .                  # the project root itself
tasks:
  - id: api
    cwd: services/api     # a subdirectory of the project
```

Every task `cwd` resolves under it, as does `allowed_write_dirs`. An absolute
task `cwd` opts out deliberately. It defaults to `.` — the spec file's own
directory — so existing specs behave as before.

Override it without editing the spec, which is what makes a spec reusable
across checkouts:

```bash
uv run orchestrator spec.yaml --project-dir ~/work/client-app
```

A missing `project_dir` is an error at **load** time, not run time: every task
would fail the same way, so a typo in the path should not cost a single token.

Note the write sandbox is still each task's own `cwd`, not the whole project —
a task with `cwd: services/api` cannot write to `services/web`. Widen it with
`defaults.allowed_write_dirs` if that is deliberate.

## Choosing a model

Tasks never name a model id. They declare how hard they are, and
`orchestrator/models.py` maps that to a model — one place to retune cost:

| difficulty | model |
|---|---|
| `easy`   | `claude-haiku-5-5` |
| `medium` | `claude-opus-5-5` |
| `hard`   | `claude-fable-5-1` |

The run summary reports cost per task and per model so you can check the
assignments are actually paying for themselves.

## Ordering and parallelism

Tasks form a DAG; LangGraph executes it. A task runs once all of its
`depends_on` have finished, and anything without a dependency edge between it
and another task runs concurrently. `-j/--max-parallel` caps how many sessions
run at once (default 4).

Each dependent receives its upstream tasks' result text prepended to its prompt,
clipped to `max_context_chars`.

**On failure, dependents are skipped and unrelated branches run to completion.**
The summary lists what failed, what was skipped, and why.

## Clarifying questions

When Claude calls `AskUserQuestion`, the run prints the question — labelled with
the task that asked — and waits for your answer:

```
========================================================================
  QUESTION from task 'choose' (agent: asker)
========================================================================

(1/1) [Format] Which format should the config file use?
   1. JSON (config.json)
   2. YAML (config.yaml)
   (enter a number, or type free text)
   >
```

Enter a number to pick an option, comma-separated numbers for multi-select, or
type anything else to answer in free text. Prompts from parallel tasks are
serialized behind a lock so they never interleave. If stdin is closed (CI), the
first option is chosen and a note goes to stderr rather than hanging the run.

## Permissions

The callback allows ordinary tool calls and denies, by default:

- recursive force deletes, `git reset --hard`, destructive `git clean`
- `git push` (pushes can trigger deploys — set `allow_push: true` to permit it)
  and `--force` pushes regardless of that setting
- `sudo`, `mkfs`/`fdisk`/`shutdown`/`reboot`, raw device writes, pipe-to-shell,
  fork bombs
- `Write`/`Edit`/`NotebookEdit` outside the task's cwd (plus
  `allowed_write_dirs`), and shell redirects to absolute paths outside it

Denials are counted in the summary and recorded per task in the JSON report.

## Automatic review

With `review.enabled`, each task that succeeds is reviewed by a **separate
session** running Claude Code's bundled `code-review` skill. The reviewer uses
the **same model as the task**, so review depth tracks the difficulty tier you
paid for, and it runs read-only (`Write`/`Edit` are disallowed) because
reviewing and fixing are different jobs.

If the reviewer returns `VERDICT: CHANGES_REQUESTED`, the fix round **resumes
the implementer's own session** rather than starting a fresh one, so it still
holds the context for why it built things the way it did. Then it is reviewed
again, up to `max_reviews` rounds.

```yaml
defaults:
  review:
    enabled: true
    max_reviews: 2          # rounds; each round is a review + possibly a fix

tasks:
  - id: docs
    review: {enabled: false}   # per-task override; unset fields inherit
```

A per-task `review:` block overrides only the fields it names — setting
`max_reviews` alone will not silently turn review off.

Only an explicit `CHANGES_REQUESTED` spends a fix round; an unparseable verdict
or a crashed reviewer is recorded and the task is left alone, because a broken
reviewer must not be able to fail working code.

**Review never fails a task.** When `max_reviews` rounds run out, whatever the
task produced is accepted and the run moves on — dependents are not skipped. The
summary still reports the verdict as `CHANGES(n)` and lists the task under
"review rounds ran out; work accepted as-is with findings open", so outstanding
findings stay visible without blocking the graph.

**Review is off by default and is not cheap.** In a measured run it was 69-89%
of total spend, because a reviewer reads what the implementer wrote and the fix
round re-runs the implementer. The run summary breaks out review cost so you can
see the trade rather than guess at it.

## Isolation

By default tasks share a working directory, which is fine when they touch
different files. A task that needs its own copy sets `isolate: worktree` and
gets a git worktree on branch `orch/<run-id>/<task-id>`, under
`<repo>/../.orchestrator-worktrees/<run-id>/`. Worktrees are **left in place**
when the run ends — removing them would throw away the work. Merge or delete
them yourself, or pass `--cleanup-worktrees`.

## Notes on the SDK (verified against claude-agent-sdk 0.2.166)

- `AskUserQuestion` does reach `can_use_tool`, and answers are accepted as
  `PermissionResultAllow(updated_input={"questions": ..., "answers": ...})`.
  No dummy `PreToolUse` hook is needed to keep the stream open.
- Putting `AskUserQuestion` in `allowed_tools` did **not** auto-approve it — the
  callback still fired. The tool is nonetheless left out of allowlists here.
- Restricting an agent's `tools:` hides `AskUserQuestion` unless it is in the
  list, so it is re-added automatically.
- `ResultMessage.is_error` is `True` for non-crash cutoffs such as hitting
  `max_turns`, so it is not a clean failure signal on its own. The reason and
  `stop_reason` are carried into the report.
- `permission_mode="dontAsk"` means "deny anything not pre-approved", which is
  not the same as bypass and not what this needs.
- Subagents (the Task tool) cannot ask clarifying questions, which is why each
  task is a full session rather than a subagent.

## Tests

```bash
uv run pytest
```

The DAG tests stub out the Claude session, so the suite runs offline and free.

## Planning a spec with an agent

`.claude/skills/orchestrate-plan/SKILL.md` is a skill for Claude itself. Any
agent working in this repo picks it up automatically and learns how to turn a
request into a spec: where real dependency edges are, how to pick a difficulty
per task by consequence rather than size, how to write prompts that hand off
cleanly (dependents see only the upstream's final summary, capped at
`max_context_chars`), when `isolate: worktree` is worth its merge cost, and
which failure modes to expect. It also pushes `--dry-run` as a plan review.

## Codebase health

Four checks, all enforced by a `pre-commit` hook (installed in `.git/hooks/`):

```bash
uv run ruff check .          # lint
uv run ruff format --check . # formatting
uv run mypy                  # strict type check
uv run pytest                # tests
```

The hook runs all four and reports every failure before blocking, rather than
stopping at the first. `git commit --no-verify` bypasses it.

mypy runs in `strict` mode. Two notes on what that required:

- `Spec.difficulty_for()` / `effort_for()` returned bare `str`, so the `Literal`
  contract with `ClaudeAgentOptions` was unchecked — a typo in a difficulty
  alias would only have surfaced at runtime. They now return the `Literal`
  types.
- `NodeFn` in `graph.py` is a `Protocol`, not a `Callable[...]` alias. mypy
  cannot bind a Callable alias to LangGraph's own node protocol, so `add_node()`
  fails to resolve its overload; a Protocol matches structurally.
