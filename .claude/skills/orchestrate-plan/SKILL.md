---
name: orchestrate-plan
description: >-
  Plan a multi-part implementation as a claude-orchestrator spec.yaml - a DAG of
  Claude tasks that run in parallel, each with its own agent persona, model tier
  and working directory. Use this whenever a request splits into several pieces
  that could run concurrently ("build the schema, then the API and the docs",
  "refactor these six modules", "port this to three languages"), whenever
  someone mentions the orchestrator, spec.yaml, task DAG, agent fan-out or
  parallel Claude runs, and whenever you are about to hand-roll your own
  multi-step plan for work that is already shaped like a dependency graph. Also
  use it when asked to estimate or reduce what a multi-task run will cost, since
  model tier is chosen per task here.
---

# Planning an implementation as an orchestrator spec

`claude-orchestrator` runs a DAG of Claude sessions. Each task is a full
session - its own persona, model, tools and cwd - so each one can still stop and
ask the human a clarifying question. Your job when this skill triggers is to
turn a request into a `spec.yaml` that is cheap, parallel, and honest about what
it does not know.

## First decide whether it is worth it

The orchestrator earns its keep when work is **wide** (several pieces that do
not block each other) or **tiered** (some pieces are much harder than others, so
paying Opus for all of them is waste). It costs you a spec file, a process per
task, and the loss of a single shared conversation.

Just do the work yourself when:

- it is two or three sequential steps - a DAG with no width is a slower, more
  expensive loop
- every task would need the full context of every other task; tasks only see
  their upstreams' final summary text, not each other's reasoning
- the whole job is one file - parallel tasks in one cwd will fight over it

A good signal: sketch the dependency edges first. If the graph is a straight
line, write the code instead.

## Build the graph from real blockers

Add an edge only where a task genuinely cannot start without its upstream's
output. The common failure is encoding your own sense of sequence - "docs should
come after tests" - as a dependency. That serialises work the machine could have
run at once, and every spurious edge also widens the blast radius of a failure,
because **a failed task skips everything downstream of it**.

Put the foundational, most-likely-to-fail task early and alone. If it dies, you
want to have spent nothing else.

Check your width with `--dry-run`: it prints the execution waves. Wave sizes of
1, 1, 1 mean you have written a shell script with extra steps.

## Point the run at a project

`project_dir` is where the agents work. Set it to the repository or directory
the task is about - relative to the spec file, or absolute. Every task `cwd`
resolves under it, so tasks name subdirectories rather than repeating a long
path:

```yaml
project_dir: ~/projects/my-app
tasks:
  - {id: api,  cwd: services/api,  ...}
  - {id: web,  cwd: services/web,  ...}
```

It must already exist - a missing one fails at load time, before any spend.
Leave it unset and it means the spec file's own directory, which is right for a
scratch workspace sitting next to the spec.

Two things to keep in mind when planning. A task's **write sandbox is its own
`cwd`**, not the whole project, so two tasks in sibling subdirectories cannot
scribble on each other - but a task that genuinely needs to touch the whole
project should have `cwd: .` rather than a narrow one plus deny-rule
surprises. And because `--project-dir` overrides the spec, prefer writing specs
that are reusable across checkouts over hardcoding one person's path.

## Choose a difficulty per task

Tasks never name a model. They declare difficulty, and the mapping lives in one
file:

| difficulty | model | use for |
|---|---|---|
| `easy` | `claude-haiku-5-5` | mechanical work with a clear spec: docs from an existing schema, boilerplate, renames, test scaffolding |
| `medium` | `claude-opus-5-5` | ordinary implementation against a known contract |
| `hard` | `claude-fable-5-1` | design and irreversible decisions: schemas, public API shape, concurrency, migrations |

Spend by consequence, not by size. A hundred-line schema that every downstream
task builds on is `hard`; a thousand lines of docs derived from it is `easy`.
The run summary reports cost per model, so you can check the split afterwards
rather than guessing.

Set `defaults.difficulty` to the common case and override per task - the
defaults apply to any task that omits the field.

## Write prompts that hand off well

A dependent task receives **only its upstreams' final result text**, clipped to
`defaults.max_context_chars` (4000 by default). It does not see the upstream's
files-read, reasoning, or tool calls. Two consequences worth designing around:

End every prompt whose output something else depends on with an instruction to
summarise what it produced. Otherwise the upstream ends with "Done!" and the
downstream task inherits nothing.

```yaml
prompt: |
  Design the database schema for a link shortener and write it to schema.sql.
  Finish by summarising each table and its columns, so the API task can work
  from your summary without re-reading the file.
```

For large artifacts, hand off a **path**, not the content - tasks sharing a cwd
can read each other's files, and that sidesteps the character cap entirely:

```yaml
prompt: |
  Read schema.sql (written by the schema task) and implement the endpoints.
```

## Let tasks ask

The reason this tool exists instead of `claude -p` is that tasks can ask the
human. That only happens if the persona invites it - a system prompt that says
nothing will make a task guess silently and commit the guess to disk.

```yaml
agents:
  db-architect:
    system_prompt: |
      You are a database architect.
      If a requirement is ambiguous in a way that would change the schema, use
      the AskUserQuestion tool before committing to a design. A wrong guess here
      propagates into every downstream task.
```

Put this on the tasks where a wrong assumption is expensive - usually the `hard`
ones at the top of the graph. Do not put it on every task, or a wide run will
block on a queue of trivia. Questions from parallel tasks are serialized and
labelled with the task id, so they never interleave, but the human still answers
them one at a time.

If an agent restricts `tools:`, `AskUserQuestion` is re-added automatically, so
a tools allowlist cannot silently disable asking.

## Decide where review earns its cost

`review.enabled` adds a second session after each successful task: a reviewer on
the **same model** running the `code-review` skill, and if it asks for changes, a
fix round that resumes the implementer's own session, repeated up to
`max_reviews`.

Budget for this honestly - in measured runs review was **69-89% of total spend**,
because the reviewer reads everything the implementer wrote and the fix round
re-runs the implementer. Roughly, review doubles a task and a fix round doubles
it again.

So spend it where a defect is expensive or hard to see later:

```yaml
defaults:
  review: {enabled: true, max_reviews: 2}

tasks:
  - id: schema        # foundational: everything downstream inherits its mistakes
    difficulty: hard
  - id: docs
    difficulty: easy
    review: {enabled: false}   # prose; a reviewer adds little and costs double
```

A per-task `review:` block overrides only the fields it names, so
`review: {max_reviews: 3}` keeps `enabled` from the defaults.

`max_reviews: 1` means "review once, fix once if needed, then stop" - the
findings still get reported. Raise it only for tasks where you would genuinely
want a second opinion on the fix; most work converges in one round.

Two behaviours worth planning around. Review runs **only on tasks that
succeeded**, so it is not a safety net for a task that failed outright. And
review never fails a task: when the rounds run out, the work is accepted as-is
and dependents run regardless, with the open findings reported. Treat review as
a quality improver, not a gate - if correctness genuinely must block the graph,
express that as a dependent task that verifies (tests, a build) and fails.

Because the reviewer reads a git diff when one is available, a task running in a
git repo (or under `isolate: worktree`) gets a sharper review than one in a bare
directory.

## Decide isolation only where tasks collide

Tasks share a cwd by default, which is what you want when they touch different
files - it is how a downstream task reads what its upstream wrote.

Set `isolate: worktree` only when two tasks that run **in the same wave** would
write the same path. Each gets its own git worktree on branch
`orch/<run-id>/<task-id>`, left in place afterwards, which means you then owe
someone a merge. That cost is real, so prefer splitting the work by file over
isolating it.

## Shape of a spec

```yaml
version: 1

project_dir: ./repo       # the agents' place of work

defaults:
  cwd: .                  # task cwds resolve under project_dir
  difficulty: medium
  max_turns: 40           # too low shows up as a failure, not a partial result
  allow_push: false

agents:
  db-architect:
    description: Designs data models.
    system_prompt: |
      You are a database architect. Ask before guessing on anything that would
      change the schema.
    tools: [Read, Write, Edit, Bash, Glob, Grep]   # omit for every tool
  backend-dev:
    description: Implements endpoints.
    system_prompt: You are a backend engineer. Match the conventions already here.
    tools: [Read, Write, Edit, Bash, Glob, Grep]
  writer:
    description: Writes documentation.
    system_prompt: You are a technical writer. Concise, no marketing language.
    tools: [Read, Write, Glob, Grep]

tasks:
  - id: schema
    agent: db-architect
    difficulty: hard
    prompt: |
      Design the schema and write schema.sql. Finish by summarising each table.

  - id: api
    agent: backend-dev
    difficulty: medium
    depends_on: [schema]
    prompt: Implement the endpoints described above.

  - id: docs
    agent: writer
    difficulty: easy
    depends_on: [schema]
    prompt: Document the data model for a new contributor.
```

Here `api` and `docs` run concurrently after `schema`.

## Before you run it

```bash
uv run orchestrator spec.yaml --dry-run   # validates, prints waves, spends nothing
uv run orchestrator spec.yaml -j 4
```

`--dry-run` catches unknown agents, dangling `depends_on` ids, duplicate ids,
dependency cycles and misspelled fields at load time. Always run it first; a
cycle caught here is free, and the same mistake caught at runtime is not.

Read the wave output as a plan review, not just a validation pass: it is the
cheapest moment to notice you have serialised something.

## Things that will bite you

**`max_turns` too low reads as a failure.** A task that runs out of turns is
reported `failed`, with the reason given, and its dependents are skipped - the
partial work is still on disk but the graph treats it as dead. Budget turns
generously for tasks that explore before writing.

**Exit code 1 covers both failed and skipped.** One dead foundational task will
fail a run that mostly succeeded; read the summary, not just the exit status.

**`git push` is denied by default**, as are recursive deletes, `sudo`,
`git reset --hard`, and writes outside the task's cwd. If a task legitimately
needs to push, set `defaults.allow_push: true` deliberately rather than
wondering why the task reported a blocked tool call.

**Subagents cannot ask questions.** Do not plan a task around spawning Task-tool
subagents to do the asking; that is precisely why each task here is its own
session.

**Cost scales with the widest wave, not the task count.** Ten `easy` tasks are
cheaper than one `hard` one. Check the per-model breakdown in the run summary
and move tiers down where the output did not need the thinking.

## A quick checklist

1. Is this actually wide or tiered? If not, stop and write the code.
2. Are all edges real blockers?
3. Is the riskiest task early and alone?
4. Does every task with dependents end by summarising its output?
5. Is each difficulty set by consequence rather than size?
6. Do the tasks that could guess wrong have permission to ask?
7. Does any same-wave pair write the same file?
8. Is review on for the tasks whose mistakes are expensive, and off for the
   ones where it would just double the bill?
9. Does `--dry-run` show the width you expected?
