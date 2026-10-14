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

## Architect it yourself; the spec carries the design

**Never put an architect, designer or "decide the approach" task in the spec.**
Do that work now, before writing any YAML, and write the decisions into the
task prompts as fixed interfaces: module paths, function names and signatures,
file formats, on-disk layouts, which file each task owns.

This is not a style preference. A design task is the worst possible fit for this
tool:

- Its output reaches dependents as **summary text, truncated at
  `max_context_chars`**. Three tasks coding against a clipped list of function
  signatures will each invent a different missing half, and the mismatch only
  surfaces when their files are imported together.
- Every implementation task has to wait for it, so the graph collapses to a line
  and the parallelism you came here for disappears.
- It has none of your conversation with the user. You know what they actually
  asked for; a sub-session gets a paragraph.
- Design is the part a user most wants a say in. Routing it through a session
  they are not watching hides exactly the decision they would have wanted to
  make.

So resolve the design questions **with the user, at plan time**, then hand the
tasks a contract:

```yaml
# Not this:
- id: design
  prompt: Design the session storage format and summarise the API.

# This:
- id: store
  prompt: |
    Implement `orchestrator/session_store.py` with exactly this API:
      begin(project_root: Path, run_id: str, spec_text: str) -> Path
      record_task(project_root: Path, run_id: str, record: TaskRecord) -> None
      load_session(project_root: Path, run_id: str) -> Session
    Records live in <project_root>/.orchestrator/sessions/<run_id>/, one JSON
    file per task so an interrupted run still reads back.
```

The second form lets the tasks that consume the API start without waiting, and
it lets you notice a hole in your own design while it is still free to change.

If writing those signatures shows you do not yet understand the problem well
enough to specify it, that is the signal to go and read the code yourself - not
to delegate the confusion to a task.

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
  Implement the endpoints in app.py against the schema in schema.sql.
  Finish by listing each endpoint, its method, path and response shape, so the
  test task can work from your summary without re-reading the file.
```

For large artifacts, hand off a **path**, not the content - tasks sharing a cwd
can read each other's files, and that sidesteps the character cap entirely:

```yaml
prompt: |
  Read schema.sql (written by the migration task) and implement the endpoints.
```

Note what is being handed off in both cases: **what was built**, not **what was
decided**. Decisions belong in the spec you wrote; only results flow between
tasks.

## Let tasks ask

The reason this tool exists instead of `claude -p` is that tasks can ask the
human. That only happens if the persona invites it - a system prompt that says
nothing will make a task guess silently and commit the guess to disk.

```yaml
agents:
  backend-dev:
    system_prompt: |
      You are a backend engineer.
      If you hit an ambiguity the spec does not settle, and guessing would be
      expensive to unwind, use the AskUserQuestion tool rather than picking for
      the user.
```

Note the division of labour. **Design ambiguity is yours to resolve with the
user before the run** - that is the previous section. What is left for a task to
ask about is what only becomes visible once it is in the code: an existing
convention that contradicts the spec, a dependency that is not installed, two
callers that disagree. If you find yourself writing "ask the user how this
should work", you have left architecture in the spec; go and decide it.

Do not put this on every agent, or a wide run will block on a queue of trivia.
Questions from parallel tasks are serialized and labelled with the task id, so
they never interleave, but the human still answers them one at a time.

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
  backend-dev:
    description: Implements against a given interface.
    system_prompt: You are a backend engineer. Match the conventions already here.
    tools: [Read, Write, Edit, Bash, Glob, Grep]   # omit for every tool
  writer:
    description: Writes documentation.
    system_prompt: You are a technical writer. Concise, no marketing language.
    tools: [Read, Write, Glob, Grep]

# No design task: the schema below was decided while writing this spec, which
# is why `migrate` and `docs` can start at the same time.
tasks:
  - id: migrate
    agent: backend-dev
    prompt: |
      Write schema.sql creating exactly these tables:
        links(id TEXT PRIMARY KEY, target TEXT NOT NULL,
              owner_id TEXT NOT NULL, created_at TIMESTAMP NOT NULL)
        clicks(id INTEGER PRIMARY KEY, link_id TEXT REFERENCES links(id),
               at TIMESTAMP NOT NULL, referrer TEXT)
      Finish by confirming what you created.

  - id: api
    agent: backend-dev
    depends_on: [migrate]
    prompt: |
      Implement app.py against schema.sql with POST /links, GET /{id} and
      GET /links/{id}/stats. Finish by listing each endpoint and its response
      shape.

  - id: docs
    agent: writer
    difficulty: easy
    prompt: |
      Document this data model for a new contributor:
        links(id, target, owner_id, created_at)
        clicks(id, link_id, at, referrer)
```

`migrate` and `docs` start together in wave 1, and `api` follows `migrate`.
`docs` has no dependency because the schema was decided here, in the spec -
with a design task at the top, all three would have queued behind it.

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

For a wide run, where several tasks in one wave stream output at once, suggest
`--live` to the user (`uv run orchestrator spec.yaml --live -j 4`). It shows one
task at a time with a tab per task, which is far easier to follow than
interleaved output. It needs a real terminal and falls back to normal output
when piped, so do not suggest it for CI.

## Past runs are recorded

Every run writes its spec and per-task results to a session directory, and
`orchestrator.session_store` can list and load them. This is useful when
planning a follow-up: before writing a new spec, look at what a similar run
actually cost per task, which tasks failed and why, and what each task produced in full. It is also the
place to check a `hard` task's output after the run, without re-running it. Records are not redacted, so do not paste
their contents into anything shared without checking first.

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

And the one that invalidates the rest if you get it wrong:

0. Is there a design, architecture or "decide the approach" task in this spec?
   If so, delete it, make those decisions yourself, and write them into the
   prompts as fixed interfaces.
