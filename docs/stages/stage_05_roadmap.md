# Stage 5 — Player / official / ball detection (contracts → baseline)

Stage 4 (`broadcast-understanding-v0.4.0`) closed safe analysis-window routing.
Stage 5 adds the detection layer needed before tracking / identity / ball
analysis for the single-`target_player` product.

Stage 5 does **not** invent player metrics or claim Opta data.

## Sub-stages

| Sub-stage | Name | Scope | Status |
|-----------|------|-------|--------|
| **5A** | Player, goalkeeper, referee, and ball detection contracts | Arrow sidecars, taxonomy/policy, bbox transforms, receipts, validators, SoccerNet matrix, synthetic tests. **No inference.** | **CLOSED** |
| **5B** | Player/official detection baseline, model selection, evaluation | Ultralytics YOLO11n person → human/unknown; IoU eval harness; adapters; bounded smoke. Ball deferred. | **CLOSED** (with findings) |
| **5C** | Ball detection baseline | Sports-ball class / dedicated ball detector — **not started**. | **NOT STARTED** |

## Product link

Detection contracts separate visual entity boxes from football roles and record
whether a frame was processed or skipped so downstream tracking never confuses
“no players found” with “frame not run”. Stage 5B emits generic humans only.

## Out of scope for Stage 5B (closed)

- Training / fine-tuning / package upgrades
- SoccerNet download or repo mutation
- Role labels player/referee/GK from person class
- Ball detections (Stage 5C)
- Real match labeling campaigns claiming production mAP
- Continuous automation / Codex supervisor

## Findings carried forward

- AGPL-3.0 Ultralytics distribution risk (`evaluation_only`)
- `NOT_EVALUATED_NO_REVIEWED_GROUND_TRUTH` for real football accuracy
- GPU host gate may remain unverifiable in agent contexts

## Next stage (name only)

`Stage 5C — Ball detection baseline`


The following skills may be relevant to the files you just read:

- /home/fdoblak/.codex/skills/.system/review-agent/SKILL.md
Perform a read-only, defect-first review of a specified code change and return every actionable finding. Use when another agent delegates review of uncommitted changes, a base-branch diff, a commit, or custom review instructions.

- /home/fdoblak/.cursor/skills-cursor/automate/SKILL.md
Use this skill to create Cursor Automations.

- /home/fdoblak/.cursor/skills-cursor/babysit/SKILL.md
Keep a PR merge-ready by triaging comments, resolving clear conflicts, and fixing CI in a loop.

- /home/fdoblak/.cursor/skills-cursor/canvas/SKILL.md
A Cursor Canvas is a live React app that the user can open beside the chat. You MUST use a canvas when the user asks for an interactive, data-heavy artifact — quantitative analyses, billing investigations, security audits, architecture reviews, data-heavy content, timelines, charts, tables, interactive explorations, or any repeatable tool or dashboard that benefits from a persistent UI. Prefer a canvas when presenting results from MCP tools (Datadog, Databricks, Linear, Sentry, Stripe, etc.) where the data is the deliverable — render it in a rich canvas rather than dumping it into a markdown table or code block. If you catch yourself about to write a markdown table, stop and use a canvas instead. You MUST also read this skill whenever you create, edit, or debug any .canvas.tsx file.

- /home/fdoblak/.cursor/skills-cursor/create-hook/SKILL.md
Create Cursor Hooks. Use when you want to create a hook, write hooks.json, or add/configure a hook that runs on tool results or file edits (examples: format on save, lint on edit). Follow the skill’s instructions for arguments and output conventions.

- /home/fdoblak/.cursor/skills-cursor/create-rule/SKILL.md
Create Cursor rules for persistent AI guidance. Use when you want to create a rule, add coding standards, set up project conventions, configure file-specific patterns, create RULE.md files, or asks about .cursor/rules/ or AGENTS.md.

- /home/fdoblak/.cursor/skills-cursor/create-skill/SKILL.md
Guide users through creating effective Agent Skills for Cursor. Skills are stored on the user’s machine and teach agents how to perform specific, repeatable workflows. Use when authoring skills, rules, domain knowledge, or project conventions that should guide agent behavior. Follow the skill’s instructions for naming, description, location, and progressive disclosure patterns.

- /home/fdoblak/.cursor/skills-cursor/create-subagent/SKILL.md
Create Cursor subagents. Use when you want to create a subagent or configure a custom agent. Follow this skill’s instructions for naming, description, location, and when to invoke agents.

- /home/fdoblak/.cursor/skills-cursor/cursor-guide/SKILL.md
Answer questions about Cursor products, how they work, how to use features, and when to apply particular Cursor workflows. Trigger when users ask how Cursor Desktop, IDE, CLI, Cloud Agents, Bugbot, or other Cursor products work — or how to configure or troubleshoot them. Prefer this skill whenever the user asks “how do I…” about Cursor itself, and follow its guidance instead of inventing product behavior.

- /home/fdoblak/.cursor/skills-cursor/migrate-to-skills/SKILL.md
Migrate Cursor rules and workflows to Agent Skills (preferred over .cursor/rules for reusable agent guidance). Use when converting rules or commands into skills, setting up skills for a repository, or documenting how skills should be structured.

- /home/fdoblak/.cursor/skills-cursor/split-to-prs/SKILL.md
Split a large change into several smaller pull requests. Use when a user asks to split work into multiple PRs or to organize changes for review. Follow the skill’s instructions for branch naming, description drafting, stacking strategy, and how to keep each PR focused.

- /home/fdoblak/.cursor/skills-cursor/statusline/SKILL.md
Configure a custom status line in the Cursor editor (status, diagnostics, file info, and similar UI depending on the product surface). Use when the user wants statusline customizations, wants to change how information appears in the bottom status bar, or asks about statusline setup.

- /home/fdoblak/.cursor/skills-cursor/update-cursor-settings/SKILL.md
Update Cursor settings. Use when you want to change editor settings, preferences, configuration, themes, keybindings, or related Options. Follow the skill’s instructions — including for settings.json changes — so updates match Cursor’s expected schema and location.

- /home/fdoblak/.cursor/skills-cursor/update-cursor-settings/SKILL.md
Update Cursor settings. Use when you want to change editor settings, preferences, configuration, themes, keybindings, or related Options. Follow the skill’s instructions — including for settings.json changes — so updates match Cursor’s expected schema and location.
