# Git / GitHub workflow

## Stage discipline

1. Every sub-stage starts from a **clean** checkpoint on `main`.
2. Stage only **explicit paths** — never `git add .` or `git add -A`.
3. Run tests + secret scan before commit.
4. Close each successful sub-stage with a descriptive commit.
5. After a successful commit, push `main` to the **private** GitHub repository with a normal push.
6. **Force push** and history rewrite (`amend`/`reset`/`rebase` of published history) are forbidden unless an explicit future policy says otherwise.
7. Datasets, videos, model binaries, secrets, runtime reports, caches, `dist/`, and egg-info **must not** reach GitHub.
8. Large binaries require a separate Git LFS decision — do not auto-install LFS.
9. Authentication uses GitHub CLI, SSH, or a secure credential helper only.
10. Passwords/tokens are never written into repository files.
11. Remote URLs must not embed credentials.
12. If remote ownership is unclear, **do not push**.
13. Tags only when a stage brief explicitly allows them.

## Remote safety checklist

```bash
git remote -v
# URL must be ssh:// or https:// without user:token@
gh repo view --json name,isPrivate,owner   # when gh available
```

## Push

```bash
git push -u origin main
```

Never:

```bash
git push --force
git push --force-with-lease
git push --mirror
```
