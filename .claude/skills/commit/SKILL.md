# Commit Skill
1. Run `git branch --show-current` — if on `main`, STOP and tell the user to run /start-work first.
2. Run `git status` to check for staged/unstaged changes.
3. Generate a conventional commit message based on the diff.
4. Stage relevant files with `git add <specific files>` — NEVER use `git add -A`.
5. Run `git diff --cached --name-only` and verify NO files match: .env*, *.pem, *credentials*, *secret*, .DS_Store. If any match, unstage them and warn the user.
6. Run `git commit -m "<message>"`.
7. Ask if the user wants to push and open a PR.
