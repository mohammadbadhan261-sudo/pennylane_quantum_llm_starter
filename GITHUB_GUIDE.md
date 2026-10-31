# GitHub guide (VS Code on Windows)

1. Open this folder in VS Code and run `git --version` in Terminal.
2. Create an empty GitHub repository in your own account at https://github.com/new. Do NOT select initial README/gitignore.
3. If not already initialized: `git init`.
4. Set your Git identity if needed:
   `git config --global user.name "Your Name"`
   `git config --global user.email "your-email@example.com"`
5. Check `git status` and verify that .venv, downloaded Common Crawl files, tokens, and model checkpoints are not staged.
6. Stage the code and docs by name, for example:
   `git add .gitignore README.md GITHUB_GUIDE.md requirements.txt requirements-commoncrawl.txt 0*.py data/sentences.csv`
7. Review with `git diff --cached --stat` and `git diff --cached`.
8. Commit: `git commit -m "Add PennyLane sentiment demo and Common Crawl import pipeline"`.
9. Rename initial branch: `git branch -M main`.
10. Copy YOUR actual empty repository URL from GitHub, then run:
    `git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO.git`
    `git push -u origin main`
11. Future updates:
    `git status`
    `git add 07_fetch_commoncrawl.py README.md`
    `git commit -m "Improve Common Crawl filtering"`
    `git push`

If `origin` already exists, run `git remote -v`; use `git remote set-url origin YOUR-URL` only if it needs correcting.
If your repository already exists and is initialized, do not repeat git init or recreate a different repository.

Never commit: `.venv/`, API credentials, raw Common Crawl text, private source URLs, unreviewed labels, or trained weight files.
`git rm --cached file` unstages tracking but does not remove secrets already written into Git history.
