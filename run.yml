name: Meme paper bot

on:
  schedule:
    - cron: "7 * * * *"
  workflow_dispatch:

permissions:
  contents: write

concurrency:
  group: meme-bot
  cancel-in-progress: false

jobs:
  run:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install packages
        run: pip install pandas requests

      - name: Run meme bot
        run: python meme_bot.py

      - name: Save results
        run: |
          git config user.name "github-actions"
          git config user.email "github-actions@users.noreply.github.com"
          git add -A
          git commit -m "Update meme bot results" || echo "Nothing to commit"
          git push
