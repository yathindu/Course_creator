"""
Backs up approved lesson media to the project's GitHub repo, so Streamlit Cloud's
ephemeral container disk doesn't silently lose approved work on a crash/restart --
confirmed as a real risk (Streamlit Cloud's own docs and this app's own
`_show_media()` docstring both note the local disk isn't durable).

Uses GitHub's Contents API directly (one file per commit) rather than shelling out
to git -- no local git checkout/credentials needed inside the running container,
just an HTTPS call with a token. Deliberately opt-in via the presence of
GITHUB_TOKEN: if it's unset (e.g. local dev without it configured), every function
here is a silent no-op and the app behaves exactly as it did before this module
existed -- callers don't need an `if enabled:` check at every call site.

Pushes land on the same branch (`main`) the Streamlit Cloud app deploys from, which
means an approval can trigger a redeploy -- a deliberate tradeoff: it's what makes a
fresh container automatically start with every previously-approved asset already on
disk, which is the whole point. A few seconds of redeploy per approval is judged
worth guaranteed durability for this use case (a small number of manual approvals,
not a high-frequency write path).

Setup:
    Add GITHUB_TOKEN to .env (local) or Streamlit Cloud's Secrets (deployed) -- a
    GitHub personal access token with "repo" scope (classic) or "Contents:
    read and write" (fine-grained) on the target repo. Optionally set
    GITHUB_REPO ("owner/name") and GITHUB_BRANCH if not the defaults below.
"""

import base64
import os

import requests
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "yathindu/Course_creator")
GITHUB_BRANCH = os.environ.get("GITHUB_BRANCH", "main")
_API_ROOT = "https://api.github.com"


def enabled():
    return bool(GITHUB_TOKEN)


def _headers():
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _existing_sha(repo_path):
    """SHA of the file currently on the branch, or None if it doesn't exist yet --
    the Contents API requires the current SHA to update a file (as a lost-update
    guard), but rejects a SHA on a brand-new file, so callers need to know which
    case they're in."""
    url = f"{_API_ROOT}/repos/{GITHUB_REPO}/contents/{repo_path}"
    response = requests.get(url, headers=_headers(), params={"ref": GITHUB_BRANCH})
    if response.status_code == 200:
        return response.json()["sha"]
    return None


def push_file(local_path, repo_path, message):
    """Upload local_path's current bytes to repo_path on GITHUB_BRANCH, creating or
    updating it as needed. No-op (returns None) if GITHUB_TOKEN isn't set. Returns
    a short human-readable status string on success; raises on real failure so the
    caller can decide how to surface it (this is a backup path, not the primary
    save -- a failure here should never be treated as the approval itself failing)."""
    if not enabled():
        return None

    content_b64 = base64.b64encode(open(local_path, "rb").read()).decode("ascii")
    payload = {"message": message, "content": content_b64, "branch": GITHUB_BRANCH}
    sha = _existing_sha(repo_path)
    if sha:
        payload["sha"] = sha

    url = f"{_API_ROOT}/repos/{GITHUB_REPO}/contents/{repo_path}"
    response = requests.put(url, headers=_headers(), json=payload)
    response.raise_for_status()
    return f"backed up to {GITHUB_REPO}@{GITHUB_BRANCH}:{repo_path}"
