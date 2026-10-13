"""
auth.py
-------
Minimal session-based authentication for the Streamlit app.

Passwords are stored as bcrypt hashes in a local JSON file (users.json),
never in plaintext. This is intentionally lightweight (no external auth
service) so the whole app is self-contained and easy to run; swap in
OAuth/SSO or a real user database for production use.

Default demo accounts (created automatically on first run if users.json
doesn't exist):
    username: admin   password: admin123
    username: demo    password: demo1234
CHANGE OR REMOVE THESE before deploying anywhere real.
"""

import json
import os
from typing import Optional

import bcrypt

from . import config


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def _default_users() -> dict:
    return {
        "admin": {"password_hash": _hash_password("admin123"), "display_name": "Admin"},
        "demo": {"password_hash": _hash_password("demo1234"), "display_name": "Demo User"},
    }


def _ensure_users_file() -> None:
    if not os.path.exists(config.USERS_FILE):
        with open(config.USERS_FILE, "w") as f:
            json.dump(_default_users(), f, indent=2)


def load_users() -> dict:
    _ensure_users_file()
    with open(config.USERS_FILE, "r") as f:
        return json.load(f)


def authenticate(username: str, password: str) -> Optional[dict]:
    """Returns the user record (without the password hash) on success, else None."""
    users = load_users()
    user = users.get(username)
    if user and _verify_password(password, user["password_hash"]):
        return {"username": username, "display_name": user.get("display_name", username)}
    return None


def register_user(username: str, password: str, display_name: str = None) -> bool:
    """Adds a new user. Returns False if the username already exists."""
    users = load_users()
    if username in users:
        return False
    users[username] = {
        "password_hash": _hash_password(password),
        "display_name": display_name or username,
    }
    with open(config.USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)
    return True
