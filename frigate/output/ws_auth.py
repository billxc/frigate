"""Authorization helpers for JSMPEG websocket clients."""

from typing import Any

from frigate.config import FrigateConfig
from frigate.models import User


def _get_valid_ws_roles(environ: dict[str, Any], config: FrigateConfig) -> list[str]:
    role_header = environ.get("HTTP_REMOTE_ROLE", "")
    roles = [
        role.strip()
        for role in role_header.split(config.proxy.separator)
        if role.strip()
    ]
    return [role for role in roles if role in config.auth.roles]


def ws_has_camera_access(ws: Any, camera_name: str, config: FrigateConfig) -> bool:
    """Return True when a websocket client is authorized for the camera path."""
    environ = ws.environ
    return environ is not None and ws_environ_has_camera_access(
        environ, camera_name, config
    )


def ws_environ_has_camera_access(
    environ: dict[str, Any], camera_name: str, config: FrigateConfig
) -> bool:
    """Authorize a camera using a captured websocket environment."""
    roles = _get_valid_ws_roles(environ, config)

    if not roles:
        return False

    roles_dict = config.auth.roles

    # Birdseye is a composite stream, so only users with unrestricted access
    # should receive it.
    if camera_name == "birdseye":
        return any(role == "admin" or not roles_dict.get(role) for role in roles)

    all_camera_names = set(config.cameras.keys())

    for role in roles:
        if role == "admin" or not roles_dict.get(role):
            return True

        allowed_cameras = User.get_allowed_cameras(role, roles_dict, all_camera_names)
        if camera_name in allowed_cameras:
            return True

    return False
