# Role-Based Access Control

> Controls who can do what in Supply Chain Command Center through JWT authentication (JSON Web Tokens) and role-permission mappings.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | N/A (infrastructure layer) |
| **Key Files** | `api/routers/auth_router.py`, `api/routers/users.py`, `api/auth.py`, `config/auth_config.yaml` |

---

## Problem

Without access control, every user can run jobs, change policies, and modify forecasts. There is no way to limit analyst access to read-only, restrict job execution to planners, or audit who changed what. In a multi-user planning team, this creates risk of accidental or unauthorized changes.

## Solution

A JWT-based authentication layer with four predefined roles (admin, planner, analyst, viewer). Users log in with email and password, receive a token, and the system checks role-permission mappings on every request. All mutations (changes to data) are logged to an audit trail. The existing API key auth (`X-API-Key` header) remains for backward compatibility -- RBAC is additive.

## How It Works

1. User submits email + password to `POST /auth/login`
2. Server verifies password (bcrypt hash), issues a JWT access token (30 min) and refresh token (7 days)
3. Client includes `Authorization: Bearer <token>` on subsequent requests
4. FastAPI dependency extracts the token, checks role membership against the permission matrix
5. If the user's role lacks the required permission, the request is rejected with 401
6. Refresh tokens allow seamless token renewal without re-entering credentials
7. All mutations are written to `fact_audit_log` with user ID, action, resource, and timestamp

## Data Model

### `dim_user`

| Column | Type | Description |
|---|---|---|
| `user_id` | `SERIAL PK` | Auto-increment user ID |
| `email` | `TEXT UNIQUE` | Login email address |
| `display_name` | `TEXT` | User's display name |
| `password_hash` | `TEXT` | bcrypt-hashed password |
| `role` | `TEXT` | Role name (default: `viewer`) |
| `is_active` | `BOOLEAN` | Soft-delete flag |
| `created_at` | `TIMESTAMPTZ` | Account creation time |
| `last_login_at` | `TIMESTAMPTZ` | Most recent login |

### `dim_role`

| Column | Type | Description |
|---|---|---|
| `role_name` | `TEXT PK` | Role identifier |
| `permissions` | `TEXT[]` | Array of permission strings |
| `description` | `TEXT` | Human-readable role description |

### `fact_audit_log`

| Column | Type | Description |
|---|---|---|
| `log_id` | `BIGSERIAL PK` | Auto-increment log ID |
| `user_id` | `INTEGER FK` | Who performed the action |
| `action` | `TEXT` | Action type: create, update, delete, login, logout |
| `resource` | `TEXT` | Table or endpoint affected |
| `resource_id` | `TEXT` | Specific entity identifier |
| `details` | `JSONB` | Additional context |
| `ip_address` | `INET` | Client IP |
| `created_at` | `TIMESTAMPTZ` | When the action occurred |

## Roles and Permissions

| Role | Permissions | Typical User |
|---|---|---|
| `admin` | `*` (everything) | System administrator |
| `planner` | `read`, `write:forecast`, `write:policy`, `write:exceptions`, `run:jobs` | Supply chain planner |
| `analyst` | `read`, `run:jobs` | Supply chain analyst |
| `viewer` | `read` | Executive, stakeholder |

## API

| Method | Path | Auth | Description |
|---|---|---|---|
| POST | `/auth/login` | None | Authenticate, returns access + refresh tokens |
| POST | `/auth/refresh` | None | Exchange refresh token for new access token |
| POST | `/auth/logout` | Any | Invalidate refresh token |
| GET | `/users` | admin | List all users with roles |
| POST | `/users` | admin | Create a new user |
| GET | `/users/me` | Any | Current user profile from token |
| PUT | `/users/{id}` | admin | Update user fields |
| PUT | `/users/{id}/role` | admin | Change a user's role |
| DELETE | `/users/{id}` | admin | Deactivate user (soft delete) |

## Configuration

`config/auth_config.yaml`:

```yaml
jwt:
  secret_key_env: "JWT_SECRET_KEY"
  algorithm: "HS256"
  access_token_expires_minutes: 30
  refresh_token_expires_days: 7
roles:
  admin:
    permissions: ["*"]
  planner:
    permissions: ["read", "write:forecast", "write:policy", "write:exceptions", "run:jobs"]
  analyst:
    permissions: ["read", "run:jobs"]
  viewer:
    permissions: ["read"]
default_role: viewer
```

JWT secret is loaded from the `JWT_SECRET_KEY` environment variable. In development, it is auto-generated if unset.

## Dependencies

- `PyJWT>=2.8` for token creation and verification
- `bcrypt>=4.0` for password hashing
- Existing `api/auth.py` (`require_api_key`) remains for backward compatibility

## See Also

- [Integration Architecture](./01-integration-architecture.md) -- overall integration overview
- [API Governance](./09-api-governance.md) -- rate limiting per client/role
- [Collaboration](./05-collaboration.md) -- annotations use author identity from RBAC
