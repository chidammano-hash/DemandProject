-- 08-02: User Management & RBAC
-- dim_user, fact_audit_log

-- User roles enum
DO $$ BEGIN
  CREATE TYPE user_role AS ENUM ('viewer', 'planner', 'manager', 'admin');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- -------------------------------------------------------
-- dim_user — platform users with role-based access
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_user (
  user_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email         TEXT NOT NULL UNIQUE,
  display_name  TEXT NOT NULL DEFAULT '',
  role          user_role NOT NULL DEFAULT 'viewer',
  password_hash TEXT NOT NULL,
  is_active     BOOLEAN NOT NULL DEFAULT TRUE,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  last_login_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_user_email ON dim_user (email);
CREATE INDEX IF NOT EXISTS idx_user_role  ON dim_user (role);

-- -------------------------------------------------------
-- fact_audit_log — change tracking for compliance
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS fact_audit_log (
  audit_id      BIGSERIAL PRIMARY KEY,
  user_id       UUID REFERENCES dim_user(user_id) ON DELETE SET NULL,
  action        TEXT NOT NULL,            -- e.g. 'create', 'update', 'delete', 'login'
  resource_type TEXT NOT NULL DEFAULT '', -- e.g. 'policy', 'insight', 'forecast_override'
  resource_id   TEXT NOT NULL DEFAULT '', -- identifier of the affected resource
  old_value     JSONB,
  new_value     JSONB,
  ip_address    TEXT,
  user_agent    TEXT,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_audit_user_ts     ON fact_audit_log (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_resource     ON fact_audit_log (resource_type, resource_id);
CREATE INDEX IF NOT EXISTS idx_audit_created       ON fact_audit_log (created_at DESC);

-- Seed a default admin user (password: 'admin123' — change immediately in production)
-- bcrypt hash of 'admin123' generated offline
INSERT INTO dim_user (email, display_name, role, password_hash)
VALUES ('admin@demandstudio.local', 'System Admin', 'admin',
        '$2b$12$LJ3m5ZVOFkSZ0Y9aMJhOBeLnhHv1V0JVkFQ5.Y9YFtYfNwX3HXKy2')
ON CONFLICT (email) DO NOTHING;
