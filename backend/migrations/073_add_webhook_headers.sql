-- 073: Add custom request headers to scheduled_test_plans
-- Stored as a JSON object, e.g. {"Authorization":"Bearer xxx","X-Custom":"value"}

ALTER TABLE scheduled_test_plans ADD COLUMN IF NOT EXISTS webhook_headers TEXT NOT NULL DEFAULT '{}';
