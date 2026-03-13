-- 072: Add webhook notification fields to scheduled_test_plans
-- When notify_on_failure is enabled, a POST request is sent to webhook_url on test failure.

ALTER TABLE scheduled_test_plans ADD COLUMN IF NOT EXISTS webhook_url      VARCHAR(500) NOT NULL DEFAULT '';
ALTER TABLE scheduled_test_plans ADD COLUMN IF NOT EXISTS notify_on_failure BOOLEAN     NOT NULL DEFAULT false;
