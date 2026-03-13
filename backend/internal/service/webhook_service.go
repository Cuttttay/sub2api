package service

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

// WebhookService sends test failure notifications to webhooks.
// Supports Feishu, WeChat Work (WeCom), DingTalk, and generic webhooks.
type WebhookService struct {
	client *http.Client
}

func NewWebhookService() *WebhookService {
	return &WebhookService{
		client: &http.Client{Timeout: 10 * time.Second},
	}
}

// WebhookTestFailurePayload holds the data needed to build a failure notification.
type WebhookTestFailurePayload struct {
	PlanID         int64
	AccountID      int64
	ModelID        string
	Status         string
	ErrorMessage   string
	LatencyMs      int64
	StartedAt      time.Time
	WebhookHeaders map[string]string
}

// SendTestFailure posts a failure notification to the given webhook URL.
// It auto-detects the platform from the URL and formats the body accordingly.
func (s *WebhookService) SendTestFailure(ctx context.Context, webhookURL string, p WebhookTestFailurePayload) error {
	if webhookURL == "" {
		return nil
	}

	body, err := buildWebhookBody(webhookURL, p)
	if err != nil {
		return fmt.Errorf("webhook: build body: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, webhookURL, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("webhook: create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	for k, v := range p.WebhookHeaders {
		req.Header.Set(k, v)
	}

	resp, err := s.client.Do(req)
	if err != nil {
		return fmt.Errorf("webhook: send: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode >= 300 {
		return fmt.Errorf("webhook: unexpected status %d", resp.StatusCode)
	}
	return nil
}

func buildWebhookBody(webhookURL string, p WebhookTestFailurePayload) ([]byte, error) {
	errMsg := p.ErrorMessage
	if len(errMsg) > 200 {
		errMsg = errMsg[:200] + "..."
	}

	text := fmt.Sprintf(
		"[定时测试失败] 账号 ID: %d | 模型: %s\n状态: %s\n错误: %s\n时间: %s",
		p.AccountID, p.ModelID, p.Status,
		errMsg,
		p.StartedAt.Format("2006-01-02 15:04:05"),
	)

	switch {
	case strings.Contains(webhookURL, "feishu.cn") || strings.Contains(webhookURL, "larksuite.com"):
		return json.Marshal(map[string]any{
			"msg_type": "text",
			"content":  map[string]string{"text": text},
		})
	case strings.Contains(webhookURL, "qyapi.weixin.qq.com"):
		return json.Marshal(map[string]any{
			"msgtype": "text",
			"text":    map[string]string{"content": text},
		})
	case strings.Contains(webhookURL, "oapi.dingtalk.com"):
		return json.Marshal(map[string]any{
			"msgtype": "text",
			"text":    map[string]string{"content": text},
		})
	default:
		// Generic JSON payload — compatible with most custom webhook receivers.
		return json.Marshal(map[string]any{
			"event":      "scheduled_test_failure",
			"plan_id":    p.PlanID,
			"account_id": p.AccountID,
			"model_id":   p.ModelID,
			"status":     p.Status,
			"error":      p.ErrorMessage,
			"latency_ms": p.LatencyMs,
			"started_at": p.StartedAt.Format(time.RFC3339),
			"text":       text,
		})
	}
}
