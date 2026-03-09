package service

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/apicompat"
	"github.com/Wei-Shaw/sub2api/internal/pkg/logger"
	"github.com/Wei-Shaw/sub2api/internal/util/responseheaders"
	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

// ForwardAsAnthropic accepts an Anthropic Messages request body, converts it
// to OpenAI Responses API format, forwards to the OpenAI upstream, and converts
// the response back to Anthropic Messages format. This enables Claude Code
// clients to access OpenAI models through the standard /v1/messages endpoint.
func (s *OpenAIGatewayService) ForwardAsAnthropic(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	body []byte,
	promptCacheKey string,
	defaultMappedModel string,
) (*OpenAIForwardResult, error) {
	startTime := time.Now()

	// 1. Parse Anthropic request
	var anthropicReq apicompat.AnthropicRequest
	if err := json.Unmarshal(body, &anthropicReq); err != nil {
		return nil, fmt.Errorf("parse anthropic request: %w", err)
	}
	originalModel := anthropicReq.Model
	// clientWantsStream preserves the caller's original preference.
	// isStream may be overridden to true for OAuth upstreams that require SSE.
	clientWantsStream := anthropicReq.Stream
	isStream := anthropicReq.Stream

	// 2. Convert Anthropic → Responses
	responsesReq, err := apicompat.AnthropicToResponses(&anthropicReq)
	if err != nil {
		return nil, fmt.Errorf("convert anthropic to responses: %w", err)
	}

	// 3. Model mapping
	mappedModel := account.GetMappedModel(originalModel)
	// 分组级降级：账号未映射时使用分组默认映射模型
	if mappedModel == originalModel && defaultMappedModel != "" {
		mappedModel = defaultMappedModel
	}
	responsesReq.Model = mappedModel

	logger.L().Debug("openai messages: model mapping applied",
		zap.Int64("account_id", account.ID),
		zap.String("original_model", originalModel),
		zap.String("mapped_model", mappedModel),
		zap.Bool("stream", isStream),
	)

	// 4. Marshal Responses request body, then apply OAuth codex transform
	responsesBody, err := json.Marshal(responsesReq)
	if err != nil {
		return nil, fmt.Errorf("marshal responses request: %w", err)
	}

	if account.Type == AccountTypeOAuth {
		var reqBody map[string]any
		if err := json.Unmarshal(responsesBody, &reqBody); err != nil {
			return nil, fmt.Errorf("unmarshal for codex transform: %w", err)
		}
		applyCodexOAuthTransform(reqBody, false, false)
		// OAuth upstream only supports SSE; force stream=true for the upstream
		// request. clientWantsStream is unchanged so the response handler will
		// still honour the caller's original stream preference.
		isStream = true
		responsesBody, err = json.Marshal(reqBody)
		if err != nil {
			return nil, fmt.Errorf("remarshal after codex transform: %w", err)
		}
	}

	// 5. Get access token
	token, _, err := s.GetAccessToken(ctx, account)
	if err != nil {
		return nil, fmt.Errorf("get access token: %w", err)
	}

	// 6. Build upstream request
	upstreamReq, err := s.buildUpstreamRequest(ctx, c, account, responsesBody, token, isStream, promptCacheKey, false)
	if err != nil {
		return nil, fmt.Errorf("build upstream request: %w", err)
	}

	// 7. Send request
	proxyURL := ""
	if account.Proxy != nil {
		proxyURL = account.Proxy.URL()
	}
	resp, err := s.httpUpstream.Do(upstreamReq, proxyURL, account.ID, account.Concurrency)
	if err != nil {
		safeErr := sanitizeUpstreamErrorMessage(err.Error())
		setOpsUpstreamError(c, 0, safeErr, "")
		appendOpsUpstreamError(c, OpsUpstreamErrorEvent{
			Platform:           account.Platform,
			AccountID:          account.ID,
			AccountName:        account.Name,
			UpstreamStatusCode: 0,
			Kind:               "request_error",
			Message:            safeErr,
		})
		writeAnthropicError(c, http.StatusBadGateway, "api_error", "Upstream request failed")
		return nil, fmt.Errorf("upstream request failed: %s", safeErr)
	}
	defer func() { _ = resp.Body.Close() }()

	// 8. Handle error response with failover
	if resp.StatusCode >= 400 {
		if s.shouldFailoverUpstreamError(resp.StatusCode) {
			respBody, _ := io.ReadAll(io.LimitReader(resp.Body, 2<<20))
			_ = resp.Body.Close()

			upstreamMsg := strings.TrimSpace(extractUpstreamErrorMessage(respBody))
			upstreamMsg = sanitizeUpstreamErrorMessage(upstreamMsg)
			upstreamDetail := ""
			if s.cfg != nil && s.cfg.Gateway.LogUpstreamErrorBody {
				maxBytes := s.cfg.Gateway.LogUpstreamErrorBodyMaxBytes
				if maxBytes <= 0 {
					maxBytes = 2048
				}
				upstreamDetail = truncateString(string(respBody), maxBytes)
			}
			appendOpsUpstreamError(c, OpsUpstreamErrorEvent{
				Platform:           account.Platform,
				AccountID:          account.ID,
				AccountName:        account.Name,
				UpstreamStatusCode: resp.StatusCode,
				UpstreamRequestID:  resp.Header.Get("x-request-id"),
				Kind:               "failover",
				Message:            upstreamMsg,
				Detail:             upstreamDetail,
			})
			if s.rateLimitService != nil {
				s.rateLimitService.HandleUpstreamError(ctx, account, resp.StatusCode, resp.Header, respBody)
			}
			return nil, &UpstreamFailoverError{StatusCode: resp.StatusCode, ResponseBody: respBody}
		}
		// Non-failover error: return Anthropic-formatted error to client
		return s.handleAnthropicErrorResponse(resp, c, account)
	}

	// 9. Handle normal response.
	// Use clientWantsStream (not isStream) to decide how to respond to the
	// caller: for OAuth accounts isStream is forced true so the upstream always
	// speaks SSE, but we must still honour the caller's original preference.
	var result *OpenAIForwardResult
	var handleErr error
	switch {
	case clientWantsStream:
		// Client wants SSE; upstream is also streaming — pass through directly.
		result, handleErr = s.handleAnthropicStreamingResponse(resp, c, originalModel, mappedModel, startTime)
	case isStream:
		// Upstream is streaming (OAuth forced) but client wants a single JSON
		// object — collect the SSE and assemble the non-streaming response.
		result, handleErr = s.handleAnthropicStreamToNonStreamingResponse(resp, c, originalModel, mappedModel, startTime)
	default:
		// Both client and upstream are non-streaming.
		result, handleErr = s.handleAnthropicNonStreamingResponse(resp, c, originalModel, mappedModel, startTime)
	}

	// Extract and save Codex usage snapshot from response headers (for OAuth accounts)
	if handleErr == nil && account.Type == AccountTypeOAuth {
		if snapshot := ParseCodexRateLimitHeaders(resp.Header); snapshot != nil {
			s.updateCodexUsageSnapshot(ctx, account.ID, snapshot)
		}
	}

	return result, handleErr
}

// handleAnthropicErrorResponse reads an upstream error and returns it in
// Anthropic error format.
func (s *OpenAIGatewayService) handleAnthropicErrorResponse(
	resp *http.Response,
	c *gin.Context,
	account *Account,
) (*OpenAIForwardResult, error) {
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 2<<20))

	upstreamMsg := strings.TrimSpace(extractUpstreamErrorMessage(body))
	if upstreamMsg == "" {
		upstreamMsg = fmt.Sprintf("Upstream error: %d", resp.StatusCode)
	}
	upstreamMsg = sanitizeUpstreamErrorMessage(upstreamMsg)

	// Record upstream error details for ops logging
	upstreamDetail := ""
	if s.cfg != nil && s.cfg.Gateway.LogUpstreamErrorBody {
		maxBytes := s.cfg.Gateway.LogUpstreamErrorBodyMaxBytes
		if maxBytes <= 0 {
			maxBytes = 2048
		}
		upstreamDetail = truncateString(string(body), maxBytes)
	}
	setOpsUpstreamError(c, resp.StatusCode, upstreamMsg, upstreamDetail)

	// Apply error passthrough rules (matches handleErrorResponse pattern in openai_gateway_service.go)
	if status, errType, errMsg, matched := applyErrorPassthroughRule(
		c, account.Platform, resp.StatusCode, body,
		http.StatusBadGateway, "api_error", "Upstream request failed",
	); matched {
		writeAnthropicError(c, status, errType, errMsg)
		if upstreamMsg == "" {
			upstreamMsg = errMsg
		}
		if upstreamMsg == "" {
			return nil, fmt.Errorf("upstream error: %d (passthrough rule matched)", resp.StatusCode)
		}
		return nil, fmt.Errorf("upstream error: %d (passthrough rule matched) message=%s", resp.StatusCode, upstreamMsg)
	}

	errType := "api_error"
	switch {
	case resp.StatusCode == 400:
		errType = "invalid_request_error"
	case resp.StatusCode == 404:
		errType = "not_found_error"
	case resp.StatusCode == 429:
		errType = "rate_limit_error"
	case resp.StatusCode >= 500:
		errType = "api_error"
	}

	writeAnthropicError(c, resp.StatusCode, errType, upstreamMsg)
	return nil, fmt.Errorf("upstream error: %d %s", resp.StatusCode, upstreamMsg)
}

// handleAnthropicNonStreamingResponse reads a Responses API JSON response,
// converts it to Anthropic Messages format, and writes it to the client.
func (s *OpenAIGatewayService) handleAnthropicNonStreamingResponse(
	resp *http.Response,
	c *gin.Context,
	originalModel string,
	mappedModel string,
	startTime time.Time,
) (*OpenAIForwardResult, error) {
	requestID := resp.Header.Get("x-request-id")

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read upstream response: %w", err)
	}

	var responsesResp apicompat.ResponsesResponse
	if err := json.Unmarshal(respBody, &responsesResp); err != nil {
		return nil, fmt.Errorf("parse responses response: %w", err)
	}

	anthropicResp := apicompat.ResponsesToAnthropic(&responsesResp, originalModel)

	var usage OpenAIUsage
	if responsesResp.Usage != nil {
		usage = OpenAIUsage{
			InputTokens:  responsesResp.Usage.InputTokens,
			OutputTokens: responsesResp.Usage.OutputTokens,
		}
		if responsesResp.Usage.InputTokensDetails != nil {
			usage.CacheReadInputTokens = responsesResp.Usage.InputTokensDetails.CachedTokens
		}
	}

	if s.responseHeaderFilter != nil {
		responseheaders.WriteFilteredHeaders(c.Writer.Header(), resp.Header, s.responseHeaderFilter)
	}
	c.JSON(http.StatusOK, anthropicResp)

	return &OpenAIForwardResult{
		RequestID:    requestID,
		Usage:        usage,
		Model:        originalModel,
		BillingModel: mappedModel,
		Stream:       false,
		Duration:     time.Since(startTime),
	}, nil
}

// handleAnthropicStreamingResponse reads Responses SSE events from upstream,
// converts each to Anthropic SSE events, and writes them to the client.
func (s *OpenAIGatewayService) handleAnthropicStreamingResponse(
	resp *http.Response,
	c *gin.Context,
	originalModel string,
	mappedModel string,
	startTime time.Time,
) (*OpenAIForwardResult, error) {
	requestID := resp.Header.Get("x-request-id")

	if s.responseHeaderFilter != nil {
		responseheaders.WriteFilteredHeaders(c.Writer.Header(), resp.Header, s.responseHeaderFilter)
	}
	c.Writer.Header().Set("Content-Type", "text/event-stream")
	c.Writer.Header().Set("Cache-Control", "no-cache")
	c.Writer.Header().Set("Connection", "keep-alive")
	c.Writer.WriteHeader(http.StatusOK)

	state := apicompat.NewResponsesEventToAnthropicState()
	state.Model = originalModel
	var usage OpenAIUsage
	var firstTokenMs *int
	firstChunk := true

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	for scanner.Scan() {
		line := scanner.Text()

		if !strings.HasPrefix(line, "data: ") || line == "data: [DONE]" {
			continue
		}
		payload := line[6:]

		if firstChunk {
			firstChunk = false
			ms := int(time.Since(startTime).Milliseconds())
			firstTokenMs = &ms
		}

		// Parse the Responses SSE event
		var event apicompat.ResponsesStreamEvent
		if err := json.Unmarshal([]byte(payload), &event); err != nil {
			logger.L().Warn("openai messages stream: failed to parse event",
				zap.Error(err),
				zap.String("request_id", requestID),
			)
			continue
		}

		// Extract usage from completion events
		if (event.Type == "response.completed" || event.Type == "response.incomplete" || event.Type == "response.failed") &&
			event.Response != nil && event.Response.Usage != nil {
			usage = OpenAIUsage{
				InputTokens:  event.Response.Usage.InputTokens,
				OutputTokens: event.Response.Usage.OutputTokens,
			}
			if event.Response.Usage.InputTokensDetails != nil {
				usage.CacheReadInputTokens = event.Response.Usage.InputTokensDetails.CachedTokens
			}
		}

		// Convert to Anthropic events
		events := apicompat.ResponsesEventToAnthropicEvents(&event, state)
		for _, evt := range events {
			sse, err := apicompat.ResponsesAnthropicEventToSSE(evt)
			if err != nil {
				logger.L().Warn("openai messages stream: failed to marshal event",
					zap.Error(err),
					zap.String("request_id", requestID),
				)
				continue
			}
			if _, err := fmt.Fprint(c.Writer, sse); err != nil {
				// Client disconnected — return collected usage
				logger.L().Info("openai messages stream: client disconnected",
					zap.String("request_id", requestID),
				)
				return &OpenAIForwardResult{
					RequestID:    requestID,
					Usage:        usage,
					Model:        originalModel,
					BillingModel: mappedModel,
					Stream:       true,
					Duration:     time.Since(startTime),
					FirstTokenMs: firstTokenMs,
				}, nil
			}
		}
		if len(events) > 0 {
			c.Writer.Flush()
		}
	}

	if err := scanner.Err(); err != nil {
		if !errors.Is(err, context.Canceled) && !errors.Is(err, context.DeadlineExceeded) {
			logger.L().Warn("openai messages stream: read error",
				zap.Error(err),
				zap.String("request_id", requestID),
			)
		}
	}

	// Ensure the Anthropic stream is properly terminated
	if finalEvents := apicompat.FinalizeResponsesAnthropicStream(state); len(finalEvents) > 0 {
		for _, evt := range finalEvents {
			sse, err := apicompat.ResponsesAnthropicEventToSSE(evt)
			if err != nil {
				continue
			}
			fmt.Fprint(c.Writer, sse) //nolint:errcheck
		}
		c.Writer.Flush()
	}

	return &OpenAIForwardResult{
		RequestID:    requestID,
		Usage:        usage,
		Model:        originalModel,
		BillingModel: mappedModel,
		Stream:       true,
		Duration:     time.Since(startTime),
		FirstTokenMs: firstTokenMs,
	}, nil
}

// handleAnthropicStreamToNonStreamingResponse handles the case where the
// upstream always returns SSE (e.g. OAuth accounts) but the client requested a
// non-streaming JSON response.  It reads the full Responses SSE stream,
// reassembles it into an Anthropic Messages response object, and writes a
// single application/json body to the client.
func (s *OpenAIGatewayService) handleAnthropicStreamToNonStreamingResponse(
	resp *http.Response,
	c *gin.Context,
	originalModel string,
	mappedModel string,
	startTime time.Time,
) (*OpenAIForwardResult, error) {
	requestID := resp.Header.Get("x-request-id")

	state := apicompat.NewResponsesEventToAnthropicState()
	state.Model = originalModel

	// contentBlock accumulates content for a single Anthropic block index.
	type contentBlock struct {
		typ       string
		textBuf   strings.Builder // text / thinking content
		inputBuf  strings.Builder // partial JSON for tool_use input
		id        string
		name      string
		inputJSON json.RawMessage // pre-built input for server_tool_use
		toolUseID string          // web_search_tool_result
		content   json.RawMessage // web_search_tool_result content
	}
	blocks := make(map[int]*contentBlock)
	stopReason := "end_turn"

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") || line == "data: [DONE]" {
			continue
		}
		payload := line[6:]

		var event apicompat.ResponsesStreamEvent
		if err := json.Unmarshal([]byte(payload), &event); err != nil {
			logger.L().Warn("openai messages stream-to-nonstream: failed to parse event",
				zap.Error(err),
				zap.String("request_id", requestID),
			)
			continue
		}

		for _, evt := range apicompat.ResponsesEventToAnthropicEvents(&event, state) {
			switch evt.Type {
			case "content_block_start":
				if evt.Index == nil || evt.ContentBlock == nil {
					continue
				}
				b := &contentBlock{typ: evt.ContentBlock.Type}
				switch evt.ContentBlock.Type {
				case "tool_use":
					b.id = evt.ContentBlock.ID
					b.name = evt.ContentBlock.Name
				case "server_tool_use":
					b.id = evt.ContentBlock.ID
					b.name = evt.ContentBlock.Name
					b.inputJSON = evt.ContentBlock.Input
				case "web_search_tool_result":
					b.toolUseID = evt.ContentBlock.ToolUseID
					b.content = evt.ContentBlock.Content
				}
				blocks[*evt.Index] = b

			case "content_block_delta":
				if evt.Index == nil || evt.Delta == nil {
					continue
				}
				b, ok := blocks[*evt.Index]
				if !ok {
					continue
				}
				switch evt.Delta.Type {
				case "text_delta":
					b.textBuf.WriteString(evt.Delta.Text)
				case "input_json_delta":
					b.inputBuf.WriteString(evt.Delta.PartialJSON)
				case "thinking_delta":
					b.textBuf.WriteString(evt.Delta.Thinking)
				}

			case "message_delta":
				if evt.Delta != nil && evt.Delta.StopReason != "" {
					stopReason = evt.Delta.StopReason
				}
			}
		}
	}

	if err := scanner.Err(); err != nil {
		if !errors.Is(err, context.Canceled) && !errors.Is(err, context.DeadlineExceeded) {
			logger.L().Warn("openai messages stream-to-nonstream: read error",
				zap.Error(err),
				zap.String("request_id", requestID),
			)
		}
	}

	// Assemble content blocks in index order.
	var content []apicompat.AnthropicContentBlock
	for idx := 0; idx < len(blocks); idx++ {
		b, ok := blocks[idx]
		if !ok {
			continue
		}
		switch b.typ {
		case "text":
			content = append(content, apicompat.AnthropicContentBlock{
				Type: "text",
				Text: b.textBuf.String(),
			})
		case "thinking":
			content = append(content, apicompat.AnthropicContentBlock{
				Type:     "thinking",
				Thinking: b.textBuf.String(),
			})
		case "tool_use":
			inputRaw := json.RawMessage("{}")
			if js := b.inputBuf.String(); js != "" {
				inputRaw = json.RawMessage(js)
			}
			content = append(content, apicompat.AnthropicContentBlock{
				Type:  "tool_use",
				ID:    b.id,
				Name:  b.name,
				Input: inputRaw,
			})
		case "server_tool_use":
			content = append(content, apicompat.AnthropicContentBlock{
				Type:  "server_tool_use",
				ID:    b.id,
				Name:  b.name,
				Input: b.inputJSON,
			})
		case "web_search_tool_result":
			content = append(content, apicompat.AnthropicContentBlock{
				Type:      "web_search_tool_result",
				ToolUseID: b.toolUseID,
				Content:   b.content,
			})
		}
	}
	if len(content) == 0 {
		content = append(content, apicompat.AnthropicContentBlock{Type: "text", Text: ""})
	}

	usage := OpenAIUsage{
		InputTokens:          state.InputTokens,
		OutputTokens:         state.OutputTokens,
		CacheReadInputTokens: state.CacheReadInputTokens,
	}
	anthropicResp := &apicompat.AnthropicResponse{
		ID:         state.ResponseID,
		Type:       "message",
		Role:       "assistant",
		Content:    content,
		Model:      originalModel,
		StopReason: stopReason,
		Usage: apicompat.AnthropicUsage{
			InputTokens:          state.InputTokens,
			OutputTokens:         state.OutputTokens,
			CacheReadInputTokens: state.CacheReadInputTokens,
		},
	}

	if s.responseHeaderFilter != nil {
		responseheaders.WriteFilteredHeaders(c.Writer.Header(), resp.Header, s.responseHeaderFilter)
	}
	c.JSON(http.StatusOK, anthropicResp)

	return &OpenAIForwardResult{
		RequestID:    requestID,
		Usage:        usage,
		Model:        originalModel,
		BillingModel: mappedModel,
		Stream:       false,
		Duration:     time.Since(startTime),
	}, nil
}

// writeAnthropicError writes an error response in Anthropic Messages API format.
func writeAnthropicError(c *gin.Context, statusCode int, errType, message string) {
	c.JSON(statusCode, gin.H{
		"type": "error",
		"error": gin.H{
			"type":    errType,
			"message": message,
		},
	})
}
