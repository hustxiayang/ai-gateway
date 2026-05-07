// Copyright Envoy AI Gateway Authors
// SPDX-License-Identifier: Apache-2.0
// The full text of the Apache license is available in the LICENSE file at
// the root of the repo.

package translator

import (
	"fmt"
	"io"
	"strconv"
	"strings"

	"google.golang.org/genai"

	"github.com/envoyproxy/ai-gateway/internal/apischema/gcp"
	"github.com/envoyproxy/ai-gateway/internal/apischema/openai"
	"github.com/envoyproxy/ai-gateway/internal/internalapi"
	"github.com/envoyproxy/ai-gateway/internal/json"
	"github.com/envoyproxy/ai-gateway/internal/metrics"
	tracing "github.com/envoyproxy/ai-gateway/internal/tracing/tracingapi"
)

const (
	gcpMethodPredict      = "predict"
	gcpMethodEmbedContent = "embedContent"
)

// NewEmbeddingOpenAIToGCPVertexAITranslator implements [Factory] for OpenAI to GCP VertexAI translation
// for embeddings.
func NewEmbeddingOpenAIToGCPVertexAITranslator(requestModel internalapi.RequestModel, modelNameOverride internalapi.ModelNameOverride) OpenAIEmbeddingTranslator {
	return &openAIToGCPVertexAITranslatorV1Embedding{
		requestModel:      requestModel,
		modelNameOverride: modelNameOverride,
	}
}

// openAIToGCPVertexAITranslatorV1Embedding translates OpenAI Embeddings API to GCP Vertex AI Gemini Embeddings API.
// It auto-detects the endpoint based on model name:
//   - Older models (text-embedding-004, gemini-embedding-001): predict endpoint
//   - Newer models (gemini-embedding-2-*, maas-*): embedContent endpoint
//
// https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api
type openAIToGCPVertexAITranslatorV1Embedding struct {
	requestModel      internalapi.RequestModel
	modelNameOverride internalapi.ModelNameOverride
	useEmbedContent   bool
}

// createInstancesFromEmbeddingInputItem converts an EmbeddingInputItem to GCP Instance(s).
// This handles the mapping of OpenAI's extended embedding input format to GCP's instance format,
// including task_type and title metadata for optimized embedding generation.
// When content is an array of strings, each string becomes a separate instance with the same task_type.
func createInstancesFromEmbeddingInputItem(item openai.EmbeddingInputItem, instances []*gcp.Instance) []*gcp.Instance {
	switch v := item.Content.Value.(type) {
	case string:
		instance := &gcp.Instance{Content: v}
		if item.TaskType != "" {
			instance.TaskType = item.TaskType
		}
		// Title is only valid with task_type=RETRIEVAL_DOCUMENT.
		// See: https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/task-types
		if item.TaskType == openai.EmbeddingTaskTypeRetrievalDocument && item.Title != "" {
			instance.Title = item.Title
		}
		instances = append(instances, instance)
	case []string:
		// Multiple strings with the same task_type - each becomes a separate instance
		for _, text := range v {
			instance := &gcp.Instance{Content: text}
			if item.TaskType != "" {
				instance.TaskType = item.TaskType
			}
			// Title is only valid with task_type=RETRIEVAL_DOCUMENT.
			if item.TaskType == openai.EmbeddingTaskTypeRetrievalDocument && item.Title != "" {
				instance.Title = item.Title
			}
			instances = append(instances, instance)
		}
	}
	return instances
}

// setInstances converts OpenAI embedding input to GCP instances.
// It handles multiple input formats: string, []string, EmbeddingInputItem, []EmbeddingInputItem.
// Each input element is converted to a separate GCP Instance for batch embedding generation.
func setInstances(input openai.EmbeddingRequestInput, instances []*gcp.Instance) ([]*gcp.Instance, error) {
	switch v := input.Value.(type) {
	case string:
		instances = append(instances, &gcp.Instance{Content: v})
		return instances, nil
	case []string:
		// Array of strings: create a separate instance for each string.
		for _, text := range v {
			instances = append(instances, &gcp.Instance{Content: text})
		}
		return instances, nil
	case openai.EmbeddingInputItem:
		// Single EmbeddingInputItem with enhanced metadata.
		// Content can be string or []string.
		instances = createInstancesFromEmbeddingInputItem(v, instances)
		return instances, nil
	case []openai.EmbeddingInputItem:
		// Array of EmbeddingInputItem objects with metadata support.
		for _, item := range v {
			instances = createInstancesFromEmbeddingInputItem(item, instances)
		}
		return instances, nil
	default:
		return nil, fmt.Errorf("unsupported input type for embedding: %T (supported: string, []string, EmbeddingInputItem, []EmbeddingInputItem)", v)
	}
}

// openAIEmbeddingToGeminiMessage converts an OpenAI EmbeddingRequest to a GCP PredictRequest.
func openAIEmbeddingToGeminiMessage(openAIReq *openai.EmbeddingRequest) (*gcp.PredictRequest, error) {
	// Convert OpenAI EmbeddingRequest's input to Gemini instances.
	var instances []*gcp.Instance
	instances, err := setInstances(openAIReq.Input, instances)
	if err != nil {
		return nil, err
	}

	// Create the embedding prediction parameters.
	parameters := &gcp.Parameters{}

	// Set output dimensionality if specified.
	if openAIReq.Dimensions != nil && *openAIReq.Dimensions > 0 {
		parameters.OutputDimensionality = *openAIReq.Dimensions
	}

	// Apply vendor-specific fields if present.
	if openAIReq.GCPVertexAIEmbeddingVendorFields != nil {
		// Set auto truncate if specified.
		if openAIReq.AutoTruncate {
			parameters.AutoTruncate = openAIReq.AutoTruncate
		}

		// Apply global task type to all instances if specified.
		// This overrides any task_type set on individual input items.
		if openAIReq.TaskType != "" {
			for _, instance := range instances {
				instance.TaskType = openAIReq.TaskType
			}
		}
	}

	// Build the request using gcp.PredictRequest.
	gcr := &gcp.PredictRequest{
		Instances:  instances,
		Parameters: *parameters,
	}

	return gcr, nil
}

// isEmbedContentModel returns true if the model should use the embedContent endpoint
// instead of the predict endpoint. This mirrors the logic in the genai SDK.
func isEmbedContentModel(model string) bool {
	return (strings.Contains(model, "gemini") && model != "gemini-embedding-001") ||
		strings.Contains(model, "maas")
}

// collectInputTexts extracts all input text strings from an OpenAI EmbeddingRequestInput.
func collectInputTexts(input openai.EmbeddingRequestInput) ([]string, error) {
	switch v := input.Value.(type) {
	case string:
		return []string{v}, nil
	case []string:
		return v, nil
	case openai.EmbeddingInputItem:
		switch c := v.Content.Value.(type) {
		case string:
			return []string{c}, nil
		case []string:
			return c, nil
		}
		return nil, fmt.Errorf("unsupported EmbeddingInputItem content type: %T", v.Content.Value)
	case []openai.EmbeddingInputItem:
		var texts []string
		for _, item := range v {
			switch c := item.Content.Value.(type) {
			case string:
				texts = append(texts, c)
			case []string:
				texts = append(texts, c...)
			default:
				return nil, fmt.Errorf("unsupported EmbeddingInputItem content type: %T", item.Content.Value)
			}
		}
		return texts, nil
	default:
		return nil, fmt.Errorf("unsupported input type for embedding: %T", v)
	}
}

// openAIEmbeddingToEmbedContentRequest converts an OpenAI EmbeddingRequest to a GCP EmbedContentRequest.
// Each input text becomes a separate Part in a single Content object.
func openAIEmbeddingToEmbedContentRequest(openAIReq *openai.EmbeddingRequest) (*gcp.EmbedContentRequest, error) {
	texts, err := collectInputTexts(openAIReq.Input)
	if err != nil {
		return nil, err
	}

	parts := make([]*genai.Part, len(texts))
	for i, text := range texts {
		parts[i] = genai.NewPartFromText(text)
	}

	req := &gcp.EmbedContentRequest{
		Content: genai.Content{Parts: parts},
	}

	if openAIReq.Dimensions != nil && *openAIReq.Dimensions > 0 {
		req.OutputDimensionality = *openAIReq.Dimensions
	}

	if openAIReq.GCPVertexAIEmbeddingVendorFields != nil {
		if openAIReq.AutoTruncate {
			req.AutoTruncate = &openAIReq.AutoTruncate
		}
		if openAIReq.TaskType != "" {
			req.TaskType = openAIReq.TaskType
		}
	}

	// Also check per-item task type from single EmbeddingInputItem (non-vendor field).
	if item, ok := openAIReq.Input.Value.(openai.EmbeddingInputItem); ok && req.TaskType == "" {
		if item.TaskType != "" {
			req.TaskType = item.TaskType
		}
		if item.TaskType == openai.EmbeddingTaskTypeRetrievalDocument && item.Title != "" {
			req.Title = item.Title
		}
	}

	return req, nil
}

// RequestBody implements [OpenAIEmbeddingTranslator.RequestBody] for GCP Gemini.
// This method translates an OpenAI Embedding request to a GCP Gemini Embeddings API request.
func (o *openAIToGCPVertexAITranslatorV1Embedding) RequestBody(_ []byte, req *openai.EmbeddingRequest, _ bool) (
	newHeaders []internalapi.Header, newBody []byte, err error,
) {
	o.requestModel = req.Model
	if o.modelNameOverride != "" {
		// Use modelName override if set.
		o.requestModel = o.modelNameOverride
	}

	var path string

	if isEmbedContentModel(o.requestModel) {
		o.useEmbedContent = true
		path = buildGCPModelPathSuffix(gcpModelPublisherGoogle, o.requestModel, gcpMethodEmbedContent)

		var gcpReq *gcp.EmbedContentRequest
		gcpReq, err = openAIEmbeddingToEmbedContentRequest(req)
		if err != nil {
			return nil, nil, fmt.Errorf("error converting EmbeddingRequest: %w", err)
		}
		newBody, err = json.Marshal(gcpReq)
	} else {
		o.useEmbedContent = false
		path = buildGCPModelPathSuffix(gcpModelPublisherGoogle, o.requestModel, gcpMethodPredict)

		var gcpReq *gcp.PredictRequest
		gcpReq, err = openAIEmbeddingToGeminiMessage(req)
		if err != nil {
			return nil, nil, fmt.Errorf("error converting EmbeddingRequest: %w", err)
		}
		newBody, err = json.Marshal(gcpReq)
	}

	if err != nil {
		return nil, nil, fmt.Errorf("error marshaling Gemini request: %w", err)
	}
	newHeaders = []internalapi.Header{
		{pathHeaderName, path},
		{contentLengthHeaderName, strconv.Itoa(len(newBody))},
	}
	return
}

// ResponseHeaders implements [OpenAIEmbeddingTranslator.ResponseHeaders].
func (o *openAIToGCPVertexAITranslatorV1Embedding) ResponseHeaders(_ map[string]string) (newHeaders []internalapi.Header, err error) {
	return nil, nil
}

// ResponseBody implements [OpenAIEmbeddingTranslator.ResponseBody] for GCP Gemini.
// This method translates a GCP Gemini Embeddings API response to the OpenAI Embeddings format.
// GCP Vertex AI uses deterministic model mapping without virtualization, where the requested model
// is exactly what gets executed. The response does not contain a model field, so we return
// the request model that was originally sent.
func (o *openAIToGCPVertexAITranslatorV1Embedding) ResponseBody(_ map[string]string, body io.Reader, _ bool, span tracing.EmbeddingsSpan) (
	newHeaders []internalapi.Header, newBody []byte, tokenUsage metrics.TokenUsage, responseModel internalapi.ResponseModel, err error,
) {
	// Read the Gemini embedding response.
	respBody, err := io.ReadAll(body)
	if err != nil {
		return nil, nil, tokenUsage, "", fmt.Errorf("failed to read gemini embedding response body: %w", err)
	}

	var openaiResp openai.EmbeddingResponse
	var promptTokens int

	if o.useEmbedContent {
		openaiResp, promptTokens, err = o.parseEmbedContentResponse(respBody)
	} else {
		openaiResp, promptTokens, err = o.parsePredictResponse(respBody)
	}
	if err != nil {
		return nil, nil, tokenUsage, "", err
	}

	// Set token usage from accumulated values.
	openaiResp.Usage.PromptTokens = promptTokens
	openaiResp.Usage.TotalTokens = promptTokens

	// Marshal the OpenAI response.
	newBody, err = json.Marshal(openaiResp)
	if err != nil {
		return nil, nil, tokenUsage, "", fmt.Errorf("failed to marshal OpenAI response: %w", err)
	}

	// Update token usage metrics.
	tokenUsage.SetInputTokens(uint32(promptTokens)) //nolint:gosec
	tokenUsage.SetTotalTokens(uint32(promptTokens)) //nolint:gosec

	// Record the response in the span for tracing.
	if span != nil {
		span.RecordResponse(&openaiResp)
	}

	newHeaders = []internalapi.Header{{contentLengthHeaderName, strconv.Itoa(len(newBody))}}
	responseModel = openaiResp.Model
	return
}

// parsePredictResponse parses a GCP PredictResponse and converts it to the OpenAI format.
func (o *openAIToGCPVertexAITranslatorV1Embedding) parsePredictResponse(respBody []byte) (openai.EmbeddingResponse, int, error) {
	var gcpResp gcp.PredictResponse
	if err := json.Unmarshal(respBody, &gcpResp); err != nil {
		return openai.EmbeddingResponse{}, 0, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	openaiResp := openai.EmbeddingResponse{
		Object: "list",
		Model:  o.requestModel,
	}

	var promptTokens int
	if len(gcpResp.Predictions) > 0 {
		openaiResp.Data = make([]openai.Embedding, len(gcpResp.Predictions))
		for i, prediction := range gcpResp.Predictions {
			if prediction != nil {
				float64Values := make([]float64, len(prediction.Embeddings.Values))
				for j, v := range prediction.Embeddings.Values {
					float64Values[j] = float64(v)
				}
				openaiResp.Data[i] = openai.Embedding{
					Object:    "embedding",
					Index:     i,
					Embedding: openai.EmbeddingUnion{Value: float64Values},
				}
				if prediction.Embeddings.Statistics != nil {
					promptTokens += prediction.Embeddings.Statistics.TokenCount
					openaiResp.Data[i].Truncated = prediction.Embeddings.Statistics.Truncated
				}
			}
		}
	} else {
		openaiResp.Data = []openai.Embedding{}
	}

	return openaiResp, promptTokens, nil
}

// parseEmbedContentResponse parses a GCP EmbedContentResponse and converts it to the OpenAI format.
func (o *openAIToGCPVertexAITranslatorV1Embedding) parseEmbedContentResponse(respBody []byte) (openai.EmbeddingResponse, int, error) {
	var gcpResp gcp.EmbedContentResponse
	if err := json.Unmarshal(respBody, &gcpResp); err != nil {
		return openai.EmbeddingResponse{}, 0, fmt.Errorf("failed to unmarshal embedContent response: %w", err)
	}

	openaiResp := openai.EmbeddingResponse{
		Object: "list",
		Model:  o.requestModel,
	}

	var promptTokens int
	if len(gcpResp.Embeddings) > 0 {
		openaiResp.Data = make([]openai.Embedding, len(gcpResp.Embeddings))
		for i, emb := range gcpResp.Embeddings {
			if emb == nil {
				continue
			}
			float64Values := make([]float64, len(emb.Values))
			for j, v := range emb.Values {
				float64Values[j] = float64(v)
			}
			openaiResp.Data[i] = openai.Embedding{
				Object:    "embedding",
				Index:     i,
				Embedding: openai.EmbeddingUnion{Value: float64Values},
			}
			if emb.Statistics != nil {
				promptTokens += int(emb.Statistics.TokenCount)
				openaiResp.Data[i].Truncated = emb.Statistics.Truncated
			}
		}
	} else {
		openaiResp.Data = []openai.Embedding{}
	}

	return openaiResp, promptTokens, nil
}

// ResponseError implements [OpenAIEmbeddingTranslator.ResponseError].
// Translate GCP Vertex AI exceptions to OpenAI error type.
// GCP error responses typically contain JSON with error details or plain text error messages.
func (o *openAIToGCPVertexAITranslatorV1Embedding) ResponseError(respHeaders map[string]string, body io.Reader) (
	newHeaders []internalapi.Header, newBody []byte, err error,
) {
	return convertGCPVertexAIErrorToOpenAI(respHeaders, body)
}
