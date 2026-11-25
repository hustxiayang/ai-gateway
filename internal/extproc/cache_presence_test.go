// Copyright Envoy AI Gateway Authors
// SPDX-License-Identifier: Apache-2.0
// The full text of the Apache license is available in the LICENSE file at
// the root of the repo.

package extproc

import (
	"fmt"
	"testing"

	extprocv3 "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"

	anthropicschema "github.com/envoyproxy/ai-gateway/internal/apischema/anthropic"
	"github.com/envoyproxy/ai-gateway/internal/filterapi"
	"github.com/envoyproxy/ai-gateway/internal/internalapi"
	"github.com/envoyproxy/ai-gateway/internal/translator"
)

func TestCacheDetailsPreservesProviderPresence(t *testing.T) {
	for _, tc := range []struct {
		name         string
		fields       string
		present      bool
		readOnly     bool
		creationOnly bool
	}{
		{name: "omitted"},
		{name: "explicit zero", fields: `,"cache_read_input_tokens":0,"cache_creation_input_tokens":0`, present: true},
		{name: "positive", fields: `,"cache_read_input_tokens":3,"cache_creation_input_tokens":2`, present: true},
		{name: "read-only zero", fields: `,"cache_read_input_tokens":0`, present: true, readOnly: true},
		{name: "creation-only zero", fields: `,"cache_creation_input_tokens":0`, present: true, creationOnly: true},
		{name: "null is omitted", fields: `,"cache_read_input_tokens":null,"cache_creation_input_tokens":null`},
	} {
		for _, stream := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/stream=%t", tc.name, stream), func(t *testing.T) {
				tr := translator.NewAnthropicToAnthropicTranslator("", "")
				_, _, err := tr.RequestBody(nil, &anthropicschema.MessagesRequest{Model: "fixture", Stream: stream}, false)
				require.NoError(t, err)
				processor := &messagesProcessorUpstreamFilter{
					translator:      tr,
					metrics:         &mockMetrics{},
					requestHeaders:  map[string]string{},
					responseHeaders: map[string]string{":status": "200"},
					parent: &messagesProcessorRouterFilter{stream: stream, config: &filterapi.RuntimeConfig{GlobalRequestCosts: []filterapi.RuntimeGlobalRequestCost{{
						GlobalLLMRequestCost: &filterapi.GlobalLLMRequestCost{Type: filterapi.LLMRequestCostTypeOutputToken, MetadataKey: "output_cost"},
					}}}},
				}
				usage := `{"input_tokens":7,"output_tokens":1` + tc.fields + `}`
				message := `{"id":"fixture","type":"message","role":"assistant","model":"claude-sonnet-4-5","content":[],"usage":` + usage + `}`
				if stream {
					message = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":" + message + "}\n\nevent: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
				}
				response, err := processor.ProcessResponseBody(t.Context(), &extprocv3.HttpBody{Body: []byte(message), EndOfStream: true})
				require.NoError(t, err)
				metadata := response.GetDynamicMetadata().GetFields()[internalapi.AIGatewayFilterMetadataNamespace].GetStructValue()
				details := metadata.GetFields()["cache_details"]
				if !tc.present {
					require.Nil(t, details, "provider-omitted counts are not inferred zeros")
				} else {
					require.NotNil(t, details)
					if tc.creationOnly {
						require.NotContains(t, details.GetStructValue().GetFields(), "cache_read_input_tokens")
					} else {
						require.Contains(t, details.GetStructValue().GetFields(), "cache_read_input_tokens")
					}
					if tc.readOnly {
						require.NotContains(t, details.GetStructValue().GetFields(), "cache_creation_input_tokens")
					} else {
						require.Contains(t, details.GetStructValue().GetFields(), "cache_creation_input_tokens")
					}
				}
			})
		}
	}
}
