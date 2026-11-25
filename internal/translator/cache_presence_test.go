// Copyright Envoy AI Gateway Authors
// SPDX-License-Identifier: Apache-2.0
// The full text of the Apache license is available in the LICENSE file at
// the root of the repo.

package translator

import (
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/stretchr/testify/require"

	"github.com/envoyproxy/ai-gateway/internal/json"
)

func TestAnthropicSDKCacheAggregatePresence(t *testing.T) {
	for _, tc := range []struct {
		name    string
		data    string
		present bool
	}{
		{name: "omitted", data: `{"input_tokens":7,"output_tokens":1}`},
		{name: "explicit zero", data: `{"input_tokens":7,"output_tokens":1,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}`, present: true},
		{name: "positive", data: `{"input_tokens":7,"output_tokens":1,"cache_read_input_tokens":3,"cache_creation_input_tokens":2}`, present: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var sdkUsage anthropic.Usage
			require.NoError(t, json.Unmarshal([]byte(tc.data), &sdkUsage))
			usage := extractAnthropicSDKTokenUsage(&sdkUsage)
			_, read := usage.CachedInputTokens()
			_, creation := usage.CacheCreationInputTokens()
			require.Equal(t, tc.present, read)
			require.Equal(t, tc.present, creation)
		})
	}
}
