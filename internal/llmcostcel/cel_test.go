// Copyright Envoy AI Gateway Authors
// SPDX-License-Identifier: Apache-2.0
// The full text of the Apache license is available in the LICENSE file at
// the root of the repo.

package llmcostcel

import (
	"testing"
	"testing/synctest"

	"github.com/stretchr/testify/require"
)

func TestNewProgram(t *testing.T) {
	t.Run("invalid", func(t *testing.T) {
		_, err := NewProgram("1 +")
		require.Error(t, err)
	})
	t.Run("int", func(t *testing.T) {
		_, err := NewProgram("1 + 1")
		require.NoError(t, err)
	})
	t.Run("uint", func(t *testing.T) {
		_, err := NewProgram("uint(1) + uint(1)")
		require.NoError(t, err)
	})
	t.Run("variables", func(t *testing.T) {
		prog, err := NewProgram("model == 'cool_model' ?  (input_tokens - cached_input_tokens - cache_creation_input_tokens) * output_tokens  : total_tokens")
		require.NoError(t, err)
		v, err := EvaluateProgram(prog, "cool_model", "cool_backend", "cool_route", 200, 100, 1, 2, 3, 0)
		require.NoError(t, err)
		require.Equal(t, uint64(198), v)

		v, err = EvaluateProgram(prog, "not_cool_model", "cool_backend", "cool_route", 200, 100, 1, 2, 3, 0)
		require.NoError(t, err)
		require.Equal(t, uint64(3), v)
	})

	t.Run("uint", func(t *testing.T) {
		_, err := NewProgram("uint(1)-uint(1200)")
		require.ErrorContains(t, err, "failed to evaluate CEL expression: failed to evaluate CEL expression: unsigned integer overflow")
	})

	t.Run("ensure concurrency safety", func(t *testing.T) {
		// Ensure that the program can be evaluated concurrently.
		synctest.Test(t, func(t *testing.T) {
			for range 100 {
				go func() {
					_, err := NewProgram("model == 'cool_model' ?  input_tokens * output_tokens : total_tokens")
					require.NoError(t, err)
				}()
			}
		}) // synctest.Test waits for all goroutines to complete.
	})
}

func TestEvaluateProgram(t *testing.T) {
	t.Run("signed integer negative", func(t *testing.T) {
		prog, err := NewProgram("int(input_tokens) - int(output_tokens)")
		require.NoError(t, err)
		_, err = EvaluateProgram(prog, "cool_model", "cool_backend", "cool_route", 100, 0, 0, 2000, 3, 0)
		require.ErrorContains(t, err, "CEL expression result is negative (-1900)")
	})
	t.Run("unsigned integer overflow", func(t *testing.T) {
		prog, err := NewProgram("input_tokens - output_tokens")
		require.NoError(t, err)
		_, err = EvaluateProgram(prog, "cool_model", "cool_backend", "cool_route", 100, 0, 0, 2000, 3, 0)
		require.ErrorContains(t, err, "failed to evaluate CEL expression: unsigned integer overflow")
	})
	t.Run("reasoning_tokens variable", func(t *testing.T) {
		prog, err := NewProgram("output_tokens + reasoning_tokens")
		require.NoError(t, err)
		v, err := EvaluateProgram(prog, "cool_model", "cool_backend", "cool_route", 0, 0, 0, 100, 0, 50)
		require.NoError(t, err)
		require.Equal(t, uint64(150), v)
	})
	t.Run("ensure concurrency safety", func(t *testing.T) {
		prog, err := NewProgram("model == 'cool_model' ?  input_tokens * output_tokens : total_tokens")
		require.NoError(t, err)

		// Ensure that the program can be evaluated concurrently.
		synctest.Test(t, func(t *testing.T) {
			for range 100 {
				go func() {
					v, err := EvaluateProgram(prog, "cool_model", "cool_backend", "cool_route", 100, 0, 0, 2, 3, 0)
					require.NoError(t, err)
					require.Equal(t, uint64(200), v)
				}()
			}
		}) // synctest.Test waits for all goroutines to complete.
	})
}

func TestLongContextPricingExpression(t *testing.T) {
	expr := "uint(input_tokens > uint(272000) ? " +
		"(double(input_tokens - cached_input_tokens - cache_creation_input_tokens) * double(2) + " +
		"double(cached_input_tokens) * double(0.2) + " +
		"double(cache_creation_input_tokens) * double(2.5) + " +
		"double(output_tokens) * double(3)) : " +
		"(double(input_tokens - cached_input_tokens - cache_creation_input_tokens) * double(1) + " +
		"double(cached_input_tokens) * double(0.1) + " +
		"double(cache_creation_input_tokens) * double(1.25) + " +
		"double(output_tokens) * double(2)))"

	prog, err := NewProgram(expr)
	require.NoError(t, err)

	tests := []struct {
		name        string
		inputTokens uint32
		want        uint64
	}{
		{name: "below threshold", inputTokens: 271_999, want: 187_199},
		{name: "exactly at threshold", inputTokens: 272_000, want: 187_200},
		{name: "above threshold", inputTokens: 272_001, want: 374_302},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, evalErr := EvaluateProgram(
				prog,
				"gpt-long-context",
				"openai",
				"gpt-long-context",
				tt.inputTokens,
				100_000,
				20_000,
				100,
				0,
				0,
			)
			require.NoError(t, evalErr)
			require.Equal(t, tt.want, got)
		})
	}
}
