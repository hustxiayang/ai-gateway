// Copyright Envoy AI Gateway Authors
// SPDX-License-Identifier: Apache-2.0
// The full text of the Apache license is available in the LICENSE file at
// the root of the repo.

package internalapi

import (
	"errors"
)

// User-facing errors that are safe to return in HTTP responses.
// These errors contain no sensitive information and can be directly
// exposed to clients with appropriate HTTP status codes.
//
// Usage: Use fmt.Errorf("%w: %s", ErrInvalidRequestBody, "specific details")
// to wrap these errors with additional context.
var (
	// ErrInvalidRequestBody indicates the request body is malformed JSON,
	// doesn't match the expected schema, or contains invalid values.
	ErrInvalidRequestBody = errors.New("invalid request body")
)

// GetUserFacingError checks if an error is a known user-facing error that's safe to expose.
// Returns the unwrapped user-facing error if it's safe, or nil if it should not be exposed to users.
func GetUserFacingError(err error) error {
	if errors.Is(err, ErrInvalidRequestBody) {
		return err
	}
	return nil
}
