// Copyright Envoy AI Gateway Authors
// SPDX-License-Identifier: Apache-2.0
// The full text of the Apache license is available in the LICENSE file at
// the root of the repo.

package internalapi

import "errors"

// User-facing errors that are safe to return in HTTP responses.
// These errors contain no sensitive information and can be directly
// exposed to clients with appropriate HTTP status codes.
var (
	// ErrInvalidRequestBody indicates the request body is malformed JSON
	// or doesn't match the expected schema.
	ErrInvalidRequestBody = errors.New("invalid request body format")

	// ErrMissingRequiredField indicates a required field is missing from the request.
	ErrMissingRequiredField = errors.New("missing required field in request")

	// ErrInvalidModelSchema indicates the request doesn't match the
	// expected schema for the target model or API.
	ErrInvalidModelSchema = errors.New("request schema incompatible with target API")
)
