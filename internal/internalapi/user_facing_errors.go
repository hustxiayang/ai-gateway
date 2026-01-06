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
var (
	// ErrInvalidRequestBody indicates the request body is malformed JSON
	// or doesn't match the expected schema.
	ErrInvalidRequestBody = errors.New("invalid request body format")

	// ErrMissingRequiredField indicates a required field is missing from the request.
	ErrMissingRequiredField = errors.New("missing required field in request")

	// ErrInvalidModelSchema indicates the request doesn't match the
	// expected schema for the target model or API.
	ErrInvalidModelSchema = errors.New("request schema incompatible with target API")

	// ErrUnsupportedToolFormat indicates the tools or tool_choice format is not supported
	// by the target API.
	ErrUnsupportedToolFormat = errors.New("tool calling format not supported by target API")

	// ErrUnsupportedMessageFormat indicates the message format or content type is not supported
	// by the target API (e.g., image formats, content types).
	ErrUnsupportedMessageFormat = errors.New("message format not supported by target API")

	// ErrInvalidParameterValue indicates a parameter value is outside the valid range
	// for the target API (e.g., temperature, top_p).
	ErrInvalidParameterValue = errors.New("parameter value out of valid range for target API")

	// ErrUnsupportedFeature indicates the request uses a feature not supported by the target API
	// (e.g., streaming, function calling, specific response formats).
	ErrUnsupportedFeature = errors.New("requested feature not supported by target API")
)

// GetUserFacingError checks if an error is a known user-facing error that's safe to expose.
// Returns the user-facing error if it's safe, or nil if it should not be exposed to users.
func GetUserFacingError(err error) error {
	switch {
	case errors.Is(err, ErrInvalidRequestBody):
		return ErrInvalidRequestBody
	case errors.Is(err, ErrMissingRequiredField):
		return ErrMissingRequiredField
	case errors.Is(err, ErrInvalidModelSchema):
		return ErrInvalidModelSchema
	case errors.Is(err, ErrUnsupportedToolFormat):
		return ErrUnsupportedToolFormat
	case errors.Is(err, ErrUnsupportedMessageFormat):
		return ErrUnsupportedMessageFormat
	case errors.Is(err, ErrInvalidParameterValue):
		return ErrInvalidParameterValue
	case errors.Is(err, ErrUnsupportedFeature):
		return ErrUnsupportedFeature
	default:
		return nil
	}
}
