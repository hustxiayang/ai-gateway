// Copyright Envoy AI Gateway Authors
// SPDX-License-Identifier: Apache-2.0
// The full text of the Apache license is available in the LICENSE file at
// the root of the repo.

package translator

import (
	"fmt"
	"strings"

	"github.com/barkimedes/go-deepcopy"
)

// retrieveRef fetches a deeply-nested reference from a schema map.
func retrieveRef(path string, schema map[string]any) (any, error) {
	components := strings.Split(path, "/")
	if components[0] != "#" {
		msg := "ref paths are expected to be URI fragments, meaning they should start with #."
		return nil, fmt.Errorf("invalid JSON schema: %s", msg)
	}

	out := schema
	for _, component := range components[1:] {
		// Type assertion to ensure `out` is a map.
		val, ok := out[component]
		if ok {
			out = val.(map[string]any)
		} else {
			msg := fmt.Sprintf("Reference '%s' not found: intermediate component is not a map", path)
			return nil, fmt.Errorf("invalid JSON schema: %s", msg)
		}
	}

	// Create and return a deep copy to prevent mutation of the original schema.
	deepCopy, err := deepcopy.Anything(out)
	if err != nil {
		return nil, fmt.Errorf("failed to create deep copy: %w", err)
	}
	return deepCopy, nil
}

// dereferenceRefsHelper recursively dereferences JSON schema references.
// Note: `processedRefs` is a pointer to a set (map[string]struct{})
// to ensure state is shared across recursive calls.
func dereferenceRefsHelper(
	obj any,
	fullSchema map[string]any,
	skipKeys []string,
	processedRefs *map[string]struct{},
) (any, error) {
	if *processedRefs == nil {
		*processedRefs = make(map[string]struct{})
	}

	// Handle dictionaries (maps).
	if dict, ok := obj.(map[string]any); ok {
		objOut := make(map[string]any)
		for k, v := range dict {
			// Check if key should be skipped.
			var shouldSkip bool
			for _, skipKey := range skipKeys {
				if k == skipKey {
					shouldSkip = true
					break
				}
			}
			if shouldSkip {
				objOut[k] = v
				continue
			}

			// Handle reference key "$ref".
			if k == "$ref" {
				refPath, isString := v.(string)
				if !isString {
					return nil, fmt.Errorf("'$ref' value must be a string")
				}
				if _, ok := (*processedRefs)[refPath]; ok {
					continue // Already processed, skip to avoid circular references.
				}
				(*processedRefs)[refPath] = struct{}{}

				ref, err := retrieveRef(refPath, fullSchema)
				if err != nil {
					return nil, err
				}
				fullRef, err := dereferenceRefsHelper(ref, fullSchema, skipKeys, processedRefs)
				if err != nil {
					return nil, err
				}
				delete(*processedRefs, refPath) // Remove from set on function exit.
				return fullRef, nil
			}

			// Recurse on nested dictionaries and lists.
			if _, isDict := v.(map[string]any); isDict {
				res, err := dereferenceRefsHelper(v, fullSchema, skipKeys, processedRefs)
				if err != nil {
					return nil, err
				}
				objOut[k] = res
			} else if _, isList := v.([]any); isList {
				res, err := dereferenceRefsHelper(v, fullSchema, skipKeys, processedRefs)
				if err != nil {
					return nil, err
				}
				objOut[k] = res
			} else {
				objOut[k] = v
			}
		}
		return objOut, nil
	}

	// Handle lists (slices).
	if list, ok := obj.([]any); ok {
		listOut := make([]any, len(list))
		for i, el := range list {
			res, err := dereferenceRefsHelper(el, fullSchema, skipKeys, processedRefs)
			if err != nil {
				return nil, err
			}
			listOut[i] = res
		}
		return listOut, nil
	}

	// Return non-dictionary and non-list types as is.
	return obj, nil
}

// modified from https://github.com/langchain-ai/langchain/blob/fce8caca16121024547fb0e8eb2d289c8f96396a/libs/core/langchain_core/utils/json_schema.py#L71
// inferSkipKeys recursively traverses a schema to find keys that should be skipped.
func inferSkipKeys(
	obj any,
	fullSchema map[string]any,
	processedRefs map[string]struct{},
) ([]string, error) {
	// Initialize the processedRefs set on the first call.
	if processedRefs == nil {
		processedRefs = make(map[string]struct{})
	}

	keys := []string{}

	// Handle dictionaries (maps).
	if dict, ok := obj.(map[string]any); ok {
		for k, v := range dict {
			if k == "$ref" {
				refPath, isString := v.(string)
				if !isString {
					return nil, fmt.Errorf("'$ref' value must be a string")
				}
				// Skip if reference has already been processed.
				if _, ok := processedRefs[refPath]; ok {
					continue
				}
				processedRefs[refPath] = struct{}{}

				ref, err := retrieveRef(refPath, fullSchema)
				if err != nil {
					return nil, err
				}

				// Add the top-level key of the reference to the list.
				// This relies on the reference path format, e.g., "#/components/..."
				components := strings.Split(refPath, "/")
				if len(components) > 1 {
					keys = append(keys, components[1])
				}

				// Recurse on the referenced schema.
				nestedKeys, err := inferSkipKeys(ref, fullSchema, processedRefs)
				if err != nil {
					return nil, err
				}
				keys = append(keys, nestedKeys...)
			} else if _, isDict := v.(map[string]any); isDict {
				nestedKeys, err := inferSkipKeys(v, fullSchema, processedRefs)
				if err != nil {
					return nil, err
				}
				keys = append(keys, nestedKeys...)
			} else if _, isList := v.([]any); isList {
				nestedKeys, err := inferSkipKeys(v, fullSchema, processedRefs)
				if err != nil {
					return nil, err
				}
				keys = append(keys, nestedKeys...)
			}
		}
	} else if list, ok := obj.([]any); ok {
		// Handle lists (slices).
		for _, el := range list {
			nestedKeys, err := inferSkipKeys(el, fullSchema, processedRefs)
			if err != nil {
				return nil, err
			}
			keys = append(keys, nestedKeys...)
		}
	}

	return keys, nil
}

// adapted from https://github.com/langchain-ai/langchain/blob/fce8caca16121024547fb0e8eb2d289c8f96396a/libs/core/langchain_core/utils/json_schema.py#L95
// DereferenceRefsOptions holds optional parameters for the DereferenceRefs function.
type DereferenceRefsOptions struct {
	FullSchema map[string]any
	SkipKeys   []string
}

// dereferenceRefs substitutes $refs in a JSON Schema object.
func dereferenceRefs(
	schemaObj map[string]any,
) (any, error) {
	skipKeys, err := inferSkipKeys(schemaObj, schemaObj, nil)
	if err != nil {
		return nil, fmt.Errorf("invalid JSON schema: %w", err)
	}

	// Call the recursive helper function to perform the dereferencing.
	processedRefs := make(map[string]struct{})
	return dereferenceRefsHelper(schemaObj, schemaObj, skipKeys, &processedRefs)
}

// GCP supports only a subset of the jsonSchema https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.cachedContents#Schema
// Define allowed schema fields to replicate the Python behavior.
var allowedSchemaFieldsSet = map[string]struct{}{
	"title":       {},
	"description": {},
	"example":     {},
	"default":     {},
	"enum":        {},
	"items":       {},
	"properties":  {},
	"required":    {},
	"type":        {},
	"nullable":    {},
	"allOf":       {},
	"anyOf":       {},
	"not":         {},
}

// GcpConversionError defines a custom error type for consistent error handling.
type GcpConversionError struct {
	msg       string
	errorType string
}

func (e *GcpConversionError) Error() string {
	return fmt.Sprintf("GcpConversionException: %s, type: %s", e.msg, e.errorType)
}

// formatJSONSchemaToGapic formats a JSON schema for a gapic request.
// Adapted from https://github.com/langchain-ai/langchain-google/blob/06a5857841675461ae282648a155e1cd90d1e5d5/libs/vertexai/langchain_google_vertexai/functions_utils.py#L109
// it ignores the null values, we change to make it compatible.
func formatJSONSchemaToGapic(schema map[string]any) (map[string]any, error) {
	convertedSchema := make(map[string]any)

	for key, value := range schema {
		switch key {
		case "$defs":
			continue
		case "items":
			if subSchema, ok := value.(map[string]any); ok {
				var err error
				convertedSchema["items"], err = formatJSONSchemaToGapic(subSchema)
				if err != nil {
					return nil, err
				}
			}
		case "properties":
			if properties, ok := value.(map[string]any); ok {
				convertedSchema["properties"] = make(map[string]any)
				for pkey, pvalue := range properties {
					if pSubSchema, ok := pvalue.(map[string]any); ok {
						var err error
						convertedSchema["properties"].(map[string]any)[pkey], err = formatJSONSchemaToGapic(pSubSchema)
						if err != nil {
							return nil, err
						}
					}
				}
			}
		case "type":
			if typeList, ok := value.([]any); ok {
				if len(typeList) == 2 {
					hasNull := false
					var nonNullType any
					for _, t := range typeList {
						if t == "null" {
							hasNull = true
						} else {
							nonNullType = t
						}
					}
					if hasNull && nonNullType != nil {
						if nonNullTypeMap, ok := nonNullType.(map[string]any); ok {
							res, err := formatJSONSchemaToGapic(nonNullTypeMap)
							if err != nil {
								return nil, err
							}
							for resKey, resVal := range res {
								convertedSchema[resKey] = resVal
							}
						} else {
							convertedSchema["type"] = fmt.Sprintf("%v", nonNullType)
						}
						convertedSchema["nullable"] = true
					} else {
						return nil, &GcpConversionError{
							msg:       "If type is a list, it must contain one non-null type and 'null'.",
							errorType: "REQUEST_CONVERSION",
						}
					}
				} else {
					return nil, &GcpConversionError{
						msg:       fmt.Sprintf("If the value of type is a list, the length of the list must be 2. Got %d.", len(typeList)),
						errorType: "REQUEST_CONVERSION",
					}
				}
			} else {
				convertedSchema["type"] = fmt.Sprintf("%v", value)
			}
		case "allOf":
			if allOfList, ok := value.([]any); ok {
				if len(allOfList) > 1 {
					return nil, &GcpConversionError{
						msg:       fmt.Sprintf("Only one value for 'allOf' key is supported. Got %d.", len(allOfList)),
						errorType: "REQUEST_CONVERSION",
					}
				}
				if subSchema, ok := allOfList[0].(map[string]any); ok {
					return formatJSONSchemaToGapic(subSchema)
				}
			}
		case "anyOf":
			if anyOfList, ok := value.([]any); ok {
				anyOfResults := make([]any, 0)
				nullable := false
				for _, v := range anyOfList {
					if subSchema, ok := v.(map[string]any); ok {
						if t, exists := subSchema["type"]; exists && t == "null" {
							nullable = true
						} else {
							res, err := formatJSONSchemaToGapic(subSchema)
							if err != nil {
								return nil, err
							}
							anyOfResults = append(anyOfResults, res)
						}
					}
				}
				if nullable {
					convertedSchema["nullable"] = true
				}
				convertedSchema["anyOf"] = anyOfResults
			}
		default:
			// Check if the key is in the allowed set.
			if _, allowed := allowedSchemaFieldsSet[key]; allowed {
				convertedSchema[key] = value
			}
			// Silently ignore other keys, mirroring the Python behavior.
		}
	}
	return convertedSchema, nil
}

// sanitizeJSONSchema first dereferences the schema and then formats it for GCP.
func sanitizeJSONSchema(schema map[string]any) (any, error) {
	// Step 1: dereference the JSON schema.
	dereferencedSchema, err := dereferenceRefs(schema)
	if err != nil {
		return nil, err
	}

	// Step 2: Assert that the dereferenced result is a map.
	dereferencedMap, ok := dereferencedSchema.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("dereferenced schema was not a map[string]any")
	}

	// Step 3: Format the dereferenced schema for GCP.
	return formatJSONSchemaToGapic(dereferencedMap)
}
