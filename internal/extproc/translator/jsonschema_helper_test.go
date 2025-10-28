// Copyright Envoy AI Gateway Authors
// SPDX-License-Identifier: Apache-2.0
// The full text of the Apache license is available in the LICENSE file at
// the root of the repo.

package translator

import (
	"encoding/json"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/stretchr/testify/require"
	"google.golang.org/genai"
)

func TestJsonSchemaToGemini(t *testing.T) {
	trueBool := true
	tests := []struct {
		name                   string
		input                  json.RawMessage
		expectedResponseSchema *genai.Schema
	}{
		{
			name: "nested schema for ResponseSchema",
			input: json.RawMessage(`{
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "$ref": "#/$defs/step"
            }
        },
        "final_answer": {
            "type": "string"
        }
    },
    "$defs": {
        "step": {
            "type": "object",
            "properties": {
                "explanation": {
                    "type": "string"
                },
                "output": {
                    "type": "string"
                }
            },
            "required": [
                "explanation",
                "output"
            ],
            "additionalProperties": false
        }
    },
    "required": [
        "steps",
        "final_answer"
    ],
    "additionalProperties": false
}`),

			expectedResponseSchema: &genai.Schema{
				Properties: map[string]*genai.Schema{
					"final_answer": {Type: "string"},
					"steps": {
						Items: &genai.Schema{
							Properties: map[string]*genai.Schema{
								"explanation": {Type: "string"},
								"output":      {Type: "string"},
							},
							Type:     "object",
							Required: []string{"explanation", "output"},
						},
						Type: "array",
					},
				},
				Type:     "object",
				Required: []string{"steps", "final_answer"},
			},
		},

		{
			name: "anyof list for ResponseSchema",
			input: json.RawMessage(`{
    "type": "object",
    "properties": {
        "item": {
            "anyOf": [
                {
                    "type": "object",
                    "description": "The user object to insert into the database",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The name of the user"
                        },
                        "age": {
                            "type": "number",
                            "description": "The age of the user"
                        }
                    },
                    "additionalProperties": false,
                    "required": [
                        "name",
                        "age"
                    ]
                },
                {
                    "type": "object",
                    "description": "The address object to insert into the database",
                    "properties": {
                        "number": {
                            "type": "string",
                            "description": "The number of the address. Eg. for 123 main st, this would be 123"
                        },
                        "street": {
                            "type": "string",
                            "description": "The street name. Eg. for 123 main st, this would be main st"
                        },
                        "city": {
                            "type": "string",
                            "description": "The city of the address"
                        }
                    },
                    "additionalProperties": false,
                    "required": [
                        "number",
                        "street",
                        "city"
                    ]
                },
                {
                    "type": "object",
                    "description": "The email address object to insert into the database",
                    "properties": {
                        "company": {
                            "type": "string",
                            "description": "The company to use."
                        },
                        "url": {
                            "type": "string",
                            "description": "The email address"
                        }
                    },
                    "additionalProperties": false,
                    "required": [
                        "company",
                        "url"
                    ]
                }
            ]
        }
    },
    "additionalProperties": false,
    "required": [
        "item"
    ]
}`),

			expectedResponseSchema: &genai.Schema{
				Properties: map[string]*genai.Schema{
					"item": {
						AnyOf: []*genai.Schema{
							{
								Description: "The user object to insert into the database",
								Properties: map[string]*genai.Schema{
									"age":  {Type: "number", Description: "The age of the user"},
									"name": {Type: "string", Description: "The name of the user"},
								},
								Type:     "object",
								Required: []string{"name", "age"},
							},
							{
								Description: "The address object to insert into the database",
								Properties: map[string]*genai.Schema{
									"city":   {Type: "string", Description: "The city of the address"},
									"number": {Type: "string", Description: "The number of the address. Eg. for 123 main st, this would be 123"},
									"street": {Type: "string", Description: "The street name. Eg. for 123 main st, this would be main st"},
								},
								Type:     "object",
								Required: []string{"number", "street", "city"},
							},
							{
								Description: "The email address object to insert into the database",
								Properties: map[string]*genai.Schema{
									"company": {Type: "string", Description: "The company to use."},
									"url":     {Type: "string", Description: "The email address"},
								},
								Required: []string{"company", "url"},
								Type:     "object",
							},
						},
					},
				},
				Type:     "object",
				Required: []string{"item"},
			},
		},

		{
			name: "anyof null for ResponseSchema",
			input: json.RawMessage(`{
    "type": "object",
    "description": "Data model identifying a single paragraph for paragraph re-ranking.",
    "properties": {
        "paragraph_id": {
            "anyOf": [
                {
                    "type": "string"
                },
                {
                    "type": "null"
                }
            ],
            "title": "Paragraph Id"
        },
        "document_id": {
            "title": "Document Id",
            "type": "string"
        }
    },
    "required": [
        "paragraph_id",
        "document_id"
    ],
    "title": "ParagraphIdentifier",
    "additionalProperties": false
}`),

			expectedResponseSchema: &genai.Schema{
				Description: "Data model identifying a single paragraph for paragraph re-ranking.",
				Properties: map[string]*genai.Schema{
					"document_id": {Type: "string", Title: "Document Id"},
					"paragraph_id": {
						AnyOf: []*genai.Schema{
							{
								Type: "string",
							},
						},
						Nullable: &trueBool,
						Title:    "Paragraph Id",
					},
				},
				Required: []string{"paragraph_id", "document_id"},
				Title:    "ParagraphIdentifier",
				Type:     "object",
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var schemaMap map[string]any
			err := json.Unmarshal([]byte(tc.input), &schemaMap)
			require.NoError(t, err)

			got, err := jsonSchemaToGemini(schemaMap)

			require.NoError(t, err)

			if diff := cmp.Diff(tc.expectedResponseSchema, got, cmpopts.IgnoreUnexported(genai.Schema{})); diff != "" {
				t.Errorf("ResponseSchema mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
