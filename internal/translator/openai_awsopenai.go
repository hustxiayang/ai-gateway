// Copyright Envoy AI Gateway Authors
// SPDX-License-Identifier: Apache-2.0
// The full text of the Apache license is available in the LICENSE file at
// the root of the repo.

package translator

import (
	"fmt"
	"strconv"

	"github.com/envoyproxy/ai-gateway/internal/apischema/openai"
	"github.com/envoyproxy/ai-gateway/internal/internalapi"
)

// NewChatCompletionOpenAIToAwsOpenAITranslator implements [Factory] for OpenAI to Aws OpenAI translations.
func NewChatCompletionOpenAIToAwsOpenAITranslator(apiVersion string, modelNameOverride internalapi.ModelNameOverride) OpenAIChatCompletionTranslator {
	return &openAIToAwsOpenAITranslatorV1ChatCompletion{
		apiVersion: apiVersion,
		openAIToOpenAITranslatorV1ChatCompletion: openAIToOpenAITranslatorV1ChatCompletion{
			modelNameOverride: modelNameOverride,
		},
	}
}

// openAIToAwsOpenAITranslatorV1ChatCompletion adapts OpenAI requests for Aws OpenAI Service.
// Azure ignores the model field in the request body, using deployment name from the URI path instead:
// https://learn.microsoft.com/en-us/azure/ai-foundry/openai/reference#chat-completions
type openAIToAwsOpenAITranslatorV1ChatCompletion struct {
	openAIToOpenAITranslatorV1ChatCompletion
}

func (o *openAIToAwsOpenAITranslatorV1ChatCompletion) RequestBody(raw []byte, req *openai.ChatCompletionRequest, forceBodyMutation bool) (
	newHeaders []internalapi.Header, newBody []byte, err error,
) {
	modelName := req.Model
	if o.modelNameOverride != "" {
		// If modelName is set we override the model to be used for the request.
		modelName = o.modelNameOverride
	}
	// Ensure the response includes a model. This is set to accommodate test or
	// misimplemented backends.
	o.requestModel = modelName

	// Azure OpenAI uses a {deployment-id} that may match the deployed model's name.
	// We use the routed model as the deployment, stored in the path.
	pathTemplate := "/openai/deployments/%s/chat/completions?api-version=%s"
	newHeaders = []internalapi.Header{{pathHeaderName, fmt.Sprintf(pathTemplate, modelName, o.apiVersion)}}
	if req.Stream {
		o.stream = true
	}

	// On retry, the path might have changed to a different provider. So, this will enesure that the path is always set to OpenAI.
	if forceBodyMutation {
		newHeaders = append(newHeaders, internalapi.Header{contentLengthHeaderName, strconv.Itoa(len(raw))})
	}
	return
}
