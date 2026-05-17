// Package ccconnect provides a unified interface for connecting to various
// AI coding assistants and language model APIs.
package ccconnect

import (
	"context"
	"errors"
	"time"
)

// ErrNotConnected is returned when an operation is attempted on a closed or
// uninitialized connector.
var ErrNotConnected = errors.New("ccconnect: not connected")

// ErrUnsupportedProvider is returned when an unsupported provider is specified.
var ErrUnsupportedProvider = errors.New("ccconnect: unsupported provider")

// Provider represents a supported AI/LLM provider.
type Provider string

const (
	ProviderOpenAI    Provider = "openai"
	ProviderAnthropic Provider = "anthropic"
	ProviderGemini    Provider = "gemini"
	ProviderOllama    Provider = "ollama"
)

// Message represents a single message in a conversation.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Request encapsulates a completion request to an AI provider.
type Request struct {
	Messages    []Message         `json:"messages"`
	Model       string            `json:"model"`
	MaxTokens   int               `json:"max_tokens,omitempty"`
	Temperature float64           `json:"temperature,omitempty"`
	Metadata    map[string]string `json:"metadata,omitempty"`
}

// Response encapsulates the response from an AI provider.
type Response struct {
	Content      string    `json:"content"`
	Model        string    `json:"model"`
	Provider     Provider  `json:"provider"`
	InputTokens  int       `json:"input_tokens"`
	OutputTokens int       `json:"output_tokens"`
	CreatedAt    time.Time `json:"created_at"`
}

// Connector defines the interface that all provider implementations must satisfy.
type Connector interface {
	// Connect initializes the connection to the provider.
	Connect(ctx context.Context) error

	// Complete sends a completion request and returns the response.
	Complete(ctx context.Context, req *Request) (*Response, error)

	// Stream sends a completion request and streams tokens via the returned channel.
	Stream(ctx context.Context, req *Request) (<-chan string, <-chan error)

	// Provider returns the provider identifier.
	Provider() Provider

	// Close releases any resources held by the connector.
	Close() error
}

// Config holds common configuration for all connectors.
type Config struct {
	APIKey     string
	BaseURL    string
	Timeout    time.Duration
	MaxRetries int
	Debug      bool
}

// DefaultConfig returns a Config with sensible defaults.
func DefaultConfig() *Config {
	return &Config{
		Timeout:    30 * time.Second,
		MaxRetries: 3,
		Debug:      false,
	}
}

// New creates a new Connector for the specified provider using the given config.
func New(provider Provider, cfg *Config) (Connector, error) {
	if cfg == nil {
		cfg = DefaultConfig()
	}

	switch provider {
	case ProviderOpenAI:
		return newOpenAIConnector(cfg), nil
	case ProviderAnthropic:
		return newAnthropicConnector(cfg), nil
	case ProviderGemini:
		return newGeminiConnector(cfg), nil
	case ProviderOllama:
		return newOllamaConnector(cfg), nil
	default:
		return nil, ErrUnsupportedProvider
	}
}
