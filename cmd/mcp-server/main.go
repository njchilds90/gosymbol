// cmd/mcp-server/main.go — Standalone HTTP MCP server for gosymbol
//
// Exposes gosymbol tools as an HTTP endpoint for AI agent frameworks.
//
// Usage:
//   go run cmd/mcp-server/main.go -port 8080
//
// Tool call endpoint: POST /tool
// Schema endpoint:    GET  /schema
// Health endpoint:    GET  /health
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"runtime/debug"
	"time"

	gosymbol "github.com/njchilds90/gosymbol"
)

const maxBodyBytes = 1 << 20 // 1 MiB

func main() {
	port := flag.Int("port", 8080, "Port to listen on")
	flag.Parse()

	mux := http.NewServeMux()

	// POST /tool — handle a tool call
	mux.HandleFunc("/tool", func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if rec := recover(); rec != nil {
				log.Printf("panic in /tool: %v\n%s", rec, string(debug.Stack()))
				http.Error(w, "internal server error", http.StatusInternalServerError)
			}
		}()

		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		r.Body = http.MaxBytesReader(w, r.Body, maxBodyBytes)
		defer r.Body.Close()

		dec := json.NewDecoder(r.Body)
		dec.DisallowUnknownFields()

		var req gosymbol.ToolRequest
		if err := dec.Decode(&req); err != nil {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			_ = json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
			return
		}
		// Ensure there's no trailing junk.
		if dec.More() {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			_ = json.NewEncoder(w).Encode(map[string]string{"error": "invalid JSON: trailing data"})
			return
		}

		resp := gosymbol.HandleToolCall(req)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	})

	// GET /schema — return tool schema for agent registration
	mux.HandleFunc("/schema", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, gosymbol.MCPToolSpec())
	})

	// GET /health — liveness check
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"status": "ok",
			"time":   time.Now().UTC().Format(time.RFC3339),
		})
	})

	addr := fmt.Sprintf(":%d", *port)
	log.Printf("gosymbol MCP server listening on %s", addr)
	log.Printf("  POST /tool   — execute a tool call")
	log.Printf("  GET  /schema — tool schema for agent registration")
	log.Printf("  GET  /health — health check")

	srv := &http.Server{
		Addr:              addr,
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       15 * time.Second,
		WriteTimeout:      15 * time.Second,
		IdleTimeout:       60 * time.Second,
	}

	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatal(err)
	}
}