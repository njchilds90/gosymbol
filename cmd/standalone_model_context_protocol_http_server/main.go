package main

import (
	"encoding/json"
	"io"
	"log"
	"net/http"
	"os"

	gosymbol "github.com/njchilds90/gosymbol"
)

func main() {
	address := ":8081"
	if configuredAddress := os.Getenv("GOSYMBOL_STANDALONE_SERVER_ADDRESS"); configuredAddress != "" {
		address = configuredAddress
	}
	multiplexer := http.NewServeMux()
	multiplexer.HandleFunc("/healthz", func(responseWriter http.ResponseWriter, request *http.Request) {
		responseWriter.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(responseWriter).Encode(map[string]string{"status": "ok"})
	})
	multiplexer.HandleFunc("/tools", func(responseWriter http.ResponseWriter, request *http.Request) {
		responseWriter.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(responseWriter).Encode(gosymbol.MCPToolSpec())
	})
	multiplexer.HandleFunc("/tool", func(responseWriter http.ResponseWriter, request *http.Request) {
		if request.Method != http.MethodPost {
			http.Error(responseWriter, "only POST is supported", http.StatusMethodNotAllowed)
			return
		}
		defer request.Body.Close()
		request.Body = http.MaxBytesReader(responseWriter, request.Body, 1<<20)
		var toolRequest gosymbol.ToolRequest
		if decodeError := json.NewDecoder(request.Body).Decode(&toolRequest); decodeError != nil {
			http.Error(responseWriter, decodeError.Error(), http.StatusBadRequest)
			return
		}
		if _, readError := io.Copy(io.Discard, request.Body); readError != nil {
			http.Error(responseWriter, readError.Error(), http.StatusBadRequest)
			return
		}
		responseWriter.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(responseWriter).Encode(gosymbol.HandleToolCall(toolRequest))
	})
	log.Printf("standalone model context protocol server listening on %s", address)
	log.Fatal(http.ListenAndServe(address, multiplexer))
}
