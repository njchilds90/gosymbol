package main

import (
	"encoding/json"
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
	http.HandleFunc("/tool", func(responseWriter http.ResponseWriter, request *http.Request) {
		if request.Method != http.MethodPost {
			http.Error(responseWriter, "only POST is supported", http.StatusMethodNotAllowed)
			return
		}
		defer request.Body.Close()
		var toolRequest gosymbol.ToolRequest
		if decodeError := json.NewDecoder(request.Body).Decode(&toolRequest); decodeError != nil {
			http.Error(responseWriter, decodeError.Error(), http.StatusBadRequest)
			return
		}
		responseWriter.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(responseWriter).Encode(gosymbol.HandleToolCall(toolRequest))
	})
	log.Printf("standalone model context protocol server listening on %s", address)
	log.Fatal(http.ListenAndServe(address, nil))
}
