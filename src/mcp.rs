//! MCP (Model Context Protocol) Server Implementation
//!
//! This module implements the JSON-RPC over stdio protocol for MCP.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{self, BufRead, BufReader, Write};
use std::panic;
use tracing::{debug, error, info};

/// JSON-RPC Request
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: Option<Value>,
    pub method: String,
    #[serde(default)]
    pub params: Option<Value>,
}

/// JSON-RPC Response
#[derive(Debug, Serialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

/// JSON-RPC Error
#[derive(Debug, Serialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl JsonRpcResponse {
    pub fn success(id: Option<Value>, result: Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(result),
            error: None,
        }
    }

    pub fn error(id: Option<Value>, code: i32, message: String) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message,
                data: None,
            }),
        }
    }
}

/// MCP Tool Definition
#[derive(Debug, Clone, Serialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    #[serde(rename = "inputSchema")]
    pub input_schema: Value,
}

/// Tool handler function type
pub type ToolHandler = Box<dyn Fn(Value) -> Result<Value> + Send + Sync>;

/// MCP Server
pub struct McpServer {
    name: String,
    version: String,
    tools: Vec<Tool>,
    handlers: HashMap<String, ToolHandler>,
}

impl McpServer {
    /// Create a new MCP server
    pub fn new(name: &str, version: &str) -> Self {
        Self {
            name: name.to_string(),
            version: version.to_string(),
            tools: Vec::new(),
            handlers: HashMap::new(),
        }
    }

    /// Register a tool
    pub fn register_tool<F>(&mut self, tool: Tool, handler: F)
    where
        F: Fn(Value) -> Result<Value> + Send + Sync + 'static,
    {
        self.tools.push(tool.clone());
        self.handlers.insert(tool.name, Box::new(handler));
    }

    /// Handle an incoming request with panic recovery.
    /// If a handler panics, the server stays alive and returns an error response.
    fn handle_request_safe(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        match panic::catch_unwind(panic::AssertUnwindSafe(|| {
            self.handle_request(request)
        })) {
            Ok(response) => response,
            Err(panic_info) => {
                let msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                    s.to_string()
                } else if let Some(s) = panic_info.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "unknown panic".to_string()
                };
                error!("Handler panicked: {}", msg);
                JsonRpcResponse::error(
                    request.id.clone(),
                    -32603,
                    format!("Internal error: handler panicked: {}", msg),
                )
            }
        }
    }

    /// Handle an incoming request
    pub fn handle_request(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        debug!("Handling request: {}", request.method);

        match request.method.as_str() {
            "initialize" => self.handle_initialize(request),
            "initialized" | "notifications/initialized" => {
                // Notification, no response needed
                JsonRpcResponse::success(request.id.clone(), json!({}))
            }
            "tools/list" => self.handle_list_tools(request),
            "tools/call" => self.handle_call_tool(request),
            "resources/list" => self.handle_list_resources(request),
            "resources/templates/list" => self.handle_list_resource_templates(request),
            "ping" => JsonRpcResponse::success(request.id.clone(), json!({})),
            _ => {
                if request.id.is_none() {
                    // Ignore unknown notifications (no response should be sent).
                    debug!("Ignoring unknown notification: {}", request.method);
                    JsonRpcResponse::success(None, json!({}))
                } else {
                    error!("Unknown method: {}", request.method);
                    JsonRpcResponse::error(
                        request.id.clone(),
                        -32601,
                        format!("Method not found: {}", request.method),
                    )
                }
            }
        }
    }

    fn handle_initialize(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        info!("Initializing MCP server: {} v{}", self.name, self.version);

        let protocol_version = request
            .params
            .as_ref()
            .and_then(|params| params.get("protocolVersion"))
            .and_then(|value| value.as_str())
            .unwrap_or("2024-11-05");

        let result = json!({
            "protocolVersion": protocol_version,
            "capabilities": {
                "tools": {}
            },
            "serverInfo": {
                "name": self.name,
                "version": self.version
            }
        });

        JsonRpcResponse::success(request.id.clone(), result)
    }

    fn handle_list_tools(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        debug!("Listing {} tools", self.tools.len());
        
        let tools: Vec<Value> = self.tools.iter().map(|t| {
            json!({
                "name": t.name,
                "description": t.description,
                "inputSchema": t.input_schema
            })
        }).collect();

        JsonRpcResponse::success(request.id.clone(), json!({ "tools": tools }))
    }

    fn handle_list_resources(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        debug!("Listing resources (none available)");
        JsonRpcResponse::success(request.id.clone(), json!({ "resources": [] }))
    }

    fn handle_list_resource_templates(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        debug!("Listing resource templates (none available)");
        JsonRpcResponse::success(request.id.clone(), json!({ "resourceTemplates": [] }))
    }

    fn handle_call_tool(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        let params = match &request.params {
            Some(p) => p,
            None => {
                return JsonRpcResponse::error(
                    request.id.clone(),
                    -32602,
                    "Missing params".to_string(),
                );
            }
        };

        let tool_name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
        let arguments = params.get("arguments").cloned().unwrap_or(json!({}));

        debug!("Calling tool: {} with args: {:?}", tool_name, arguments);

        match self.handlers.get(tool_name) {
            Some(handler) => match handler(arguments) {
                Ok(result) => {
                    let content = json!([{
                        "type": "text",
                        "text": serde_json::to_string(&result).unwrap_or_else(|_| result.to_string())
                    }]);
                    JsonRpcResponse::success(request.id.clone(), json!({ "content": content }))
                }
                Err(e) => {
                    error!("Tool error: {}", e);
                    let content = json!([{
                        "type": "text",
                        "text": json!({ "error": e.to_string() }).to_string()
                    }]);
                    JsonRpcResponse::success(request.id.clone(), json!({ "content": content, "isError": true }))
                }
            },
            None => {
                error!("Tool not found: {}", tool_name);
                JsonRpcResponse::error(
                    request.id.clone(),
                    -32602,
                    format!("Tool not found: {}", tool_name),
                )
            }
        }
    }

    /// Run the server on stdio
    pub fn run(&self) -> Result<()> {
        info!("Starting MCP server on stdio");

        // Use 64MB buffer to handle large payloads
        let stdin = io::stdin();
        let reader = BufReader::with_capacity(64 * 1024 * 1024, stdin.lock());
        let mut stdout = io::stdout();

        for line in reader.lines() {
            let line = line.context("Failed to read line")?;

            if line.trim().is_empty() {
                continue;
            }

            debug!("Received: {} bytes", line.len());

            let request: JsonRpcRequest = match serde_json::from_str(&line) {
                Ok(req) => req,
                Err(e) => {
                    error!("Failed to parse request: {}", e);
                    let response = JsonRpcResponse::error(None, -32700, "Parse error".to_string());
                    let response_str = serde_json::to_string(&response)?;
                    writeln!(stdout, "{}", response_str)?;
                    stdout.flush()?;
                    continue;
                }
            };

            let response = self.handle_request_safe(&request);

            // Don't send response for notifications (no id)
            if request.id.is_some() || response.error.is_some() {
                let response_str = serde_json::to_string(&response)?;
                debug!("Sending: {} bytes", response_str.len());
                writeln!(stdout, "{}", response_str)?;
                stdout.flush()?;
            }
        }

        Ok(())
    }
}
