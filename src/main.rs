mod belief;
mod bll;
mod config;
mod embedding;
mod mcp;
mod storage;
mod tools;
mod types;
mod vector_store;

use anyhow::Result;
use config::Config;
use mcp::McpServer;
use storage::StorageManager;
use tools::ToolHandler;
use tracing_subscriber::EnvFilter;

fn main() -> Result<()> {
    // Init tracing (stderr so it doesn't interfere with MCP stdio)
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    // Create default config if missing
    Config::create_default_if_missing();

    // Init storage
    let storage = StorageManager::new()?;
    let handler = ToolHandler::new(storage);

    // Create MCP server and register tools
    let mut server = McpServer::new("mcp-threadbridge", env!("CARGO_PKG_VERSION"));

    for tool in ToolHandler::get_tools() {
        let h = handler.clone();
        let name = tool.name.clone();
        server.register_tool(tool, move |args| h.handle(&name, args));
    }

    // Run on stdio
    server.run()
}
