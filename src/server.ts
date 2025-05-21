#!/usr/bin/env node
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ErrorCode,
  ListToolsRequestSchema,
  McpError,
  type ServerResult,
} from '@modelcontextprotocol/sdk/types.js';
import { spawn } from 'node:child_process';
import { existsSync, watch } from 'node:fs';
import { promises as fs_async } from 'node:fs'; // Renamed to avoid conflict with 'fs' module if used synchronously
import { homedir } from 'node:os';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve as pathResolve, normalize as pathNormalize, sep as pathSep } from 'node:path';
import * as path from 'path'; // Ensure 'path' module is available for path.sep
import * as os from 'os'; // Added os import
import retry from 'async-retry';
// import packageJson from '../package.json' with { type: 'json' }; // Import package.json with attribute
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const packageJson = require('../package.json');

// --- BEGIN MODIFICATION ---
// --- Configuration for Guardrails ---
const rawAllowedBaseDir = process.env.MCP_ALLOWED_BASE_DIR;
if (!rawAllowedBaseDir) {
    console.error("[Critical Error] MCP_ALLOWED_BASE_DIR environment variable is not set. This is required for security. Server will not start.");
    process.exit(1); 
}
// Resolve '~' to home directory and then normalize + resolve the path
const resolvedRawAllowedBaseDir = rawAllowedBaseDir.startsWith(`~${path.sep}`) 
    ? path.join(homedir(), rawAllowedBaseDir.substring(2)) 
    : (rawAllowedBaseDir === '~' ? homedir() : rawAllowedBaseDir);

const ALLOWED_BASE_WORK_DIRECTORY = pathNormalize(pathResolve(resolvedRawAllowedBaseDir));


const DANGEROUS_COMMAND_PATTERNS: RegExp[] = [
    /rm -rf \//,         // Deleting root
    /rm -rf \.\.\//,     // Deleting parent directory content
    /rm -rf ~/,          // Deleting home directory
    /sudo rm/,           // Sudo remove
    /mkfs/,              // Formatting disks
    /> \/dev\/sd[a-z]/,  // Writing directly to disk devices
    /dd if=\/dev\/zero of=\/dev\/sd[a-z]/, // Wiping disks
    /:(){:|:&};:/,       // Fork bomb
    /\^.\*\/c\/\^.\*\/c\/\^.\*\/c/, // Another fork bomb variant
    // Add more patterns as needed
];
// --- End Configuration for Guardrails ---
// --- END MODIFICATION ---

// Define environment variables globally
const debugMode = process.env.MCP_CLAUDE_DEBUG === 'true';
const heartbeatIntervalMs = parseInt(process.env.MCP_HEARTBEAT_INTERVAL_MS || '15000', 10); // Default: 15 seconds
const executionTimeoutMs = parseInt(process.env.MCP_EXECUTION_TIMEOUT_MS || '1800000', 10); // Default: 30 minutes
const useRooModes = process.env.MCP_USE_ROOMODES === 'true'; // Enable Roo mode integration
const maxRetries = parseInt(process.env.MCP_MAX_RETRIES || '3', 10); // Default: 3 retries
const retryDelayMs = parseInt(process.env.MCP_RETRY_DELAY_MS || '1000', 10); // Default: 1 second
const watchRooModes = process.env.MCP_WATCH_ROOMODES === 'true'; // Auto-reload .roomodes file on changes

// Dedicated debug logging function
function debugLog(message?: any, ...optionalParams: any[]): void {
  if (debugMode) {
    console.error(message, ...optionalParams);
  }
}

/**
 * Determine the Claude CLI command/path.
 * 1. Checks for Claude CLI at the local user path: ~/.claude/local/claude.
 * 2. If not found, defaults to 'claude', relying on the system's PATH for lookup.
 */
function findClaudeCli(): string {
  debugLog('[Debug] Attempting to find Claude CLI...');

  // 1. Try local install path: ~/.claude/local/claude
  const userPath = join(homedir(), '.claude', 'local', 'claude');
  debugLog(`[Debug] Checking for Claude CLI at local user path: ${userPath}`);

  if (existsSync(userPath)) {
    debugLog(`[Debug] Found Claude CLI at local user path: ${userPath}. Using this path.`);
    return userPath;
  } else {
    debugLog(`[Debug] Claude CLI not found at local user path: ${userPath}.`);
  }

  // 2. Fallback to 'claude' (PATH lookup)
  debugLog('[Debug] Falling back to "claude" command name, relying on spawn/PATH lookup.');
  console.warn('[Warning] Claude CLI not found at ~/.claude/local/claude. Falling back to "claude" in PATH. Ensure it is installed and accessible.');
  return 'claude';
}

/**
 * Interface for Claude Code tool arguments
 */
interface ClaudeCodeArgs {
  prompt: string;
  workFolder?: string; // Will be validated to be within ALLOWED_BASE_WORK_DIRECTORY
  parentTaskId?: string;
  returnMode?: 'summary' | 'full';
  taskDescription?: string;
  mode?: string; // Roo mode to use (matches slug in .roomodes)
}

// --- BEGIN MODIFICATION ---
/**
 * Interface for Convert Task Markdown tool arguments
 */
interface ConvertTaskMarkdownArgs {
    markdownPath: string; // Relative to workFolder
    workFolder: string; // Absolute path, must be within ALLOWED_BASE_WORK_DIRECTORY
    outputPath?: string; // Relative to workFolder
}
// --- END MODIFICATION ---

// Cache for Roo modes configuration to improve performance
let roomodesCache: { data: any, timestamp: number } | null = null;
const CACHE_TTL_MS = 60000; // 1 minute cache TTL

// Setup file watcher for roomodes if enabled
if (useRooModes && watchRooModes) {
  const roomodesPath = path.join(process.cwd(), '.roomodes');
  if (existsSync(roomodesPath)) {
    try {
      const watcher = watch(roomodesPath, (eventType, filename) => {
        if (eventType === 'change') {
          // Invalidate cache when file changes
          roomodesCache = null;
          console.error(`[Info] .roomodes file changed, cache invalidated`);
        }
      });
      
      // Ensure the watcher is closed on process exit
      process.on('exit', () => {
        try {
          watcher.close();
        } catch (err) {
          // Ignore errors during shutdown
        }
      });
      
      console.error(`[Setup] Watching .roomodes file for changes`);
    } catch (error) {
      console.error(`[Warning] Failed to set up watcher for .roomodes file:`, error);
    }
  } else {
    console.error(`[Warning] Cannot watch .roomodes file as it doesn't exist at: ${roomodesPath}`);
  }
}

// Function to load Roo modes configuration with caching
function loadRooModes(): any {
  try {
    const roomodesPath = path.join(process.cwd(), '.roomodes');
    if (!existsSync(roomodesPath)) {
      return null;
    }
    
    // Check if we have a fresh cached version
    const fs_sync = require('fs'); // Use require for sync fs ops in this function
    const stats = fs_sync.statSync(roomodesPath);
    const fileModifiedTime = stats.mtimeMs;
    
    // Use cache if available and fresh
    if (roomodesCache && roomodesCache.timestamp > fileModifiedTime) {
      if (Date.now() - roomodesCache.timestamp < CACHE_TTL_MS) {
        debugLog('[Debug] Using cached .roomodes configuration');
        return roomodesCache.data;
      }
    }
    
    // Otherwise read the file and update cache
    const roomodesContent = fs_sync.readFileSync(roomodesPath, 'utf8');
    const parsedData = JSON.parse(roomodesContent);
    
    // Update cache
    roomodesCache = {
      data: parsedData,
      timestamp: Date.now()
    };
    
    debugLog('[Debug] Loaded fresh .roomodes configuration');
    return parsedData;
  } catch (error) {
    debugLog('[Error] Failed to load .roomodes file:', error);
    return null;
  }
}

// Ensure spawnAsync is defined correctly *before* the class
/**
 * Execute a command asynchronously with progress reporting to prevent client timeouts.
 * Sends heartbeat messages to stderr every 15 seconds to keep the connection alive.
 */
async function spawnAsync(command: string, args: string[], options?: { timeout?: number, cwd?: string }): Promise<{ stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    debugLog(`[Spawn] Running command: ${command} ${args.join(' ')} in CWD: ${options?.cwd}`);
    const proc = spawn(command, args, { // Renamed to proc
      shell: false, 
      timeout: options?.timeout,
      cwd: options?.cwd,
      stdio: ['ignore', 'pipe', 'pipe']
    });

    let stdout = '';
    let stderr = '';
    let executionStartTime = Date.now();
    let heartbeatCounter = 0;

    const progressReporter = setInterval(() => {
      heartbeatCounter++;
      const elapsedSeconds = Math.floor((Date.now() - executionStartTime) / 1000);
      const heartbeatMessage = `[Progress] Claude Code execution in progress: ${elapsedSeconds}s elapsed (heartbeat #${heartbeatCounter})`;
      
      console.error(heartbeatMessage); 
      debugLog(heartbeatMessage); 
    }, heartbeatIntervalMs);

    proc.stdout.on('data', (data) => { stdout += data.toString(); });
    proc.stderr.on('data', (data) => {
      stderr += data.toString();
      debugLog(`[Spawn Stderr Chunk] ${data.toString()}`);
    });

    proc.on('error', (error: NodeJS.ErrnoException) => {
      clearInterval(progressReporter); 
      debugLog(`[Spawn Error Event] Full error object:`, error);
      let errorMessage = `Spawn error: ${error.message}`;
      if (error.path) {
        errorMessage += ` | Path: ${error.path}`;
      }
      if (error.syscall) {
        errorMessage += ` | Syscall: ${error.syscall}`;
      }
      errorMessage += `\nStderr: ${stderr.trim()}`;
      reject(new Error(errorMessage));
    });

    proc.on('close', (code) => {
      clearInterval(progressReporter); 
      const executionTimeMs = Date.now() - executionStartTime;
      debugLog(`[Spawn Close] Exit code: ${code}, Execution time: ${executionTimeMs}ms`);
      debugLog(`[Spawn Stderr Full] ${stderr.trim()}`);
      debugLog(`[Spawn Stdout Full] ${stdout.trim()}`);
      if (code === 0) {
        resolve({ stdout, stderr });
      } else {
        reject(new Error(`Command failed with exit code ${code}\nStderr: ${stderr.trim()}\nStdout: ${stdout.trim()}`));
      }
    });
  });
}

// --- BEGIN MODIFICATION ---
// Helper function to check if a path is within the allowed base directory
function isPathSafe(pathToTest: string, allowedBase: string): boolean {
    const normalizedPathToTest = pathNormalize(pathResolve(pathToTest));
    const normalizedAllowedBase = pathNormalize(pathResolve(allowedBase));
    return normalizedPathToTest.startsWith(normalizedAllowedBase + pathSep) || normalizedPathToTest === normalizedAllowedBase;
}
// --- END MODIFICATION ---

/**
 * MCP Server for Claude Code
 * Provides a simple MCP tool to run Claude CLI in one-shot mode
 */
class ClaudeCodeServer {
  private server: Server;
  private claudeCliPath: string; 
  private packageVersion: string; 
  private activeRequests: Set<string> = new Set(); 

  constructor() {
    this.claudeCliPath = findClaudeCli(); 
    console.error(`[Setup] Using Claude CLI command/path: ${this.claudeCliPath}`);
    // --- BEGIN MODIFICATION ---
    console.error(`[Setup] Allowed base working directory: ${ALLOWED_BASE_WORK_DIRECTORY}`); 
    if (!existsSync(ALLOWED_BASE_WORK_DIRECTORY)) {
        console.warn(`[Warning] ALLOWED_BASE_WORK_DIRECTORY "${ALLOWED_BASE_WORK_DIRECTORY}" does not exist. Some operations might fail if it's not created.`);
    }
    // --- END MODIFICATION ---
    this.packageVersion = packageJson.version; 

    this.server = new Server(
      {
        name: 'claude_code_mcp_server', // Server name, not tool name prefix
        version: this.packageVersion, 
      },
      {
        capabilities: {
          tools: {}, 
        },
      }
    );

    this.setupToolHandlers();

    this.server.onerror = (error) => console.error('[MCP Server Error]', error); 
    
    const handleShutdown = async (signal: string) => {
      console.error(`[Shutdown] Received ${signal} signal. Graceful shutdown initiated.`);
      if (this.activeRequests.size > 0) {
        console.error(`[Shutdown] Waiting for ${this.activeRequests.size} active requests to complete...`);
        const shutdownTimeoutMs = 10000;
        const shutdownStart = Date.now();
        while (this.activeRequests.size > 0 && (Date.now() - shutdownStart) < shutdownTimeoutMs) {
          await new Promise(resolve => setTimeout(resolve, 100));
        }
        if (this.activeRequests.size > 0) {
          console.error(`[Shutdown] ${this.activeRequests.size} requests still active after timeout. Proceeding with shutdown anyway.`);
        } else {
          console.error('[Shutdown] All active requests completed successfully.');
        }
      }
      await this.server.close();
      console.error('[Shutdown] Server closed. Exiting process.');
      process.exit(0);
    };
    
    process.on('SIGINT', () => handleShutdown('SIGINT'));
    process.on('SIGTERM', () => handleShutdown('SIGTERM'));
  }

  /**
   * Set up the MCP tool handlers
   */
  private setupToolHandlers(): void {
    this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: [
        {
          name: 'health', 
          description: 'Returns health status, version information, and current configuration of the Claude Code MCP server.',
          inputSchema: {
            type: 'object',
            properties: {},
            required: [],
          },
        },
        {
          name: 'convert_task_markdown',
          // --- BEGIN MODIFICATION for convert_task_markdown description and inputSchema ---
          description: `Converts markdown task files into Claude Code MCP-compatible JSON format. Returns an array of tasks that can be executed using the claude_code tool. The 'workFolder' argument must be an absolute path within the server's configured allowed base directory.`,
          inputSchema: {
            type: 'object',
            properties: {
              markdownPath: {
                type: 'string',
                description: 'Relative path from workFolder to the markdown task file to convert.',
              },
              workFolder: { 
                type: 'string',
                description: `The absolute path to the project root. Must be within the server's allowed base directory (${ALLOWED_BASE_WORK_DIRECTORY}).`,
              },
              outputPath: {
                type: 'string',
                description: 'Optional relative path from workFolder where to save the JSON output. If not provided, returns the JSON directly.',
              },
            },
            required: ['markdownPath', 'workFolder'], 
          },
          // --- END MODIFICATION for convert_task_markdown description and inputSchema ---
        },
        {
          name: 'claude_code',
          // --- BEGIN MODIFICATION for claude_code description and inputSchema ---
          description: `Claude Code Agent: Your versatile multi-modal assistant for code, file, Git, and terminal operations via Claude CLI. Use \`workFolder\` for contextual execution.

• File ops: Create, read, (fuzzy) edit, move, copy, delete, list files, analyze/ocr images, file content analysis
    └─ e.g., "Create /tmp/log.txt with 'system boot'", "Edit main.py to replace 'debug_mode = True' with 'debug_mode = False'", "List files in /src", "Move a specific section somewhere else"

• Code: Generate / analyse / refactor / fix
    └─ e.g. "Generate Python to parse CSV→JSON", "Find bugs in my_script.py"

• Git: Stage ▸ commit ▸ push ▸ tag (any workflow)
    └─ "Commit '/workspace/src/main.java' with 'feat: user auth' to develop."

• Terminal: Run any CLI cmd or open URLs
    └─ "npm run build", "Open https://developer.mozilla.org"

• Web search + summarise content on-the-fly

• Multi-step workflows  (Version bumps, changelog updates, release tagging, etc.)

• GitHub integration  Create PRs, check CI status

• Confused or stuck on an issue? Ask Claude Code for a second opinion, it might surprise you!

• Task Orchestration with "Boomerang" pattern
    └─ Break down complex tasks into subtasks for Claude Code to execute separately
    └─ Pass parent task ID and get results back for complex workflows
    └─ Specify return mode (summary or full) for tailored responses

**Prompt tips**

1. Be concise, explicit & step-by-step for complex tasks. No need for niceties, this is a tool to get things done.
2. For multi-line text, write it to a temporary file in the project root, use that file, then delete it.
3. If you get a timeout, split the task into smaller steps.
4. **Seeking a second opinion/analysis**: If you're stuck or want advice, you can ask \`claude_code\` to analyze a problem and suggest solutions. Clearly state in your prompt that you are looking for analysis only and no actual file modifications should be made.
5. If workFolder is set to the project path, there is no need to repeat that path in the prompt and you can use relative paths for files.
6. Claude Code is really good at complex multi-step file operations and refactorings and faster than your native edit features.
7. Combine file operations, README updates, and Git commands in a sequence.
8. **Task Orchestration**: For complex workflows, use \`parentTaskId\` to create subtasks and \`returnMode: "summary"\` to get concise results back.
9. Claude can do much more, just ask it!

The 'workFolder' argument must be an absolute path within the server's configured allowed base directory (${ALLOWED_BASE_WORK_DIRECTORY}).
        `, 
          inputSchema: {
            type: 'object',
            properties: {
              prompt: {
                type: 'string',
                description: 'The detailed natural language prompt for Claude to execute.',
              },
              workFolder: { 
                type: 'string',
                description: `Mandatory. The absolute working directory for the Claude CLI execution. Must be within the server's allowed base directory (${ALLOWED_BASE_WORK_DIRECTORY}).`,
              },
              parentTaskId: {
                type: 'string',
                description: 'Optional ID of the parent task that created this task (for task orchestration/boomerang).',
              },
              returnMode: {
                type: 'string',
                enum: ['summary', 'full'],
                description: 'How results should be returned: summary (concise) or full (detailed). Defaults to full.',
              },
              taskDescription: {
                type: 'string',
                description: 'Short description of the task for better organization and tracking in orchestrated workflows.',
              },
              mode: {
                type: 'string',
                description: 'When MCP_USE_ROOMODES=true, specifies the mode from .roomodes to use (e.g., "boomerang-mode", "coder", "designer", etc.).',
              },
            },
            required: ['prompt', 'workFolder'], 
          },
          // --- END MODIFICATION for claude_code description and inputSchema ---
        }
      ],
    }));

    this.server.setRequestHandler(CallToolRequestSchema, async (args, call): Promise<ServerResult> => {
      const requestId = `req_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
      this.activeRequests.add(requestId);
      debugLog(`[Debug] Handling CallToolRequest: ${requestId}`, args);

      const fullToolNameFromRequest = args.params.name;
      const toolName = fullToolNameFromRequest.includes(':') ? fullToolNameFromRequest.split(':')[1] : fullToolNameFromRequest;
      debugLog(`[Debug] Full tool name from request: ${fullToolNameFromRequest}, Local tool name: ${toolName}`);

      const toolArguments = args.params.arguments as any; 

      if (toolName === 'health') {
        let claudeCliStatus = 'unknown';
        try {
          // --- BEGIN MODIFICATION (Original used /bin/bash, direct call is fine) ---
          // The original spawnAsync call for health check was:
          // const { stdout } = await spawnAsync('/bin/bash', [this.claudeCliPath, '--version'], { timeout: 5000 });
          // Changing to direct call:
          await spawnAsync(this.claudeCliPath, ['--version'], { timeout: 5000 });
          // --- END MODIFICATION ---
          claudeCliStatus = 'available';
        } catch (error) { claudeCliStatus = 'unavailable'; }
        const healthInfo = {
          status: 'ok', 
          version: this.packageVersion,
          claudeCli: { 
            path: this.claudeCliPath, 
            status: claudeCliStatus 
          },
          config: { 
            debugMode, 
            heartbeatIntervalMs, 
            executionTimeoutMs, 
            useRooModes, 
            maxRetries, 
            retryDelayMs, 
            // --- BEGIN MODIFICATION ---
            allowedBaseWorkDirectory: ALLOWED_BASE_WORK_DIRECTORY 
            // --- END MODIFICATION ---
          },
          system: { 
            platform: os.platform(), 
            release: os.release(), 
            arch: os.arch(), 
            cpus: os.cpus().length, 
            memory: { 
              total: Math.round(os.totalmem() / (1024 * 1024)) + 'MB', 
              free: Math.round(os.freemem() / (1024 * 1024)) + 'MB' 
            }, 
              uptime: Math.round(os.uptime() / 60) + ' minutes' 
            },
          timestamp: new Date().toISOString()
        };
        this.activeRequests.delete(requestId);
        debugLog(`[Debug] Health check request ${requestId} completed`);
        return { content: [{ type: 'text', text: JSON.stringify(healthInfo, null, 2) }] };
      }

      if (toolName === 'convert_task_markdown') {
        // --- BEGIN MODIFICATIONS for convert_task_markdown logic ---
        const converterArgs = toolArguments as ConvertTaskMarkdownArgs; 
        if (!converterArgs || typeof converterArgs.markdownPath !== 'string' || typeof converterArgs.workFolder !== 'string') {
          this.activeRequests.delete(requestId);
          throw new McpError(ErrorCode.InvalidParams, 'Missing or invalid required parameters: markdownPath (string) and workFolder (string) for convert_task_markdown tool');
        }
        
        const workFolder_conv = pathNormalize(pathResolve(converterArgs.workFolder)); // Use a different variable name
        if (!isPathSafe(workFolder_conv, ALLOWED_BASE_WORK_DIRECTORY)) {
            this.activeRequests.delete(requestId);
            throw new McpError(ErrorCode.InvalidParams, `convert_task_markdown: workFolder "${workFolder_conv}" is outside the allowed base directory "${ALLOWED_BASE_WORK_DIRECTORY}".`);
        }
        if (!existsSync(workFolder_conv)) { 
            this.activeRequests.delete(requestId);
            throw new McpError(ErrorCode.InvalidParams, `convert_task_markdown: workFolder "${workFolder_conv}" does not exist. It must exist to locate the markdown file.`);
        }

        const markdownFileFullPath = pathResolve(workFolder_conv, converterArgs.markdownPath);
         if (!isPathSafe(markdownFileFullPath, workFolder_conv)) { 
            this.activeRequests.delete(requestId);
            throw new McpError(ErrorCode.InvalidParams, `convert_task_markdown: markdownPath "${converterArgs.markdownPath}" (resolved to ${markdownFileFullPath}) attempts to access outside of its workFolder "${workFolder_conv}".`);
        }
        if (!existsSync(markdownFileFullPath)) {
            this.activeRequests.delete(requestId);
            throw new McpError(ErrorCode.InvalidParams, `convert_task_markdown: Markdown file not found at resolved path: ${markdownFileFullPath}`);
        }

        let outputFileFullPath: string | undefined;
        if (converterArgs.outputPath && typeof converterArgs.outputPath === 'string') {
          outputFileFullPath = pathResolve(workFolder_conv, converterArgs.outputPath);
           if (!isPathSafe(outputFileFullPath, workFolder_conv)) { 
                this.activeRequests.delete(requestId);
                throw new McpError(ErrorCode.InvalidParams, `convert_task_markdown: outputPath "${converterArgs.outputPath}" (resolved to ${outputFileFullPath}) attempts to access outside of its workFolder "${workFolder_conv}".`);
            }
        }
        
        debugLog(`[Debug] Converting markdown task file: ${markdownFileFullPath} within project: ${workFolder_conv}`);
        let stderr_from_converter = '';
        try {
          const pythonPath = 'python3'; 
          const currentModuleUrl = import.meta.url;
          const currentModulePath = fileURLToPath(currentModuleUrl); // Use fileURLToPath
          const currentModuleDir = dirname(currentModulePath);       // Use dirname
          
          // Note: On Windows, fileURLToPath might return a path like /C:/...
          // path.resolve will handle this correctly.
          const converterScriptPath = pathResolve(currentModuleDir, '../docs/task_converter.py');
          // --- END CORRECTION for __dirname in ES Modules ---
          
          debugLog(`[Debug] Path to converter script: ${converterScriptPath}`); 
          if (!existsSync(converterScriptPath)) {
            this.activeRequests.delete(requestId);
            throw new McpError(ErrorCode.InternalError, `Task converter script not found at ${converterScriptPath}. Check server installation.`);
          }
          const pythonArgs = ['--json-output', markdownFileFullPath, '--project-path', workFolder_conv];
          
          const result = await spawnAsync(pythonPath, [converterScriptPath, ...pythonArgs], {
            cwd: workFolder_conv, 
            timeout: 60000 
          });
          
          const stdout_from_converter = result.stdout;
          stderr_from_converter = result.stderr;
          
          const stderrLines = stderr_from_converter.split('\n');
          const progressMessages = stderrLines.filter(line => line.includes('[Progress]') || line.includes('[Warning]'));
          const errorMessages = stderrLines.filter(line => !line.includes('[Progress]') && !line.includes('[Warning]') && line.trim());
          
          progressMessages.forEach(msg => { console.error(msg); debugLog(msg); });
          if (errorMessages.length > 0) { 
            stderr_from_converter = errorMessages.join('\n'); 
            debugLog(`[Debug] Task converter error output: ${stderr_from_converter}`); 
          } else if (progressMessages.length > 0 && !stderr_from_converter.toLowerCase().includes('error')) {
            stderr_from_converter = ''; 
          }
          
          if (stderr_from_converter && stderr_from_converter.toLowerCase().includes('error')) { 
            const validationError = { status: 'error', error: 'Markdown conversion process reported errors', details: stderr_from_converter, helpUrl: 'https://github.com/sfearl1/claude-code-mcp/blob/main/README.md#markdown-task-file-format' };
            this.activeRequests.delete(requestId);
            return { content: [{ type: 'text', text: JSON.stringify(validationError, null, 2) }] };
          }
          
          const tasks = JSON.parse(stdout_from_converter);
          if (outputFileFullPath) {
            await fs_async.writeFile(outputFileFullPath, JSON.stringify(tasks, null, 2)); // Use fs_async
            debugLog(`[Debug] Saved converted tasks to: ${outputFileFullPath}`);
          }
          
          const response = { status: 'success', tasksCount: tasks.length, outputPath: outputFileFullPath || 'none', tasks: tasks };
          this.activeRequests.delete(requestId);
          return { content: [{ type: 'text', text: JSON.stringify(response, null, 2) }] };
          
        } catch (error) { 
          this.activeRequests.delete(requestId);
          const errorMessage = error instanceof Error ? error.message : String(error);
          const details = stderr_from_converter || errorMessage; 
          const finalError = { status: 'error', error: 'Task conversion failed', details: details, helpUrl: 'https://github.com/sfearl1/claude-code-mcp/blob/main/README.md#markdown-task-file-format' };
          return { content: [{ type: 'text', text: JSON.stringify(finalError, null, 2) }] };
        }
        // --- END MODIFICATIONS for convert_task_markdown logic ---
      }
      
      // This check was in the original, ensuring it's still here
      // If it's not health or convert_task_markdown, it must be claude_code or an error
      if (toolName !== 'claude_code') { // Simplified this check as health and convert_task_markdown are handled above
        this.activeRequests.delete(requestId);
        throw new McpError(ErrorCode.MethodNotFound, `Tool ${toolName} not found on this server.`);
      }

      // --- BEGIN MODIFICATIONS for claude_code argument validation and guardrails ---
      // The original file had a block here:
      // if (toolName !== 'claude_code' && toolName !== 'health' && toolName !== 'convert_task_markdown') {
      //   throw new McpError(ErrorCode.MethodNotFound, `Tool ${toolName} not found`);
      // }
      // This is now covered by the check above. The following logic is for 'claude_code'.

      const claudeArgs = toolArguments as ClaudeCodeArgs; 
      if (!claudeArgs || typeof claudeArgs.prompt !== 'string' || typeof claudeArgs.workFolder !== 'string') {
        this.activeRequests.delete(requestId);
        throw new McpError(ErrorCode.InvalidParams, 'Missing or invalid required parameters: prompt (string) and workFolder (string) for claude_code tool');
      }
      
      // Destructure after validation
      let currentPrompt = claudeArgs.prompt; // Use new variable name to avoid redeclaration
      let currentWorkFolder = claudeArgs.workFolder;
      let currentParentTaskId = claudeArgs.parentTaskId;
      let currentReturnMode = claudeArgs.returnMode || 'full';
      let currentTaskDescription = claudeArgs.taskDescription;
      let currentMode = claudeArgs.mode;

      const currentEffectiveCwd = pathNormalize(pathResolve(currentWorkFolder!)); 
      if (!isPathSafe(currentEffectiveCwd, ALLOWED_BASE_WORK_DIRECTORY)) {
        this.activeRequests.delete(requestId);
        throw new McpError(ErrorCode.InvalidParams, `claude_code: workFolder "${currentEffectiveCwd}" is outside the allowed base directory "${ALLOWED_BASE_WORK_DIRECTORY}".`);
      }
      
      // Original logic for workFolder existence check (more lenient for claude_code)
      if (!existsSync(currentEffectiveCwd)) {
         // This was the original behavior for claude_code's workFolder check, so we keep it.
        debugLog(`[Debug] claude_code: Specified workFolder "${currentEffectiveCwd}" does not exist. Claude CLI might create it if the prompt instructs to.`);
      }


      // Dangerous command check
      for (const pattern of DANGEROUS_COMMAND_PATTERNS) {
        if (pattern.test(currentPrompt)) { // Check currentPrompt
            this.activeRequests.delete(requestId);
            throw new McpError(ErrorCode.InvalidParams, `Prompt contains a potentially dangerous command pattern: ${pattern.toString()}. Execution blocked.`);
        }
      }
      // --- END MODIFICATIONS for claude_code argument validation and guardrails ---
      
      // Original logic for parentTaskId, returnMode, taskDescription, mode using current* variables
      if (currentParentTaskId) {
        const taskContext = `
# Boomerang Task
${currentTaskDescription ? `## Task Description\n${currentTaskDescription}\n\n` : ''}
## Parent Task ID
${currentParentTaskId}

## Return Instructions
You are part of a larger workflow. After completing your task, you should ${currentReturnMode === 'summary' ? 'provide a BRIEF SUMMARY of the results' : 'return your FULL RESULTS'}.

${currentReturnMode === 'summary' ? 'IMPORTANT: Keep your response concise and focused on key findings/changes only!' : ''}

---

`;
        currentPrompt = taskContext + currentPrompt; // Modify currentPrompt
        debugLog(`[Debug] Prepended boomerang task context to prompt`);
      }
      
      try {
        debugLog(`[Debug] Attempting to execute Claude CLI with prompt: "${currentPrompt.substring(0,100)}..." in CWD: "${currentEffectiveCwd}"`);
        
        let claudeProcessArgs = ['--dangerously-skip-permissions']; 
        
        if (useRooModes && currentMode) { // Use currentMode
          const roomodes = loadRooModes();
          if (roomodes && roomodes.customModes) {
            const selectedMode = roomodes.customModes.find((m: any) => m.slug === currentMode); // Use currentMode
            if (selectedMode) {
              debugLog(`[Debug] Found Roo mode configuration for: ${currentMode}`); // Use currentMode
              claudeProcessArgs.push('--role', selectedMode.roleDefinition);
              if (selectedMode.apiConfiguration && selectedMode.apiConfiguration.modelId) {
                claudeProcessArgs.push('--model', selectedMode.apiConfiguration.modelId);
              }
            } else { debugLog(`[Warning] Specified Roo mode "${currentMode}" not found in .roomodes`); } // Use currentMode
          } else { debugLog(`[Warning] Roo modes configuration not found or invalid, cannot apply mode: ${currentMode}`); } // Use currentMode
        }
        claudeProcessArgs.push('-p', currentPrompt); // Use currentPrompt
        
        // The original code used /bin/bash here. We are now calling claudeCliPath directly.
        debugLog(`[Debug] Invoking Claude CLI (${this.claudeCliPath}) with args: ${claudeProcessArgs.join(' ')}`);

        const { stdout, stderr } = await retry(
          async (bail: (err: Error) => void, attemptNumber: number) => {
            try {
              if (attemptNumber > 1) {
                debugLog(`[Retry] Attempt ${attemptNumber}/${maxRetries + 1} for Claude CLI execution`);
              }
              return await spawnAsync(this.claudeCliPath, claudeProcessArgs, { timeout: executionTimeoutMs, cwd: currentEffectiveCwd }); // Use currentEffectiveCwd
            } catch (err: any) {
              debugLog(`[Retry] Error during attempt ${attemptNumber}/${maxRetries + 1}: ${err.message}`);
              const isNetworkError = err.message.includes('ECONNRESET') || 
                                    err.message.includes('ETIMEDOUT') ||
                                    err.message.includes('ECONNREFUSED');
              const isTransientError = isNetworkError || 
                                      err.message.includes('429') || 
                                      err.message.includes('500'); 
              if (!isTransientError) { 
                debugLog(`[Retry] Non-retryable error encountered. Bailing.`);
                bail(err); 
                return { stdout: '', stderr: '' }; 
              }
              throw err; 
            }
          },
          { 
            retries: maxRetries, 
            minTimeout: retryDelayMs, 
            onRetry: (err: Error, attempt: number) => {
                console.error(`[Progress] Retry attempt ${attempt}/${maxRetries} for claude_code tool due to: ${err.message.split('\n')[0]}`);
                debugLog(`[Retry Full Error] Attempt ${attempt}:`, err);
            }
          }
        );

        debugLog('[Debug] Claude CLI stdout:', stdout.trim().substring(0, 200) + (stdout.trim().length > 200 ? "..." : ""));
        if (stderr) debugLog('[Debug] Claude CLI stderr:', stderr.trim().substring(0, 200) + (stderr.trim().length > 200 ? "..." : ""));

        let processedOutput = stdout;
        if (currentParentTaskId) { // Use currentParentTaskId
          const boomerangInfo = { 
            parentTaskId: currentParentTaskId, // Use currentParentTaskId
            returnMode: currentReturnMode, // Use currentReturnMode
            taskDescription: currentTaskDescription || 'Unknown task', // Use currentTaskDescription
            completed: new Date().toISOString() 
          };
          const boomerangMarker = `\n\n<!-- BOOMERANG_RESULT ${JSON.stringify(boomerangInfo)} -->`;
          processedOutput += boomerangMarker;
          debugLog(`[Debug] Added boomerang marker to output for parent task: ${currentParentTaskId}`); // Use currentParentTaskId
        }

        this.activeRequests.delete(requestId);
        debugLog(`[Debug] Request ${requestId} completed successfully`);
        return { content: [{ type: 'text', text: processedOutput }] };

      } catch (error: any) {
        debugLog('[Error] Error executing Claude CLI:', error);
        let errorMessage = error.message || 'Unknown error';
        // Stderr/Stdout might be part of error.message from spawnAsync already
        this.activeRequests.delete(requestId);
        debugLog(`[Debug] Request ${requestId} failed: ${errorMessage}`);
        if (error.signal === 'SIGTERM' || (error.message && error.message.includes('ETIMEDOUT')) || (error.code === 'ETIMEDOUT')) {
          throw new McpError(ErrorCode.InternalError, `Claude CLI command timed out after ${executionTimeoutMs / 1000}s. Details: ${errorMessage}`);
        }
        throw new McpError(ErrorCode.InternalError, `Claude CLI execution failed: ${errorMessage}`);
      }
    });
  }

  /**
   * Start the MCP server
   */
  async run(): Promise<void> {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    // --- BEGIN MODIFICATION ---
    console.error(`Claude Code MCP server (Customized) running on stdio. Restricted to base directory: ${ALLOWED_BASE_WORK_DIRECTORY}`);
    // --- END MODIFICATION ---
  }
}

// Create and run the server
const server = new ClaudeCodeServer();
// --- BEGIN MODIFICATION ---
server.run().catch(error => {
    console.error("[Critical Server Startup Error]", error); 
    process.exit(1);
});
// --- END MODIFICATION ---