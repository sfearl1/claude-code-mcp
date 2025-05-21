#!/usr/bin/env python3
"""
Task Converter for Claude Code MCP

This script converts Markdown validation tasks into machine-readable JSON
format that is compatible with Claude Code MCP (Model Context Protocol).

The script analyzes Markdown files containing validation task definitions and outputs
a JSON list of tasks that can be directly consumed by the Claude Code MCP server.

### Claude Code MCP Integration ###

The output from this converter is specifically designed to work with the Claude Code MCP
server described in: https://github.com/grahama1970/claude-code-mcp/

The MCP (Model Context Protocol) is a standardized way for AI models to interact with
external tools and services. This converter generates prompts that are formatted
to be processed by the Claude Code MCP, which allows Claude to:

1. Understand code tasks and their requirements
2. Execute appropriate code generation based on task specifications
3. Format output according to expectations
4. Process multiple tasks in sequence

### Output Format ###

The output JSON has the following structure:
[
  {
    "tool": "claude_code",
    "arguments": {
      "prompt": "cd /path/to/project && [Detailed prompt for the task]", // Modified by script
      "workFolder": "/path/to/project", // Added by script
      "dangerously_skip_permissions": true,
      "timeout_ms": 300000
    }
  },
  ...
]

The "dangerously_skip_permissions" flag is set to true to allow Claude Code to execute
operations without permission interruptions, and a timeout is set to prevent long-running tasks.

### Usage ###
    # File output mode:
    python task_converter.py <input_markdown> <output_json> --project-path /abs/path/to/project
    
    # JSON stdout mode (for MCP integration):
    python task_converter.py --json-output <input_markdown> --project-path /abs/path/to/project

### Example ###
    # File output:
    python task_converter.py docs/tasks/011_db_operations_validation.md tasks.json --project-path /Users/youruser/your_project_path
    
    # JSON stdout (for MCP):
    python task_converter.py --json-output docs/tasks/011_db_operations_validation.md --project-path /Users/youruser/your_project_path
"""

import re
import json
import sys
import os
# --- BEGIN MODIFICATION ---
import argparse # Ensure argparse is imported
# --- END MODIFICATION ---
from typing import List, Dict, Tuple, Any, Optional

def load_file(filename: str) -> str:
    """
    Load content from a markdown file.
    
    Args:
        filename: Path to the markdown file to load
        
    Returns:
        String containing the file content
    """
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()

def extract_title(md: str) -> str:
    """
    Extract the title from the markdown content.
    
    Args:
        md: Markdown content
        
    Returns:
        The title of the task
    """
    title_match = re.search(r'^#\s+(.+)$', md, re.MULTILINE)
    return title_match.group(1) if title_match else "Untitled Task"

def extract_objective(md: str) -> str:
    """
    Extract the objective section from the markdown content.
    
    Args:
        md: Markdown content
        
    Returns:
        The objective of the task
    """
    objective_match = re.search(r'## Objective\n(.+?)(?=\n##|\Z)', md, re.DOTALL)
    return objective_match.group(1).strip() if objective_match else ""

def extract_requirements(md: str) -> List[str]:
    """
    Extract the requirements list from the markdown content.
    
    Args:
        md: Markdown content
        
    Returns:
        List of requirement strings
    """
    requirements = []
    req_section = re.search(r'## Requirements\n(.*?)(?=\n##|\Z)', md, re.DOTALL)
    
    if req_section:
        req_text = req_section.group(1)
        # Extract all requirements (numbered lists with checkboxes)
        req_matches = re.findall(r'\d+\.\s+\[\s?\]\s*(.+)', req_text)
        requirements = [r.strip() for r in req_matches]
    
    return requirements

def extract_validation_tasks(md: str) -> List[Tuple[str, str]]:
    """
    Extract validation tasks and their corresponding steps.
    
    Args:
        md: Markdown content
        
    Returns:
        List of tuples containing (module_name, steps_block)
    """
    # Find all "- [ ] Validate `module_name`" entries and capture the module name
    # and the indented block of steps that follows
    pattern = re.compile(
        r'- \[ \] Validate `([^`]+)`\n((?:\s{3,}- \[ \].+\n?)*)',
        re.MULTILINE
    )
    return pattern.findall(md)

def extract_steps(block: str) -> List[str]:
    """
    Extract steps from an indented block.
    
    Args:
        block: Text block containing indented checklist items
        
    Returns:
        List of step strings
    """
    steps = []
    for line in block.splitlines():
        m = re.match(r'\s+- \[ \] (.+)', line)
        if m:
            steps.append(m.group(1).strip())
    return steps

# --- BEGIN MODIFICATION ---
def build_validation_prompt(title: str, objective: str, module: str, steps: List[str], 
                          requirements: List[str], project_path: str) -> str: # Added project_path
    """
    Build a detailed prompt for validating a module.
    
    Args:
        title: Task title
        objective: Task objective
        module: Name of the module to validate (relative to project_path)
        steps: List of validation steps
        requirements: List of requirements
        project_path: Absolute path to the project root
        
    Returns:
        Formatted prompt string
    """
    task_id_match = re.search(r'Task (\d+):', title)
    task_id = task_id_match.group(1) if task_id_match else "unknown"
    
    # Use the provided project_path.
    prompt = f"cd \"{project_path}\" && "
    
    # More robust venv activation check
    venv_paths_to_check = [
        os.path.join(project_path, ".venv", "bin", "activate"),
        os.path.join(project_path, "venv", "bin", "activate"),
        os.path.join(project_path, ".env", "bin", "activate"),
        os.path.join(project_path, "env", "bin", "activate"),
    ]
    activated = False
    for venv_activate_path in venv_paths_to_check:
        if os.path.exists(venv_activate_path):
            prompt += f"source \"{venv_activate_path}\" && "
            activated = True
            break
    if not activated:
        # Send warning to stderr so it doesn't interfere with JSON output if --json-output is used
        print(f"[Warning] No common virtual environment activation script found in project {project_path} (checked .venv, venv, .env, env). Skipping activation.", file=sys.stderr)

    prompt += f"\n\nTASK TYPE: Validation\n" # Keep original task type
    prompt += f"TASK ID: validation-task-{task_id}\n" # Keep original task ID format
    prompt += f"CURRENT SUBTASK: Validate {module}\n\n"
    
    prompt += f"CONTEXT:\n"
    prompt += f"- Objective: {objective}\n" # Use the extracted objective
    prompt += "- Validation must use real connections/data, not mocks.\n" # Generic instruction
    prompt += "- Results must be verified with both JSON and rich table outputs (if applicable).\n"
    
    # Construct the absolute path to the module file within the project
    absolute_module_path = os.path.join(project_path, module) # module is relative path from markdown
    prompt += f"- File to validate is located at: {absolute_module_path}\n\n"
    
    prompt += "REQUIREMENTS:\n"
    for i, req in enumerate(requirements, 1):
        prompt += f"{i}. {req}\n"
    
    prompt += f"\nVALIDATION STEPS for {module}:\n"
    for i, step in enumerate(steps, 1):
        prompt += f"{i}. {step}\n"
    
    # The instruction to update the markdown file needs to be handled carefully.
    # The agent executing this prompt will need the path to the original markdown file.
    # This script doesn't know the original markdown file's path when called by server.ts with --json-output.
    # For now, the instruction remains generic. The AI agent will need context.
    prompt += f"""
INSTRUCTIONS:
1. Execute each validation step in sequence for module: {absolute_module_path}
2. For each step:
   - Show the actual code executed (using absolute paths where necessary).
   - Show the actual output.
   - Verify the output matches expectations.
   - Include both JSON and rich table outputs where appropriate.
3. After completing all steps:
   - Update the original task list markdown file (the agent should know its path from the initial call).
   - Change "- [ ] Validate `{module}`" to "- [x] Validate `{module}`" in that file.
   - Document any issues found and fixes applied.
   - Confirm all requirements were met.
   - Confirm actual database connection/real data was used (no mocks).

After completion, provide summary in this format:

COMPLETION SUMMARY:
- What was validated: 
- Results:
- Files modified:
- Issues encountered:
- Fixes applied:
- Requirements met: [Yes/No with details]
- Used real database/connection: [Confirmed/Not confirmed]

Begin validation of {module} now.
"""
    return prompt.strip()

def format_tasks_for_mcp(validation_prompts: List[str], project_path: str) -> List[Dict[str, Any]]: # Added project_path
    """
    Format validation tasks for the Claude Code MCP format.
    
    Args:
        validation_prompts: List of formatted validation prompts
        project_path: The absolute path to the project, to be used as workFolder
        
    Returns:
        List of tasks in Claude Code MCP compatible format
    """
    mcp_tasks = []
    
    for prompt_text in validation_prompts: # Renamed 'prompt' to 'prompt_text' to avoid confusion
        mcp_task = {
            "tool": "claude_code",
            "arguments": {
                "prompt": prompt_text, # The prompt already includes the 'cd' command
                "workFolder": project_path, # Explicitly set workFolder for the claude_code tool call
                "dangerously_skip_permissions": True,
                "timeout_ms": 300000 
            }
        }
        mcp_tasks.append(mcp_task)
    
    return mcp_tasks

def process_markdown(input_file: str, project_path: str, progress_callback: Optional[callable] = None) -> List[Dict[str, Any]]: # Added project_path
    """
    Process a markdown file and extract validation tasks.
    
    Args:
        input_file: Path to the markdown file
        project_path: Absolute path to the project root for context
        progress_callback: Optional callback for progress updates
        
    Returns:
        List of tasks in Claude Code MCP format
        
    Raises:
        ValueError: If markdown format is invalid or missing required sections
    """
# --- END MODIFICATION ---
    if progress_callback:
        progress_callback(f"Loading task file: {input_file}") # Modified progress message
    
    md = load_file(input_file)
    
    if progress_callback:
        progress_callback("Validating markdown structure...")
    
    # Validate markdown structure
    validation_errors = []
    
    # Extract and validate title
    title = extract_title(md)
    if title == "Untitled Task":
        validation_errors.append("Missing required title. Format: '# Task NNN: Title'")
    
    # Extract and validate objective
    objective = extract_objective(md)
    if not objective:
        validation_errors.append("Missing required 'Objective' section. Format: '## Objective\\nDescription'")
    
    # Extract and validate requirements
    requirements = extract_requirements(md)
    if not requirements:
        validation_errors.append("Missing or empty 'Requirements' section. Format: '## Requirements\\n1. [ ] Requirement'")
    
    # Extract and validate tasks
    validation_tasks = extract_validation_tasks(md)
    if not validation_tasks:
        validation_errors.append("No validation tasks found. Format: '- [ ] Validate `module.py`' with indented steps")
    
    # Check for task steps
    empty_tasks = []
    for module, block in validation_tasks:
        steps = extract_steps(block)
        if not steps:
            empty_tasks.append(module)
    
    if empty_tasks:
        validation_errors.append(f"Tasks without steps: {', '.join(empty_tasks)}. Each task needs indented steps")
    
    # If there are validation errors, raise exception with helpful message
    if validation_errors:
        error_msg = "Markdown format validation failed:\n"
        error_msg += "\n".join(f"  - {error}" for error in validation_errors)
        error_msg += "\n\nRequired markdown format:\n"
        error_msg += "# Task NNN: Title\n"
        error_msg += "## Objective\n"
        error_msg += "Clear description\n"
        error_msg += "## Requirements\n"
        error_msg += "1. [ ] First requirement\n"
        error_msg += "## Task Section (example)\n" # Added example for clarity
        error_msg += "- [ ] Validate `src/module_name/file.py`\n" # Example with path
        error_msg += "   - [ ] Step 1 for file.py\n"
        error_msg += "   - [ ] Step 2 for file.py\n"
        raise ValueError(error_msg)
    
    if progress_callback:
        progress_callback(f"Converting {len(validation_tasks)} validation tasks from {input_file} for project {project_path}...") # Modified progress
    
    prompts = []
    for i, (module, block) in enumerate(validation_tasks, 1):
        if progress_callback:
            progress_callback(f"Task {i}/{len(validation_tasks)}: Converting {module}")
        
        steps = extract_steps(block)
        if not steps:
            continue  
        
        # --- BEGIN MODIFICATION ---
        prompt = build_validation_prompt(title, objective, module, steps, requirements, project_path) # Pass project_path
        # --- END MODIFICATION ---
        prompts.append(prompt)
    
    if progress_callback:
        progress_callback("Conversion complete!")
    
    # --- BEGIN MODIFICATION ---
    return format_tasks_for_mcp(prompts, project_path) # Pass project_path
    # --- END MODIFICATION ---

class MarkdownValidator:
    """
    Validates markdown task files for required structure and format.
    """
    
    @staticmethod
    def validate_markdown_structure(md: str) -> Tuple[bool, List[str]]:
        """
        Validate the structure of a markdown task file.
        
        Args:
            md: Markdown content to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check for required sections
        if not re.search(r'^#\s+Task\s+\d+:', md, re.MULTILINE):
            errors.append("Missing task title. Format: '# Task NNN: Title'")
            
        if not re.search(r'^##\s+Objective', md, re.MULTILINE | re.IGNORECASE):
            errors.append("Missing '## Objective' section")
            
        if not re.search(r'^##\s+Requirements', md, re.MULTILINE | re.IGNORECASE):
            errors.append("Missing '## Requirements' section")
            
        # Check for at least one task
        if not re.search(r'^\s*-\s*\[\s*\]\s*Validate\s*`[^`]+`', md, re.MULTILINE): # Path can be complex
            errors.append("No validation tasks found. Format: '- [ ] Validate `path/to/file.py`'")
            
        # Check for checkboxes in requirements
        if re.search(r'^##\s+Requirements', md, re.MULTILINE | re.IGNORECASE):
            req_section = re.search(r'## Requirements\n(.*?)(?=\n##|\Z)', md, re.DOTALL | re.IGNORECASE)
            if req_section and not re.search(r'\[\s*\]', req_section.group(1)):
                errors.append("Requirements should use checkboxes. Format: '1. [ ] Requirement'")
        
        return len(errors) == 0, errors

class TaskConverter:
    """
    Converts Markdown task files into Claude Code MCP compatible JSON format.
    
    This class parses markdown files structured as task definitions and 
    converts them into a standardized JSON format that is compatible with
    the Claude Code MCP requirements. It supports validation tasks and
    ensures the output is properly structured for the MCP server.
    
    Key features:
    - Parses Markdown task files with structured sections
    - Extracts metadata and content for MCP processing
    - Generates well-formatted task prompts for Claude
    - Outputs JSON compatible with Claude Code MCP server
    - Validates markdown structure and provides helpful error messages
    """
    def __init__(self):
        """Initialize the TaskConverter."""
        pass
        
    def validate_mcp_format(self, tasks_data: List[Dict[str, Any]]) -> bool:
        """
        Validate the task data against the expected Claude Code MCP format.
        
        Args:
            tasks_data: List of task dictionaries
            
        Returns:
            True if valid, False otherwise
        """
        # Check if tasks_data is a list
        if not isinstance(tasks_data, list):
            print("Error: MCP tasks_data is not a list", file=sys.stderr) # Print to stderr
            return False
            
        # Check if the list is empty
        if not tasks_data:
            print("Warning: MCP tasks_data is empty", file=sys.stderr) # Print to stderr
            return True # Empty list is valid format-wise
            
        # Check each task for required fields
        for i, task in enumerate(tasks_data):
            # Check required fields
            if 'tool' not in task:
                print(f"Error: Task {i+1} missing required field 'tool'", file=sys.stderr)
                return False
                
            if task['tool'] != 'claude_code':
                print(f"Error: Task {i+1} has incorrect tool value. Expected 'claude_code', got '{task['tool']}'", file=sys.stderr)
                return False
                
            if 'arguments' not in task:
                print(f"Error: Task {i+1} missing required field 'arguments'", file=sys.stderr)
                return False
                
            arguments = task['arguments']
            if not isinstance(arguments, dict):
                print(f"Error: Task {i+1} has invalid 'arguments' type. Expected dict, got {type(arguments)}", file=sys.stderr)
                return False
                
            if 'prompt' not in arguments or not isinstance(arguments['prompt'], str): # Check prompt type
                print(f"Error: Task {i+1} missing/invalid 'arguments.prompt'", file=sys.stderr)
                return False
            
            # --- BEGIN MODIFICATION ---
            # Check for workFolder (added in format_tasks_for_mcp)
            if 'workFolder' not in arguments or not isinstance(arguments['workFolder'], str):
                print(f"Error: Task {i+1} missing/invalid 'arguments.workFolder'", file=sys.stderr)
                return False
            # --- END MODIFICATION ---
                
            # Verify that dangerously_skip_permissions is set to true
            if 'dangerously_skip_permissions' not in arguments:
                print(f"Error: Task {i+1} missing required field 'arguments.dangerously_skip_permissions'", file=sys.stderr)
                return False
                
            if arguments['dangerously_skip_permissions'] is not True:
                print(f"Error: Task {i+1} has incorrect value for 'arguments.dangerously_skip_permissions'. Expected true", file=sys.stderr)
                return False
                
            # Verify timeout is set
            if 'timeout_ms' not in arguments: # This is optional, so a warning is fine
                print(f"Warning: Task {i+1} missing 'arguments.timeout_ms'", file=sys.stderr)
                
        return True

# This function was part of the original, but its logic is now integrated into main() with argparse
# I'm keeping it here commented out for reference to the original structure.
# def convert_tasks(input_file: str, output_file: str) -> bool:
#     """
#     Convert markdown tasks to Claude Code MCP format and save to JSON.
    
#     Args:
#         input_file: Path to the markdown file
#         output_file: Path to save the output JSON file
        
#     Returns:
#         True if successful, False otherwise
#     """
#     try:
#         # Validate input file
#         if not os.path.isfile(input_file):
#             print(f"Error: Input file '{input_file}' does not exist.")
#             return False
        
#         # Make sure the output directory exists
#         output_dir = os.path.dirname(output_file)
#         if output_dir and not os.path.exists(output_dir):
#             os.makedirs(output_dir)
        
#         print(f"Processing '{input_file}' to generate MCP tasks...")
        
#         # Process markdown and generate MCP tasks with progress
#         def progress_print(msg):
#             print(f"[Progress] {msg}")
        
#         # This call would need project_path if we were to use it directly
#         tasks = process_markdown(input_file, progress_callback=progress_print) 
        
#         # Validate MCP format
#         print("\nValidating Claude Code MCP format...")
#         temp_converter = TaskConverter()
#         valid = temp_converter.validate_mcp_format(tasks)
        
#         if not valid:
#             print("\nValidation failed. Please check the format requirements.")
#             return False
#         else:
#             print("Validation successful.")
        
#         with open(output_file, 'w', encoding='utf-8') as f:
#             json.dump(tasks, f, indent=2)
        
#         print(f"\nSuccessfully converted markdown to {len(tasks)} validation tasks")
#         print(f"JSON saved to '{output_file}'")
        
#         return True
#     except Exception as e:
#         print(f"Error during conversion: {str(e)}")
#         return False

# --- BEGIN MODIFICATION ---
def main():
    """Main function to execute the script from command line."""
    parser = argparse.ArgumentParser(description="Convert Markdown validation tasks to Claude Code MCP JSON.")
    parser.add_argument("input_markdown", help="Path to the input markdown file. If relative, it's resolved against --project-path.")
    parser.add_argument("output_json_file", nargs='?', help="Path to the output JSON file. If relative, it's resolved against --project-path. Required if not using --json-output.")
    parser.add_argument("--json-output", action="store_true", help="Output JSON to stdout instead of a file.")
    parser.add_argument("--project-path", required=True, help="Absolute path to the root of the project being processed.")

    args = parser.parse_args()

    project_path = os.path.abspath(args.project_path) # Ensure project_path is absolute

    # Validate project_path (server.ts also does this, but good for standalone script use)
    if not os.path.isdir(project_path):
        print(f"Error: Project path '{project_path}' does not exist or is not a directory.", file=sys.stderr)
        sys.exit(1)

    # Resolve input_markdown path: if not absolute, assume it's relative to project_path
    # This path is already resolved and validated by server.ts when called from there.
    # This logic is for when the script is run standalone.
    input_file_path = args.input_markdown
    if not os.path.isabs(input_file_path):
        input_file_path = os.path.join(project_path, input_file_path)
    
    if not os.path.isfile(input_file_path):
        print(f"Error: Input markdown file '{args.input_markdown}' (resolved to '{input_file_path}') does not exist.", file=sys.stderr)
        sys.exit(1)
            
    try:
        def progress_to_stderr(msg):
            # Send progress messages to stderr so they don't interfere with JSON stdout
            print(f"{msg}", file=sys.stderr) 
        
        tasks = process_markdown(input_file_path, project_path, progress_callback=progress_to_stderr)
        
        # Validate MCP format before outputting
        temp_converter = TaskConverter() # Instantiate for validation
        if not temp_converter.validate_mcp_format(tasks):
            # Errors from validate_mcp_format already go to stderr
            print("\nValidation of generated MCP tasks failed. Please check errors above.", file=sys.stderr)
            sys.exit(1)

        if args.json_output:
            print(json.dumps(tasks, indent=2))
        else:
            if not args.output_json_file:
                print("Error: Output JSON file path is required when not using --json-output.", file=sys.stderr)
                sys.exit(1)
            
            output_file = args.output_json_file
            # If output_file is relative, resolve it against project_path
            if not os.path.isabs(output_file):
                output_file = os.path.join(project_path, output_file)

            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True) # exist_ok=True to avoid error if dir exists

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(tasks, f, indent=2)
            # Send success message to stderr as well if not in json_output_mode
            print(f"Successfully converted to {len(tasks)} tasks. JSON saved to '{output_file}'", file=sys.stderr)
        
        sys.exit(0)

    except ValueError as ve: # Catch specific ValueError from process_markdown for format issues
        print(f"Error during conversion (Markdown Format): {str(ve)}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error during conversion (General): {str(e)}", file=sys.stderr)
        sys.exit(1)
# --- END MODIFICATION ---

if __name__ == "__main__":
    main()