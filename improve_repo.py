import os
import sys
import argparse
import asyncio
import subprocess
import shutil
from typing import List, Optional, Dict
from google import genai
from google.genai import types

# Constants
DEFAULT_MODEL = "gemini-3.1-pro-preview"
CONTEXT_LIMIT = 1_000_000  # Conservative estimate for alerting

class Repoimprover:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, root_path: str = "."):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.root_path = os.path.abspath(root_path)
        self.repo_context = ""
        self.total_tokens = 0
        self.backups: Dict[str, str] = {}

    def backup_file(self, relative_path: str):
        full_path = os.path.join(self.root_path, relative_path)
        if relative_path not in self.backups and os.path.exists(full_path):
            with open(full_path, 'r', encoding='utf-8') as f:
                self.backups[relative_path] = f.read()

    def restore_all(self):
        for rel_path, content in self.backups.items():
            full_path = os.path.join(self.root_path, rel_path)
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
        self.backups.clear()

    def apply_change(self, relative_path: str, new_content: str):
        self.backup_file(relative_path)
        full_path = os.path.join(self.root_path, relative_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

    async def run_command(self, command: str) -> str:
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.root_path
            )
            stdout, stderr = await process.communicate()
            return f"STDOUT:\n{stdout.decode()}\nSTDERR:\n{stderr.decode()}\nRETURN CODE: {process.returncode}"
        except Exception as e:
            return f"ERROR running command: {e}"

    def read_repo(self, path: str, skip_hidden: bool = True) -> str:
        context = []
        for root, dirs, files in os.walk(path):
            if skip_hidden:
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                files = [f for f in files if not f.startswith('.')]

            for file in files:
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        relative_path = os.path.relpath(file_path, path)
                        context.append(f"--- FILE: {relative_path} ---\n{content}\n")
                except Exception as e:
                    print(f"Skipping {file_path}: {e}")

        full_context = "\n".join(context)

        try:
            resp = self.client.models.count_tokens(model=self.model, contents=full_context)
            self.total_tokens = resp.total_tokens
        except:
            self.total_tokens = len(full_context) // 4

        if self.total_tokens > CONTEXT_LIMIT * 0.9:
            print(f"WARNING: Approaching context limit ({self.total_tokens} tokens)")

        self.repo_context = full_context
        return full_context

    async def run(self, user_prompt: str):
        print(f"Analyzing repository with prompt: {user_prompt}")

        system_instruction = """You are a repository improvement agent using Gemini 3.
You have access to the entire repository content. Your goal is to solve the problem described in the user prompt.

You can perform the following actions by using a specific tag format in your response:
1. <THOUGHT>: Your internal reasoning or "passing thoughts". These will be streamed to the user.
2. <CLARIFY>: Ask the user for clarification.
3. <EDIT path="relative/path/to/file">: Provide the full new content for a file.
4. <RUN>: Run a shell command (e.g., to run tests).
5. <SOLVED>: Signal that the task is complete. Provide a summary of your changes.
6. <IMPOSSIBLE>: Signal that the task cannot be completed. Explain why.
7. <ROLLBACK>: Restore all files to their state before the last set of edits.

Rules:
- You should always output <THOUGHT> blocks to explain what you are doing.
- When you use <EDIT>, the system will backup the file first.
- When you use <RUN>, the system will return the output of the command.
- IMPORTANT: If you provide both an <EDIT> and a <RUN> (e.g. for a test), the system will check the return code of the <RUN> command. If it fails, the <EDIT> will be automatically ROLLED BACK.
- If you need input, use <CLARIFY>.
- You can provide multiple <THOUGHT> blocks.
- Your goal is to provide PASSING THOUGHTS. Be verbose with your reasoning.

Format for Action Responses:
When you use <EDIT> or <RUN>, the system will provide the result in the next turn.

Example:
<THOUGHT> I will start by checking the existing tests.
<RUN> pytest
"""
        history = [
            {"role": "user", "parts": [f"System Instruction: {system_instruction}\n\nRepository Context:\n{self.repo_context}\n\nUser Prompt: {user_prompt}"]}
        ]

        while True:
            response_stream = self.client.models.generate_content_stream(
                model=self.model,
                contents=history,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                )
            )

            full_response = ""
            current_thought = ""

            in_thought = False
            async for chunk in response_stream:
                text = chunk.text
                full_response += text

                # Highlight thoughts during streaming
                if "<THOUGHT>" in text and not in_thought:
                    sys.stdout.write("\033[94m[THOUGHT]\033[0m ")
                    in_thought = True
                if "</THOUGHT>" in text and in_thought:
                    in_thought = False

                sys.stdout.write(text)
                sys.stdout.flush()

            history.append({"role": "model", "parts": [full_response]})

            # Parse actions from full_response
            if "<SOLVED>" in full_response:
                print("\nTask marked as SOLVED.")
                break
            if "<IMPOSSIBLE>" in full_response:
                print("\nTask marked as IMPOSSIBLE.")
                break

            if "<ROLLBACK>" in full_response:
                print("\nRolling back all changes...")
                self.restore_all()
                history.append({"role": "user", "parts": ["All changes have been rolled back to their original state."]})
                continue

            # Handle EDIT and RUN in parallel where possible
            import re
            edits = re.findall(r'<EDIT path="(.*?)">(.*?)</EDIT>', full_response, re.DOTALL)
            runs = re.findall(r'<RUN>(.*?)</RUN>', full_response, re.DOTALL)

            action_tasks = []

            # Edits are usually fast and sequential is safer for now, but we can apply them quickly
            for path, content in edits:
                print(f"\nApplying edit to {path}...")
                self.apply_change(path, content)

            # Commands can definitely be run in parallel
            for cmd in runs:
                print(f"\nQueueing command: {cmd}")
                action_tasks.append(self.run_command(cmd))

            run_results = []
            if action_tasks:
                print(f"Executing {len(action_tasks)} commands in parallel...")
                run_results = await asyncio.gather(*action_tasks)

            result_parts = []
            if edits or runs:
                if edits:
                    result_parts.append(f"Applied {len(edits)} edits.")
                for i, res in enumerate(run_results):
                    result_parts.append(f"Command: {runs[i]}\nResult: {res}")

                # Check for automatic rollback if tests failed
                # If any RUN failed and there were edits, we might want to rollback
                # This is a bit complex for multiple runs, but let's assume if any RUN fails, it's a regression
                if edits and any("RETURN CODE: 0" not in res for res in run_results):
                    print("\nTest failed. Rolling back edits for this turn...")
                    # We need a more granular rollback for edits in THIS turn only
                    # For simplicity, we'll restore all since our backup is currently all-or-nothing
                    self.restore_all()
                    result_parts.append("CRITICAL: A command failed. All edits from this turn have been rolled back.")

            # Handle CLARIFY without blocking other actions
            if "<CLARIFY>" in full_response:
                # Non-blocking input
                user_input = await asyncio.to_thread(input, "\n[CLARIFICATION REQUESTED] Please respond: ")
                result_parts.append(f"User Clarification: {user_input}")

            if result_parts:
                history.append({"role": "user", "parts": ["\n\n".join(result_parts)]})
            else:
                # If no explicit action but not solved, maybe it's just thinking or forgot tags
                if not any(tag in full_response for tag in ["<SOLVED>", "<IMPOSSIBLE>", "<CLARIFY>", "<EDIT>", "<RUN>"]):
                     history.append({"role": "user", "parts": ["Please continue or provide an action (EDIT, RUN, CLARIFY, SOLVED, IMPOSSIBLE)."]})

async def main():
    parser = argparse.ArgumentParser(description="Improve a repository using Gemini 3")
    parser.add_argument("prompt", help="The improvement prompt")
    parser.add_argument("--path", default=".", help="Path to the repository (default: current directory)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Gemini model to use (default: {DEFAULT_MODEL})")

    args = parser.parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        sys.exit(1)

    improver = Repoimprover(api_key, model=args.model)
    improver.read_repo(args.path)
    await improver.run(args.prompt)

if __name__ == "__main__":
    asyncio.run(main())
