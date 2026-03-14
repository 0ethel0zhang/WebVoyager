import os
import sys
import argparse
import asyncio
import subprocess
import re
from typing import List, Optional, Dict, Any
import google.generativeai as genai
from google.generativeai.types import content_types

# Constants
DEFAULT_MODEL = "gemini-2.0-pro-exp-02-05"
CONTEXT_LIMIT = 1_000_000

class Repoimprover:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, root_path: str = "."):
        # Using transport='rest' to avoid common gRPC illegal header issues
        genai.configure(api_key=api_key, transport='rest')
        self.model_name = model
        self.root_path = os.path.abspath(root_path)
        self.repo_context = ""
        self.total_tokens = 0
        self.backups: Dict[str, str] = {}
        self.created_files: List[str] = []

    def _validate_path(self, relative_path: str) -> str:
        full_path = os.path.abspath(os.path.join(self.root_path, relative_path))
        if not full_path.startswith(self.root_path):
            raise ValueError(f"Path traversal detected: {relative_path} is outside {self.root_path}")
        return full_path

    def backup_file(self, relative_path: str):
        full_path = self._validate_path(relative_path)
        if relative_path not in self.backups and os.path.exists(full_path):
            with open(full_path, 'r', encoding='utf-8') as f:
                self.backups[relative_path] = f.read()
        elif not os.path.exists(full_path):
            if relative_path not in self.created_files:
                self.created_files.append(relative_path)

    def restore_all(self):
        for rel_path, content in self.backups.items():
            full_path = self._validate_path(rel_path)
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)

        for rel_path in self.created_files:
            full_path = self._validate_path(rel_path)
            if os.path.exists(full_path):
                os.remove(full_path)

        self.backups.clear()
        self.created_files.clear()

    def commit_changes(self):
        self.backups.clear()
        self.created_files.clear()

    def apply_change(self, relative_path: str, new_content: str):
        full_path = self._validate_path(relative_path)
        self.backup_file(relative_path)
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

        model = genai.GenerativeModel(self.model_name)
        try:
            self.total_tokens = model.count_tokens(full_context).total_tokens
        except:
            self.total_tokens = len(full_context) // 4

        if self.total_tokens > CONTEXT_LIMIT * 0.9:
            print(f"WARNING: Approaching context limit ({self.total_tokens} tokens)")

        self.repo_context = full_context
        return full_context

    async def run(self, user_prompt: str, screenshot_paths: List[str] = None):
        print(f"Analyzing repository with prompt: {user_prompt}")

        system_instruction = """You are a repository improvement agent using Gemini 3.
You have access to the entire repository content. Your goal is to solve the problem described in the user prompt.

You can perform the following actions by using a specific tag format in your response:
1. <THOUGHT>Your internal reasoning or "passing thoughts".</THOUGHT>
2. <CLARIFY>Ask the user for clarification.</CLARIFY>
3. <EDIT path="relative/path/to/file">Full new content for the file</EDIT>
4. <RUN>shell command</RUN>
5. <SOLVED>Task summary</SOLVED>
6. <IMPOSSIBLE>Explanation</IMPOSSIBLE>
7. <ROLLBACK /> (Restores all files modified in the current turn)

Rules:
- You should always output <THOUGHT> blocks to explain what you are doing.
- When you use <EDIT>, the system will backup the file first.
- When you use <RUN>, the system will return the output of the command.
- IMPORTANT: If you provide both an <EDIT> and a <RUN> (e.g. for a test), the system will check the return code of the <RUN> command. If it fails, ALL <EDIT>s from the current turn will be automatically ROLLED BACK.
- If you need input, use <CLARIFY>.
- You can provide multiple <THOUGHT> blocks.
- Your goal is to provide PASSING THOUGHTS. Be verbose with your reasoning.

Example:
<THOUGHT>I will start by checking the existing tests.</THOUGHT>
<RUN>pytest</RUN>
"""
        model = genai.GenerativeModel(self.model_name, system_instruction=system_instruction)

        initial_content = [f"Repository Context:\n{self.repo_context}\n\nUser Prompt: {user_prompt}"]
        if screenshot_paths:
            print(f"Including {len(screenshot_paths)} screenshots in context.")
            for path in screenshot_paths:
                try:
                    # google-generativeai expects PIL Image or bytes with mime type
                    import PIL.Image
                    img = PIL.Image.open(path)
                    initial_content.append(img)
                except Exception as e:
                    print(f"Error loading screenshot {path}: {e}")

        chat = model.start_chat(history=[])
        current_user_msg = "\n".join([c if isinstance(c, str) else "[Image Content]" for c in initial_content])
        # We need to send images in the first message
        first_msg_parts = initial_content

        while True:
            if first_msg_parts:
                response_stream = await chat.send_message_async(first_msg_parts, stream=True)
                first_msg_parts = None
            else:
                response_stream = await chat.send_message_async(current_user_msg, stream=True)

            full_response = ""
            streaming_buffer = ""
            in_thought = False

            async for chunk in response_stream:
                text = chunk.text
                full_response += text
                streaming_buffer += text

                if "<THOUGHT>" in streaming_buffer and not in_thought:
                    sys.stdout.write("\033[94m[THOUGHT]\033[0m ")
                    in_thought = True
                if "</THOUGHT>" in streaming_buffer and in_thought:
                    in_thought = False

                sys.stdout.write(text)
                sys.stdout.flush()
                streaming_buffer = streaming_buffer[-20:]

            if "<ROLLBACK" in full_response:
                print("\nRolling back changes from this turn...")
                self.restore_all()
                current_user_msg = "The changes from the previous turn have been rolled back."
                continue

            edits = re.findall(r'<EDIT path="(.*?)">(.*?)</EDIT>', full_response, re.DOTALL)
            runs = re.findall(r'<RUN>(.*?)</RUN>', full_response, re.DOTALL)

            action_tasks = []
            for path, content in edits:
                print(f"\nApplying edit to {path}...")
                try:
                    self.apply_change(path, content)
                except ValueError as e:
                    print(f"Error: {e}")

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

                if edits and any("RETURN CODE: 0" not in res for res in run_results):
                    print("\nTest failed. Rolling back edits for this turn...")
                    self.restore_all()
                    result_parts.append("CRITICAL: A command failed. All edits from this turn have been rolled back.")
                else:
                    self.commit_changes()

            clarifications = re.findall(r'<CLARIFY>(.*?)</CLARIFY>', full_response, re.DOTALL)
            for clarification in clarifications:
                user_input = await asyncio.to_thread(input, f"\n[CLARIFICATION REQUESTED: {clarification.strip()}] Please respond: ")
                result_parts.append(f"User Clarification for '{clarification.strip()}': {user_input}")

            if "<SOLVED>" in full_response:
                print("\nTask marked as SOLVED.")
                break
            if "<IMPOSSIBLE>" in full_response:
                print("\nTask marked as IMPOSSIBLE.")
                break

            if result_parts:
                current_user_msg = "\n\n".join(result_parts)
            else:
                if not any(tag in full_response for tag in ["<SOLVED>", "<IMPOSSIBLE>", "<CLARIFY>", "<EDIT>", "<RUN>"]):
                     current_user_msg = "Please continue or provide an action (EDIT, RUN, CLARIFY, SOLVED, IMPOSSIBLE)."

async def main():
    parser = argparse.ArgumentParser(description="Improve a repository using Gemini 3")
    parser.add_argument("prompt", help="The improvement prompt")
    parser.add_argument("--path", default=".", help="Path to the repository (default: current directory)")
    parser.add_argument("--model", default="gemini-2.0-pro-exp-02-05", help=f"Gemini model to use")
    parser.add_argument("--screenshots", nargs="+", help="Paths to screenshots to include as context")

    args = parser.parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        sys.exit(1)

    # Strip whitespace/newlines and any non-printable chars which can cause gRPC "Illegal header value" errors
    api_key = "".join(c for c in api_key if c.isprintable()).strip()

    improver = Repoimprover(api_key, model=args.model, root_path=args.path)
    improver.read_repo(args.path)
    await improver.run(args.prompt, screenshot_paths=args.screenshots)

if __name__ == "__main__":
    asyncio.run(main())
