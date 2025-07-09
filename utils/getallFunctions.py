import os
import ast
from typing import Dict, List

def get_all_functions_in_folder(folder_path: str) -> Dict[str, List[str]]:
    """
    Returns a dictionary mapping each Python file in the folder to a list of its top-level function names.
    Only searches the top-level of the folder (no recursion).
    """
    functions_dict = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.py'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
            try:
                tree = ast.parse(file_content)
                functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                functions_dict[filename] = functions
            except Exception as e:
                functions_dict[filename] = [f"Error parsing file: {e}"]
    return functions_dict
