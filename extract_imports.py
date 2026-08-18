import ast
import os

stdlib = set(__import__('sys').builtin_module_names)
# also adding common stdlib modules
import sysconfig
std_lib_dir = sysconfig.get_path('stdlib')
for f in os.listdir(std_lib_dir):
    if f.endswith('.py'): stdlib.add(f[:-3])
    elif os.path.isdir(os.path.join(std_lib_dir, f)): stdlib.add(f)

imports = set()
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.py') and file != 'extract_imports.py':
            path = os.path.join(root, file)
            with open(path, 'r', encoding='utf-8') as f:
                try:
                    tree = ast.parse(f.read(), filename=path)
                except SyntaxError:
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.add(alias.name.split('.')[0])
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.add(node.module.split('.')[0])

# remove local imports by checking if there is a .py file or folder with that name
local_modules = set()
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.py'):
            local_modules.add(file[:-3])
    for d in dirs:
        local_modules.add(d)

external_imports = imports - stdlib - local_modules
print("External imports:")
print(sorted(list(external_imports)))
