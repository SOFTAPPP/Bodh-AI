import os
import tokenize

def remove_comments_from_file(filepath):
    try:
        with open(filepath, 'rb') as f:
            tokens = list(tokenize.tokenize(f.readline))
        
        out = ""
        last_lineno = -1
        last_col = 0
        
        for tok in tokens:
            token_type = tok.type
            token_string = tok.string
            start_line, start_col = tok.start
            end_line, end_col = tok.end
            
            # tokenize includes an ENCODING token at the beginning
            if token_type == tokenize.ENCODING:
                continue
                
            if start_line > last_lineno:
                last_col = 0
            
            if start_col > last_col:
                out += " " * (start_col - last_col)
                
            if token_type == tokenize.COMMENT:
                # Skip adding the comment token string
                pass
            else:
                out += token_string
                
            last_lineno = end_line
            last_col = end_col
            
        final_lines = []
        for line in out.splitlines():
            final_lines.append(line.rstrip())
            
        compressed_lines = []
        for line in final_lines:
            if line == "":
                if not compressed_lines or compressed_lines[-1] != "":
                    compressed_lines.append(line)
            else:
                compressed_lines.append(line)
                
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(compressed_lines) + "\n")
            
        return True
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

if __name__ == "__main__":
    app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
    count = 0
    for root, dirs, files in os.walk(app_dir):
        # Exclude pycache
        if "__pycache__" in root:
            continue
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                if remove_comments_from_file(path):
                    print(f"Cleaned: {os.path.relpath(path, app_dir)}")
                    count += 1
    print(f"Successfully removed comments from {count} files in the app directory.")
