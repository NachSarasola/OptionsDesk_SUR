import os
import zipfile

zip_filename = "update.zip"
print(f"Creating {zip_filename}...")

exclude_dirs = {
    ".git", ".claude", ".playwright-mcp", ".pytest_cache", ".ruff_cache",
    "venv", "recovered_writes", "extracted_blocks", "dashboard_patches",
    "tab_stocks_extracted_versions", "mentions", "logs", "__pycache__"
}

with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk("."):
        # Filter out directories in-place to prevent os.walk from entering them
        dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith("tmp")]
        
        # Skip pycache directories
        if "__pycache__" in root or ".git" in root or "venv" in root:
            continue
            
        for file in files:
            # Skip output zip, temporary/backup files, and local logs/data if we want a clean upload
            # But wait, should we skip the local data/ folder?
            # Yes, we want to skip the local data/ folder because the VPS has its own production data,
            # and we don't want to overwrite the VPS data with local demo paper trades!
            # Wait, let's skip the local data/ folder. Yes, that is extremely important!
            if file == zip_filename or file.endswith(".log") or file.endswith(".tmp") or file.startswith("recovered_blob_"):
                continue
            parts = root.split(os.sep)
            if len(parts) > 1 and parts[1] == "data":
                # Skip files inside root data/ folder to preserve remote production data
                continue
                
            file_path = os.path.join(root, file)
            # Make sure we don't pack scripts we just used for search or cleaning
            if file in [
                "create_update_archive.py", "extract_mention_codes.py", "find_full_func.py",
                "find_subheader.py", "find_dangling_stocks.py", "find_dashboard_edits.py",
                "find_dashboard_patches.py", "extract_raw_tab_stocks.py", "extract_transcript_stocks.py",
                "extract_step_1343.py", "find_diff_stocks.py", "get_git_diff.py", "find_dashboard_writes.py",
                "extract_by_line_number.py", "get_git_diff.py"
            ]:
                continue
                
            archive_name = os.path.relpath(file_path, ".")
            zipf.write(file_path, archive_name)
            print(f"  Added: {archive_name}")

print("Zip archive created successfully!")
