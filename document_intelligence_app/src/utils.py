# src/utils.py
import os

def sanitize_filename_for_pinecone(filename: str) -> str:
    """Sanitizes a filename to be suitable for use as a Pinecone index name."""
    name_without_ext = os.path.splitext(filename)[0]
    lower_name = name_without_ext.lower()
    # Pinecone index names typically allow lowercase, hyphens, and numbers.
    # Replace non-alphanumeric characters with hyphens and remove leading/trailing hyphens.
    sanitized_name = ''.join(c if c.isalnum() else '-' for c in lower_name)
    sanitized_name = sanitized_name.strip('-')
    # Ensure it's not too long and meets Pinecone's specific rules if any
    return sanitized_name[:45] # Pinecone max index name length is 45 characters