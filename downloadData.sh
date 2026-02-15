#!/bin/bash

DATA_DIR="data"
ASSETS_FILE="assets.txt"

mkdir -p "$DATA_DIR"

echo "Starting download of assets from $ASSETS_FILE..."

if [ ! -f "$ASSETS_FILE" ]; then
    echo "Error: $ASSETS_FILE not found."
    exit 1
fi

while IFS= read -r url || [ -n "$url" ]; do
    [[ -z "$url" || "$url" =~ ^# ]] && continue
    filename=$(basename "${url%%\?*}")
    
    echo "Downloading $filename..."

    curl -L -sS -A "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36" -o "$DATA_DIR/$filename" "$url"
    
    if [ $? -eq 0 ]; then
        echo "Successfully downloaded $filename"
    else
        echo "Failed to download $filename"
    fi
done < "$ASSETS_FILE"

echo "All downloads completed."
