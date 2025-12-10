#!/bin/bash

# Usage:
# ./run_batch.sh <directory> <learning-type> [--with-smt2] [--rust-only] [--merge <n>] [--annotated]

# Do NOT stop on errors
set +e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <directory> <learning-type> [optional flags]"
    echo "Optional flags: --with-smt2 --rust-only --merge <n> --annotated"
    exit 1
fi

DIR="$1"
LEARNING_TYPE="$2"
shift 2  # remove required args so $@ now contains only optionals

OPTIONAL_ARGS=()
ANNOTATED_FLAG=false
WITH_SMT2_FLAG=false
WITH_CRUX_FLAG=false

# Parse optional flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-smt2)
            OPTIONAL_ARGS+=("--with-smt2")
            WITH_SMT2_FLAG=true
            shift
            ;;
        --with-crux)
            OPTIONAL_ARGS+=("--with-crux")
            WITH_CRUX_FLAG=true
            shift
            ;;
        --rust-only)
            OPTIONAL_ARGS+=("--rust-only")
            shift
            ;;
        --annotated)
            OPTIONAL_ARGS+=("--annotated")
            ANNOTATED_FLAG=true
            shift
            ;;
        --merge)
            OPTIONAL_ARGS+=("--merge" "$2")
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Iterate through all .rs files inside the directory
for FILE in "$DIR"/*.rs; do
    if [ -f "$FILE" ]; then
        BASENAME=$(basename "$FILE")

        if [ "$WITH_CRUX_FLAG" = true ] && [ "$WITH_SMT2_FLAG" = true ] && [ "$ANNOTATED_FLAG" = true ]; then
            OUTFILE="${BASENAME%.rs}_verified_crux_smt2_verified.rs"
        elif [ "$WITH_CRUX_FLAG" = true ] && [ "$WITH_SMT2_FLAG" = true ]; then
            OUTFILE="${BASENAME%.rs}_verified_crux_smt2.rs"
        elif [ "$WITH_CRUX_FLAG" = true ] && [ "$ANNOTATED_FLAG" = true ]; then
            OUTFILE="${BASENAME%.rs}_verified_crux_annotated.rs"
        elif [ "$WITH_SMT2_FLAG" = true ] && [ "$ANNOTATED_FLAG" = true ]; then
            OUTFILE="${BASENAME%.rs}_verified_smt2_annotated.rs"
        elif [ "$WITH_CRUX_FLAG" = true ]; then
            OUTFILE="${BASENAME%.rs}_verified_crux.rs"
        elif [ "$WITH_SMT2_FLAG" = true ]; then
            OUTFILE="${BASENAME%.rs}_verified_smt2.rs"
        elif [ "$ANNOTATED_FLAG" = true ]; then
            OUTFILE="${BASENAME%.rs}_verified_annotated.rs"
        else
            OUTFILE="${BASENAME%.rs}_verified.rs"
        fi

        echo "==========================================="
        echo "Processing: $FILE"
        echo "Output will be: $OUTFILE"
        echo "Using extra flags: ${OPTIONAL_ARGS[*]}"
        echo "==========================================="

        # Run the python script but do NOT stop execution if it errors
        python main.py \
            --input "$FILE" \
            --output "$OUTFILE" \
            --config config-artifact-openai.json \
            --learning-type "$LEARNING_TYPE" \
            "${OPTIONAL_ARGS[@]}"

        # If Python exits with non-zero code, show message but continue
        if [ $? -ne 0 ]; then
            echo "❌ Error processing file: $FILE"
            echo "   Continuing to next file..."
        fi
    fi
done

echo "✔ Batch processing completed."
