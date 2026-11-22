#!/usr/bin/env bash
# ==============================================================================
# setup_agent_tools.sh
#
# Installs everything needed for the lab automation / agent pipeline:
#   - Python virtual environment + core packages (ollama client, isatools,
#     pandas, openpyxl, requests)
#   - Ollama itself + the qwen2.5:14b model
#   - mermaid-cli (for rendering flowchart syntax to images/SVG)
#   - Core ontology files (CL, GO, ChEBI, EFO, OBI, NCBITaxon) as OWL/OBO
#   - Cellosaurus flat file
#   - PubMed baseline + daily update files (optional — large, confirms first)
#
# Everything downloaded here is stored locally under ~/lab-agent-pipeline/
# and works fully offline once downloaded.
#
# Run with:  bash setup_agent_tools.sh
# ==============================================================================

# "set -e" tells bash to stop the whole script immediately if any single
# command fails (returns a non-zero exit code). Without this, a failed
# download or install could go unnoticed and the script would just keep
# going with missing pieces.
set -e

# --- Define all the folder paths we'll use, up front, as variables -----------
# Using variables instead of typing the full path every time means if you
# ever want to change where things get installed, you only edit these lines.
PROJECT_DIR="$HOME/lab-agent-pipeline"        # top-level folder for everything
DATA_DIR="$PROJECT_DIR/data/ontologies"       # where ontology files go
PUBMED_DIR="$PROJECT_DIR/data/pubmed"         # top-level PubMed folder
BASELINE_DIR="$PUBMED_DIR/baseline"           # annual full-snapshot files
UPDATES_DIR="$PUBMED_DIR/updatefiles"         # daily delta files
VENV_DIR="$PROJECT_DIR/venv"                  # Python virtual environment

echo "=== 1. Creating project folders ==="
# "mkdir -p" creates a folder AND any missing parent folders needed to get
# there, and does NOT error out if the folder already exists (unlike plain
# mkdir). That's why it's safe to re-run this whole script multiple times.
mkdir -p "$PROJECT_DIR" "$DATA_DIR" "$PUBMED_DIR" "$BASELINE_DIR" "$UPDATES_DIR"
cd "$PROJECT_DIR"   # move into the project folder so relative paths below make sense

# ------------------------------------------------------------------------
echo "=== 2. Setting up Python virtual environment ==="
# A "virtual environment" is an isolated folder holding its own copy of
# Python packages, separate from your system-wide Python. This keeps the
# packages this project needs from clashing with packages other projects
# on your machine might need in different versions.
if [ ! -d "$VENV_DIR" ]; then
    # Only create it if it doesn't already exist — otherwise re-running
    # this script would wipe out a venv you've already been using.
    python3 -m venv "$VENV_DIR"
fi
# "source" runs the venv's activation script in your CURRENT shell, which
# changes your shell so that "python" and "pip" now point at the venv's
# copies instead of the system ones. This only lasts for this terminal
# session/script run — you'll need to "source" it again next time you
# open a new terminal and want to use this environment.
source "$VENV_DIR/bin/activate"

echo "=== 3. Installing Python packages ==="
pip install --upgrade pip   # updates pip itself first, to avoid install issues with older pip versions

# Each package below, and why it's here:
#   ollama    - Python client for talking to your local Ollama server (running qwen2.5:14b)
#   isatools  - library for building/validating ISA-JSON (your structured protocol output format)
#   pandas    - dataframes, used for any tabular data wrangling (e.g. before Excel export)
#   openpyxl  - lets pandas read/write .xlsx files
#   requests  - simple HTTP calls, useful for any custom API/download logic you write later
#   pyyaml    - read/write YAML config files, if you end up using them for agent configs
pip install ollama isatools pandas openpyxl requests pyyaml

# ------------------------------------------------------------------------
echo "=== 4. Checking for Ollama ==="
# "command -v ollama" checks if the "ollama" command exists on your PATH.
# It prints nothing and returns a non-zero exit code if it's not found,
# which is what "! command -v ollama &> /dev/null" is testing for.
# "&> /dev/null" throws away both normal output and error output, since
# we only care about whether it succeeded or failed, not what it printed.
if ! command -v ollama &> /dev/null; then
    echo "Ollama not found. Installing..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # "$OSTYPE" is a built-in bash variable describing the OS.
        # It starts with "darwin" on macOS.
        brew install ollama
    else
        # On Linux, Ollama provides an official install script you can
        # pipe straight into a shell. "curl -fsSL" fetches it quietly
        # (-f fail silently on HTTP errors, -s no progress bar,
        # -S show errors if -s would hide them, -L follow redirects).
        curl -fsSL https://ollama.com/install.sh | sh
    fi
else
    echo "Ollama already installed."
fi

echo "=== 5. Pulling qwen2.5:14b model ==="
# "ollama pull" downloads the model weights to Ollama's local storage
# (this is separate from anything else in this script — Ollama manages
# its own model cache location on disk).
ollama pull qwen2.5:14b

# ------------------------------------------------------------------------
echo "=== 6. Installing mermaid-cli (for rendering flowchart syntax) ==="
# mermaid-cli (the "mmdc" command) takes the Mermaid text syntax we talked
# about (flowchart TD, etc.) and renders it into an actual image/SVG file
# from the command line — useful if you want rendered diagrams outside of
# a markdown viewer.
if ! command -v mmdc &> /dev/null; then
    if command -v npm &> /dev/null; then
        # "-g" installs it globally so the "mmdc" command is available
        # anywhere on your system, not just inside one project folder.
        npm install -g @mermaid-js/mermaid-cli
    else
        echo "WARNING: npm not found — skipping mermaid-cli install."
        echo "Install Node.js first (https://nodejs.org), then run:"
        echo "  npm install -g @mermaid-js/mermaid-cli"
    fi
else
    echo "mermaid-cli already installed."
fi

# ------------------------------------------------------------------------
echo "=== 7. Downloading core ontologies ==="

# An "associative array" in bash is a set of key-value pairs — here, each
# key is the filename we want to save locally, and each value is the URL
# to download it from. This lets us loop over all five ontologies with
# one block of code instead of repeating the same download logic 5 times.
declare -A ONTOLOGIES=(
    ["cl.owl"]="http://purl.obolibrary.org/obo/cl.owl"        # Cell Ontology
    ["go.owl"]="http://purl.obolibrary.org/obo/go.owl"        # Gene Ontology
    ["chebi.owl"]="http://purl.obolibrary.org/obo/chebi.owl"  # Chemical Entities of Biological Interest
    ["efo.owl"]="https://github.com/EBISPOT/efo/releases/latest/download/efo.owl"  # Experimental Factor Ontology
    ["obi.owl"]="http://purl.obolibrary.org/obo/obi.owl"      # Ontology for Biomedical Investigations
)

# "${!ONTOLOGIES[@]}" gives you all the KEYS of the array (the filenames).
# For each one, we look up its matching URL and download it.
for filename in "${!ONTOLOGIES[@]}"; do
    url="${ONTOLOGIES[$filename]}"      # look up the URL for this filename
    target="$DATA_DIR/$filename"        # where we'll save it locally

    if [ -f "$target" ]; then
        # "-f" checks if a regular file already exists at that path.
        # If it does, skip re-downloading it — saves time on re-runs.
        echo "  $filename already present, skipping."
    else
        echo "  Downloading $filename ..."
        # "curl -L" follows redirects (some of these URLs redirect to a
        # different final location), "-o" writes the output to our target
        # filename instead of printing it to the terminal.
        # "|| echo ..." means: if the curl command fails, don't stop the
        # whole script (set -e would normally do that) — just print a
        # warning and move on to the next file.
        curl -L -o "$target" "$url" || echo "  WARNING: failed to download $filename — check URL/network."
    fi
done

# NCBI Taxonomy is handled separately because it's much bigger and only
# available as a single combined file rather than fitting the same
# filename/URL pattern as the others.
echo "  Downloading NCBITaxon (this one is big, ~500MB-1GB)..."
curl -L -o "$DATA_DIR/ncbitaxon.owl.gz" \
    "http://purl.obolibrary.org/obo/ncbitaxon/ncbitaxon.owl.gz" \
    || echo "  WARNING: failed to download NCBITaxon — check URL/network."

# ------------------------------------------------------------------------
echo "=== 8. Downloading Cellosaurus ==="
# Cellosaurus is distributed as a single flat text file (not OWL/OBO like
# the ontologies above), so it gets its own simple curl call.
curl -L -o "$DATA_DIR/cellosaurus.txt" \
    "https://ftp.expasy.org/databases/cellosaurus/cellosaurus.txt" \
    || echo "  WARNING: failed to download Cellosaurus — check URL/network."

# ------------------------------------------------------------------------
echo ""
echo "=== 9. PubMed baseline + updates (large — ~40-50GB compressed) ==="
# "df -h" shows disk free space in human-readable form (GB/MB instead of
# raw bytes). "tail -1" trims it down to just the line for your home
# folder's disk, so you're not scrolling through every mounted drive.
df -h "$HOME" | tail -1

# "read -p" prompts the user with a message and waits for keyboard input,
# storing whatever they type into the CONFIRM_PUBMED variable. This is
# the checkpoint before committing to a huge, slow download.
read -p "Download PubMed baseline + update files now? (y/n) " CONFIRM_PUBMED

if [[ "$CONFIRM_PUBMED" == "y" ]]; then
    echo "--- Downloading PubMed baseline ---"
    # wget flags explained:
    #   -r             recursive — follow links to download a whole folder, not just one file
    #   -np            "no parent" — don't wander up to parent directories
    #   -nH            "no host directory" — don't recreate the ftp.ncbi.nlm.nih.gov folder locally
    #   --cut-dirs=2   strips the first 2 folder levels (pubmed/baseline) from the saved path,
    #                  so files land directly in our target folder instead of nested deep inside it
    #   -c             "continue" — if this gets interrupted, re-running resumes instead of restarting
    #   -A "..."       only download files matching these patterns (the actual data + checksum files),
    #                  ignoring any other clutter on the page (like index.html)
    #   -P             the folder to save everything into
    wget -r -np -nH --cut-dirs=2 -c \
        -A "*.xml.gz,*.xml.gz.md5" \
        -P "$BASELINE_DIR" \
        "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"

    echo "--- Verifying baseline checksums ---"
    # NCBI provides a .md5 checksum file alongside each data file, so you
    # can confirm the download wasn't corrupted in transit. We loop over
    # every downloaded .xml.gz file and compare its actual hash against
    # the expected hash from the matching .md5 file.
    cd "$BASELINE_DIR"
    for f in *.xml.gz; do
        if [ -f "$f.md5" ]; then
            # "grep -o '[a-f0-9]\{32\}'" pulls just the 32-character hex
            # hash out of the .md5 file's text (which also contains the
            # filename and some formatting we don't need).
            expected=$(grep -o '[a-f0-9]\{32\}' "$f.md5")
            # "md5sum" computes the actual hash of the file we downloaded;
            # "awk '{print $1}'" grabs just the hash from md5sum's output
            # (which normally prints "hash  filename").
            actual=$(md5sum "$f" | awk '{print $1}')
            if [ "$expected" != "$actual" ]; then
                echo "  MISMATCH: $f — consider re-downloading this file"
            fi
        fi
    done
    cd "$PROJECT_DIR"   # go back to the main project folder

    echo "--- Downloading PubMed daily update files ---"
    # Same wget pattern as the baseline, just pointed at the updatefiles
    # folder instead. These are the daily deltas (new/revised/deleted
    # citations) published since the baseline was cut.
    wget -r -np -nH --cut-dirs=2 -c \
        -A "*.xml.gz,*.xml.gz.md5" \
        -P "$UPDATES_DIR" \
        "https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/"
else
    echo "Skipping PubMed download. Re-run this script anytime and answer 'y'"
    echo "at this step to grab it later."
fi

# ------------------------------------------------------------------------
echo ""
echo "=== DONE ==="
echo "Project folder:      $PROJECT_DIR"
echo "Ontology/data files: $DATA_DIR"
echo "PubMed baseline:     $BASELINE_DIR"
echo "PubMed updates:      $UPDATES_DIR"
echo ""
echo "To activate the Python environment later, run:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "NOT included in this script (ask if you want these):"
echo "  - PMC Open Access Subset (very large — worth scoping to your"
echo "    research area rather than downloading in full)"
echo "  - Parsing the PubMed XML into a queryable local database"
echo "  - Applying update files in sequence to keep the corpus current"
