"""
download_day_data.py
────────────────────
Windows-compatible, targeted downloader for TartanAviation data.
Downloads ONLY the data needed for the 2020-10-22 prototype day.

Usage (from project root, venv activated):
    python download_day_data.py

What it downloads:
    1. ADS-B  : kbtp/raw/2020.zip  → extracts only 10-22-20/1.csv
    2. Audio  : kbtp/2020/10/10-22-20_audio.zip  → all .wav + .txt
    3. Weather: copies from tartan_scripts/weather/BTP.csv

Author: XCAS Project
"""

import io
import os
import shutil
import zipfile
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError

# ── Project paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).parent          # folder where this script lives
TARTAN_DATA   = PROJECT_ROOT / "tartan_data"   # destination root
TARTAN_SCRIPTS= PROJECT_ROOT / "tartan_scripts"# where weather CSV already lives

# ── S3 client (public, no credentials needed) ────────────────────────────────
# endpoint_url points to CMU's object store instead of AWS
# signature_version=UNSIGNED means we skip AWS credential signing
ENDPOINT_URL = (
    "https://airlab-cloud.andrew.cmu.edu:8080"
    "/swift/v1/AUTH_ac8533a83cff4d48bc8c608ad222d330"
)

def make_client(bucket_name: str):
    """Create a boto3 S3 client pointed at the CMU TartanAviation store."""
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT_URL,
        config=Config(signature_version=UNSIGNED),
    ), bucket_name


def check_object_exists(client, bucket: str, key: str) -> bool:
    """Return True if the S3 object exists, False otherwise."""
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise  # re-raise unexpected errors


def download_object_to_memory(client, bucket: str, key: str) -> bytes:
    """
    Stream an S3 object into memory in 4 MB chunks.
    Returns the raw bytes of the downloaded file.
    
    Why chunked streaming?
    ─────────────────────
    Large files (hundreds of MB) would OOM your process if loaded at once.
    Streaming in chunks keeps memory usage flat at ~4 MB regardless of file size.
    This is a standard pattern for any large file I/O in Python.
    """
    print(f"  ⬇  Streaming s3://{bucket}/{key} ...")
    response = client.get_object(Bucket=bucket, Key=key)
    
    buf = io.BytesIO()
    total = 0
    chunk_size = 4 * 1024 * 1024  # 4 MB per chunk
    
    for chunk in response["Body"].iter_chunks(chunk_size=chunk_size):
        buf.write(chunk)
        total += len(chunk)
        # Simple progress indicator — \r overwrites the same line
        print(f"\r     {total / 1_048_576:.1f} MB downloaded", end="", flush=True)
    
    print(f"\r  ✅ {total / 1_048_576:.1f} MB downloaded total")
    buf.seek(0)  # rewind to start before reading
    return buf.read()


def extract_zip_filtered(zip_bytes: bytes, dest_dir: Path,
                          filter_fn=None, strip_levels: int = 0):
    """
    Extract a zip archive from bytes, optionally filtering entries.

    Parameters
    ──────────
    zip_bytes   : raw bytes of the .zip file
    dest_dir    : destination directory (created if it doesn't exist)
    filter_fn   : callable(ZipInfo) → bool; only extract if returns True
                  Pass None to extract everything
    strip_levels: number of leading path components to remove from each entry
                  e.g. strip_levels=1 turns "kbtp/raw/2020/10-22-20/1.csv"
                  into "10-22-20/1.csv" in dest_dir

    📚 Why zipfile instead of subprocess("unzip")?
    ──────────────────────────────────────────────
    subprocess("unzip") calls a system binary that doesn't exist on Windows.
    Python's built-in `zipfile` module works identically on every OS.
    Always prefer stdlib over shell commands in cross-platform projects.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        entries = zf.infolist()
        
        # Apply optional filter
        if filter_fn:
            entries = [e for e in entries if filter_fn(e)]
        
        print(f"  📦 Extracting {len(entries)} entries → {dest_dir}")
        
        for entry in entries:
            # Strip leading path levels
            parts = Path(entry.filename).parts
            if len(parts) <= strip_levels:
                continue  # skip entries that would become empty path
            
            rel_path = Path(*parts[strip_levels:])
            
            # Skip macOS metadata garbage
            if "__MACOSX" in str(rel_path) or ".DS_Store" in str(rel_path):
                continue
            
            out_path = dest_dir / rel_path
            
            if entry.is_dir():
                out_path.mkdir(parents=True, exist_ok=True)
            else:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(entry) as src, open(out_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        
        print(f"  ✅ Extraction complete")


# ══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD 1: ADS-B for 2020-10-22
# ══════════════════════════════════════════════════════════════════════════════
def download_adsb():
    """
    Downloads kbtp/raw/2020.zip from tartanaviation-adsb bucket.
    Extracts ONLY the 10-22-20/ folder into tartan_data/kbtp/raw/2020/
    
    Why extract only one day?
    ─────────────────────────
    The 2020.zip likely contains the full year's daily folders.
    We only need one day for the prototype. Filtering during extraction
    saves disk space and time — no need to keep the rest.
    """
    print("\n" + "═"*60)
    print("DOWNLOAD 1: ADS-B data (kbtp, 2020-10-22)")
    print("═"*60)
    
    dest_day_dir = TARTAN_DATA / "kbtp" / "raw" / "2020" / "10-22-20"
    
    # Skip if already downloaded
    if (dest_day_dir / "1.csv").exists():
        print(f"  ✅ Already exists: {dest_day_dir / '1.csv'}")
        return True
    
    client, bucket = make_client("tartanaviation-adsb")
    s3_key = "kbtp/raw/2020.zip"
    
    print(f"  Checking s3://{bucket}/{s3_key} ...")
    if not check_object_exists(client, bucket, s3_key):
        print(f"  ✗ Object not found on server: {s3_key}")
        print("  → The server may be temporarily unavailable. Try again later.")
        return False
    
    # Download to memory
    zip_bytes = download_object_to_memory(client, bucket, s3_key)
    
    # Extract only the 10-22-20 folder
    # The zip structure is: kbtp/raw/2020/10-22-20/1.csv
    # strip_levels=3 removes "kbtp/raw/2020/" prefix
    # filter keeps only entries containing "10-22-20"
    def adsb_filter(entry: zipfile.ZipInfo) -> bool:
        return "10-22-20" in entry.filename
    
    dest_base = TARTAN_DATA / "kbtp" / "raw" / "2020"
    extract_zip_filtered(
        zip_bytes,
        dest_dir=dest_base,
        filter_fn=adsb_filter,
        strip_levels=0,   # removes "kbtp/raw/2020/" prefix
    )
    
    # Verify
    target = dest_day_dir / "1.csv"
    if target.exists():
        size_kb = target.stat().st_size / 1024
        print(f"\n  ✅ ADS-B CSV ready: {target}")
        print(f"     Size: {size_kb:.1f} KB")
        return True
    else:
        print(f"\n  ✗ Expected file not found after extraction: {target}")
        print("  → Check zip contents below for correct path structure:")
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for name in zf.namelist()[:20]:
                print(f"     {name}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD 2: Audio for 2020-10-22
# ══════════════════════════════════════════════════════════════════════════════
def download_audio():
    """
    Downloads kbtp/2020/10/10-22-20_audio.zip from tartanaviation-audio bucket.
    Extracts all .wav and .txt files into tartan_data/kbtp/2020/10/10-22-20_audio/
    """
    print("\n" + "═"*60)
    print("DOWNLOAD 2: Audio data (kbtp, 2020-10-22)")
    print("═"*60)
    
    dest_audio_dir = TARTAN_DATA / "kbtp" / "2020" / "10" / "10-22-20_audio"
    
    # Skip if already downloaded and non-empty
    if dest_audio_dir.exists():
        wav_count = len(list(dest_audio_dir.glob("*.wav")))
        if wav_count > 0:
            print(f"  ✅ Already exists: {wav_count} .wav files in {dest_audio_dir}")
            return True
    
    client, bucket = make_client("tartanaviation-audio")
    s3_key = "kbtp/2020/10/10-22-20_audio.zip"
    
    print(f"  Checking s3://{bucket}/{s3_key} ...")
    if not check_object_exists(client, bucket, s3_key):
        print(f"  ✗ Object not found: {s3_key}")
        print("  → Trying alternate key format ...")
        
        # Sometimes the path format varies — try without leading location folder
        alt_key = "kbtp/2020/10/10-22-20_audio.zip"
        print(f"  Trying: {alt_key}")
        if not check_object_exists(client, bucket, alt_key):
            print("  ✗ Not found with alternate format either.")
            print("  → Will attempt Sample download as fallback.")
            return download_audio_sample_fallback(client, bucket, dest_audio_dir)
        s3_key = alt_key
    
    zip_bytes = download_object_to_memory(client, bucket, s3_key)
    
    # Audio zips use -j flag (junk paths) in original script
    # meaning all files land flat in the destination directory
    # strip_levels=0 with flat structure extraction
    def audio_filter(entry: zipfile.ZipInfo) -> bool:
        name = entry.filename.lower()
        return name.endswith(".wav") or name.endswith(".txt")
    
    # Flat extraction — all files go directly into dest_audio_dir
    dest_audio_dir.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        entries = [e for e in zf.infolist() if audio_filter(e)]
        print(f"  📦 Extracting {len(entries)} audio files → {dest_audio_dir}")
        
        for entry in entries:
            # Extract just the filename (junk path, like unzip -j)
            flat_name = Path(entry.filename).name
            if not flat_name:
                continue
            out_path = dest_audio_dir / flat_name
            with zf.open(entry) as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
    
    wav_count = len(list(dest_audio_dir.glob("*.wav")))
    txt_count = len(list(dest_audio_dir.glob("*.txt")))
    print(f"\n  ✅ Audio ready: {wav_count} .wav + {txt_count} .txt in {dest_audio_dir}")
    return wav_count > 0


def download_audio_sample_fallback(client, bucket, dest_audio_dir):
    """
    Fallback: download the official Sample (Nov 2020) if Oct 22 isn't found.
    This gives us valid audio files to prove the pipeline works, even if
    the exact date isn't available.
    """
    print("\n  ⚠  Fallback: downloading official Sample audio (Nov 2020)...")
    sample_key = "kbtp/2020/11/11-02-20_audio.zip"
    
    if not check_object_exists(client, bucket, sample_key):
        print("  ✗ Sample also not found. Check network/VPN access.")
        return False
    
    sample_dest = TARTAN_DATA / "kbtp" / "2020" / "11" / "11-02-20_audio"
    sample_dest.mkdir(parents=True, exist_ok=True)
    
    zip_bytes = download_object_to_memory(client, bucket, sample_key)
    
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        entries = [e for e in zf.infolist() 
                   if e.filename.endswith(".wav") or e.filename.endswith(".txt")]
        print(f"  📦 Extracting {len(entries)} sample audio files → {sample_dest}")
        for entry in entries:
            flat_name = Path(entry.filename).name
            out_path = sample_dest / flat_name
            with zf.open(entry) as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
    
    wav_count = len(list(sample_dest.glob("*.wav")))
    print(f"  ✅ Sample fallback: {wav_count} .wav files ready")
    print(f"  ⚠  NOTE: Update AUDIO_DIR in your notebook to: {sample_dest}")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Weather CSV (already on disk — just copy to expected path)
# ══════════════════════════════════════════════════════════════════════════════
def setup_weather():
    """
    The weather CSV is already downloaded as part of the repo clone.
    We just need to make sure it's accessible at the path the notebook expects.
    We'll create a symlink-style copy in tartan_data/ so paths stay consistent.
    """
    print("\n" + "═"*60)
    print("STEP 3: Weather CSV setup")
    print("═"*60)
    
    src = TARTAN_SCRIPTS / "weather" / "BTP.csv"
    dest_dir = TARTAN_DATA / "weather"
    dest = dest_dir / "BTP.csv"
    
    if not src.exists():
        print(f"  ✗ Source not found: {src}")
        return False
    
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    if dest.exists():
        print(f"  ✅ Already in place: {dest}")
    else:
        shutil.copy2(src, dest)
        print(f"  ✅ Copied: {src.name} → {dest}")
    
    size_mb = src.stat().st_size / (1024*1024)
    print(f"     Size: {size_mb:.1f} MB")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# FINAL VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════
def verify_all():
    """Check all expected files exist and print a summary."""
    print("\n" + "═"*60)
    print("VERIFICATION SUMMARY")
    print("═"*60)
    
    checks = {
        "ADS-B CSV (10-22-20)":
            TARTAN_DATA / "kbtp/raw/2020/10-22-20/1.csv",
        "Audio dir (10-22-20)":
            TARTAN_DATA / "kbtp/2020/10/10-22-20_audio",
        "Weather BTP.csv":
            TARTAN_DATA / "weather/BTP.csv",
    }
    
    all_ok = True
    for label, path in checks.items():
        if path.exists():
            if path.is_dir():
                n = len(list(path.iterdir()))
                print(f"  ✅ {label}: {n} files in {path.name}/")
            else:
                kb = path.stat().st_size / 1024
                print(f"  ✅ {label}: {kb:.1f} KB")
        else:
            print(f"  ✗  {label}: NOT FOUND → {path}")
            all_ok = False
    
    if all_ok:
        print("\n🎉 All data ready. You can now run your notebook.")
        print(f"\nUpdate Cell 1 WEATHER_PATH to:\n"
              f"  WEATHER_PATH = PROJECT_ROOT / 'tartan_data/weather/BTP.csv'")
    else:
        print("\n⚠  Some files are missing. See errors above.")
    
    return all_ok


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("XCAS-GA: TartanAviation Data Downloader (Windows-Compatible)")
    print("Target: KBTP, 2020-10-22")
    print(f"Destination root: {TARTAN_DATA}\n")
    
    results = {
        "adsb"   : download_adsb(),
        "audio"  : download_audio(),
        "weather": setup_weather(),
    }
    
    verify_all()
    
    print("\nResults:")
    for k, v in results.items():
        status = "✅ OK" if v else "✗ FAILED"
        print(f"  {k:10s} : {status}")