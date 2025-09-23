import os
import zipfile
import shutil
import time
import pandas as pd
from datetime import timedelta
from functools import wraps
from urllib.parse import urlparse, urljoin

from flask import (
    Blueprint, render_template, request, redirect, url_for,
    send_from_directory, flash, current_app, session
)

# Import your processing functions
from app.services.k_electric import process_k_electric_bills
from app.services.general_disco import process_disco_bills

bp = Blueprint('main', __name__)

# =============================
# Session policy
# =============================
IDLE_TIMEOUT_SECONDS = 30 * 60   # 30 minutes of inactivity -> logout
REMEMBER_LIFETIME_DAYS = 7       # absolute max age for "Remember me" sessions

# ---------------- Emissions helpers ----------------
# >>> Fill with your exact CO2 factor (Tonnes per unit/kWh)
CO2_FACTOR_PER_UNIT  = 0.0005          # TODO: set real value from your attachment
CH4_FACTOR_PER_UNIT  = 0.00000002   # Tonnes per unit/kWh
N2O_FACTOR_PER_UNIT  = 0.00000009   # Tonnes per unit/kWh

def _num_series(s: pd.Series) -> pd.Series:
    """Safely coerce a series to numeric (handles commas/blanks)."""
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False), errors="coerce").fillna(0)

def add_emissions_columns(df: pd.DataFrame, on_peak_col: str, off_peak_col: str) -> pd.DataFrame:
    """
    Adds per-row totals and emissions columns to df:
      - TOTAL_UNITS_CONSUMED
      - CO2_tonnes, CH4_tonnes, N2O_tonnes
      - NET_CARBON_FOOTPRINT_tonnes
    Mutates and returns df.
    """
    if on_peak_col not in df.columns or off_peak_col not in df.columns:
        missing = [c for c in (on_peak_col, off_peak_col) if c not in df.columns]
        raise KeyError(f"Missing expected columns: {missing}")

    onp  = _num_series(df[on_peak_col])
    offp = _num_series(df[off_peak_col])

    df["TOTAL_UNITS_CONSUMED"] = onp + offp
    df["CO2_tonnes"] = df["TOTAL_UNITS_CONSUMED"] * CO2_FACTOR_PER_UNIT
    df["CH4_tonnes"] = df["TOTAL_UNITS_CONSUMED"] * CH4_FACTOR_PER_UNIT
    df["N2O_tonnes"] = df["TOTAL_UNITS_CONSUMED"] * N2O_FACTOR_PER_UNIT
    df["NET_CARBON_FOOTPRINT_tonnes"] = df["CO2_tonnes"] + df["CH4_tonnes"] + df["N2O_tonnes"]
    return df

def write_summary_sheet(writer: pd.ExcelWriter, sheet_name: str, net_total: float):
    """
    Writes a tiny 2-column table with a single row:
    Metric | Value
    NET_CARBON_FOOTPRINT_tonnes | <sum>
    """
    summary_df = pd.DataFrame(
        {"Metric": ["NET_CARBON_FOOTPRINT_tonnes"], "Value": [net_total]}
    )
    summary_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0, startcol=0)
# ---------------------------------------------------

# -----------------------------
# Helpers
# -----------------------------
def clear_downloads_folder():
    downloads_path = os.path.join(os.getcwd(), 'downloads')

    # Create the folder if it doesn't exist
    if not os.path.exists(downloads_path):
        os.makedirs(downloads_path)
        return

    # Remove all files and subfolders
    for filename in os.listdir(downloads_path):
        file_path = os.path.join(downloads_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # remove file or symbolic link
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # remove folder
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

def _uploads_dir():
    return current_app.config['UPLOAD_FOLDER']

def _downloads_dir():
    return current_app.config['DOWNLOAD_FOLDER']

def _excel_path():
    # Master Excel placed manually in uploads
    return os.path.join(_uploads_dir(), 'DISCO_ACCOUNT_LIST.xlsx')

def _is_safe_url(target):
    host_url = request.host_url
    ref_url = urlparse(host_url)
    test_url = urlparse(urljoin(host_url, target))
    return (test_url.scheme in ('http', 'https') and ref_url.netloc == test_url.netloc)

def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if 'user' not in session:
            flash('Please sign in first.', 'error')
            next_url = request.full_path if request.method == 'GET' else url_for('main.index')
            return redirect(url_for('main.login', next=next_url))
        return view(*args, **kwargs)
    return wrapped

# Make sure folders exist
@bp.before_app_request
def _ensure_dirs():
    os.makedirs(_uploads_dir(), exist_ok=True)
    os.makedirs(_downloads_dir(), exist_ok=True)

# Enforce idle timeout + absolute lifetime for "remember me"
@bp.before_app_request
def _session_guard():
    # Configure absolute lifetime for permanent sessions
    current_app.permanent_session_lifetime = timedelta(days=REMEMBER_LIFETIME_DAYS)

    # Skip checks on auth/static endpoints
    if request.endpoint in ('main.login', 'main.logout', 'static'):
        return

    # If not logged in, nothing to do
    if 'user' not in session:
        return

    # Idle-timeout check
    now = int(time.time())
    last_seen = session.get('last_seen', now)
    if now - last_seen > IDLE_TIMEOUT_SECONDS:
        session.clear()
        flash('Session expired due to inactivity. Please sign in again.', 'error')
        return redirect(url_for('main.login', next=request.url))

    # Refresh activity timestamp
    session['last_seen'] = now

# -----------------------------
# Auth (very simple demo)
# -----------------------------
@bp.route("/login", methods=["GET", "POST"])
def login():
    next_url = request.args.get('next') or request.form.get('next')
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        remember = request.form.get("remember") == "1"

        # TODO: replace with your real user lookup
        valid = (username in ("admin", "admin@example.com")) and (password == "admin123")

        if not username or not password:
            flash("Please fill in all fields.", "error")
            return render_template("login.html", next=next_url)

        if valid:
            session['user'] = username
            session.permanent = remember  # True -> cookie persists up to permanent_session_lifetime
            session['last_seen'] = int(time.time())  # start idle timer
            flash("Welcome back!", "success")
            if next_url and _is_safe_url(next_url):
                return redirect(next_url)
            return redirect(url_for("main.index"))
        else:
            flash("Invalid credentials. Try again.", "error")
            return render_template("login.html", next=next_url)

    return render_template("login.html", next=next_url)

@bp.route("/logout")
def logout():
    session.clear()
    flash("Signed out successfully.", "success")
    return redirect(url_for("main.login"))

# -----------------------------d
# Pages
# -----------------------------
@bp.route("/", methods=["GET"])
@login_required
def index():
    # Read DISCO list from Excel
    try:
        df = pd.read_excel(_excel_path())
        disco_list = sorted(df['DISCO'].dropna().unique())
    except Exception as e:
        disco_list = []
        flash(f"Could not load Excel: {e}", "error")
    return render_template("index.html", disco_list=disco_list)

# -----------------------------
# Fetch single DISCO
# -----------------------------
@bp.route("/fetch/<disco_name>", methods=["POST"])
@login_required
def fetch_disco(disco_name):
    try:
        clear_downloads_folder()
        df = pd.read_excel(_excel_path())
        df['DISCO'] = df['DISCO'].astype(str)

        sub_df = df[df['DISCO'].str.upper() == disco_name.upper()]
        uploads_dir = _uploads_dir()
        downloads_dir = _downloads_dir()

        temp_path = os.path.join(uploads_dir, f"temp_{disco_name}.xlsx")
        sub_df.to_excel(temp_path, index=False)

        # Process based on DISCO and add emissions
        if disco_name.strip().upper() == "K-ELECTRIC":
            results_df = process_k_electric_bills(temp_path, download_dir=downloads_dir)
            results_df = add_emissions_columns(
                results_df,
                on_peak_col="ACTIVE UNITS ON PEAK",
                off_peak_col="ACTIVE UNITS OFF PEAK",
            )
        else:
            results_df = process_disco_bills(temp_path, download_dir=downloads_dir)
            results_df = add_emissions_columns(
                results_df,
                on_peak_col="KWH METER READING UNITS CONSUMED (P)",
                off_peak_col="KWH METER READING UNITS CONSUMED (O)",
            )

        # Totals for summary cell
        net_total = float(results_df["NET_CARBON_FOOTPRINT_tonnes"].sum())

        # Save Excel: Results + Summary (single cell with NET total)
        excel_file = os.path.join(downloads_dir, f"results_{disco_name}.xlsx")
        with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
            results_df.to_excel(writer, sheet_name='Results', index=False)
            write_summary_sheet(writer, sheet_name="Summary", net_total=net_total)

        # Create a ZIP excluding the Excel file and any .zip
        zip_file = os.path.join(downloads_dir, f"bills_{disco_name}.zip")
        with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(downloads_dir):
                for file in files:
                    if file == os.path.basename(excel_file):
                        continue  # skip Excel file
                    if file.endswith(".zip"):
                        continue  # skip other zips (avoid self-inclusion loop)
                    full_path = os.path.join(root, file)
                    arcname = os.path.relpath(full_path, downloads_dir)
                    zipf.write(full_path, arcname)

        table_data = results_df.fillna("").to_dict(orient="records")

        return render_template(
            "results.html",
            table_data=table_data,
            download_link=url_for('main.download_file', filename=os.path.basename(excel_file)),
            zip_link=url_for('main.download_file', filename=os.path.basename(zip_file)),
            result_summary=(
                f"Processed {disco_name} bills: {len(results_df)} records. "
                f"NET Carbon Footprint: {net_total:.6f} tonnes"
            ),
            error=None
        )
    except Exception as e:
        return render_template(
            "results.html",
            table_data=None,
            download_link=None,
            zip_link=None,
            result_summary=None,
            error=f"Error processing {disco_name}: {e}"
        )

# -----------------------------
# Fetch ALL
# -----------------------------
@bp.route("/fetch_all", methods=["POST"])
@login_required
def fetch_all():
    try:
        clear_downloads_folder()
        # Load master file
        df = pd.read_excel(_excel_path())
        df['DISCO'] = df['DISCO'].astype(str)

        # Split K-Electric vs others
        ke_df = df[df['DISCO'].str.upper() == "K-ELECTRIC"]
        others_df = df[df['DISCO'].str.upper() != "K-ELECTRIC"]

        uploads_dir = _uploads_dir()
        downloads_dir = _downloads_dir()

        # Process both separately and add emissions
        if not ke_df.empty:
            temp_ke = os.path.join(uploads_dir, "temp_KE.xlsx")
            ke_df.to_excel(temp_ke, index=False)
            df_ke_results = process_k_electric_bills(temp_ke, download_dir=downloads_dir)
            df_ke_results = add_emissions_columns(
                df_ke_results,
                on_peak_col="ACTIVE UNITS ON PEAK",
                off_peak_col="ACTIVE UNITS OFF PEAK",
            )
        else:
            df_ke_results = pd.DataFrame()

        if not others_df.empty:
            temp_others = os.path.join(uploads_dir, "temp_OTHERS.xlsx")
            others_df.to_excel(temp_others, index=False)
            df_others_results = process_disco_bills(temp_others, download_dir=downloads_dir)
            df_others_results = add_emissions_columns(
                df_others_results,
                on_peak_col="KWH METER READING UNITS CONSUMED (P)",
                off_peak_col="KWH METER READING UNITS CONSUMED (O)",
            )
        else:
            df_others_results = pd.DataFrame()

        # Combine results
        parts = []
        if not df_others_results.empty: parts.append(df_others_results)
        if not df_ke_results.empty: parts.append(df_ke_results)
        combined_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

        # Grand NET total for Summary cell
        net_total = float(combined_df["NET_CARBON_FOOTPRINT_tonnes"].sum()) if not combined_df.empty else 0.0

        # Write combined Excel results with Summary sheet
        excel_path = os.path.join(downloads_dir, "results_ALL.xlsx")
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            combined_df.to_excel(writer, sheet_name='Results', index=False)
            write_summary_sheet(writer, sheet_name="Summary", net_total=net_total)

        # Create ZIP file for bills only (excluding the Excel file and other zips)
        zip_path = os.path.join(downloads_dir, "bills_only.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(downloads_dir):
                for file in files:
                    if file.endswith(".xlsx") or file.endswith(".zip"):
                        continue
                    full_path = os.path.join(root, file)
                    arcname = os.path.relpath(full_path, downloads_dir)
                    zipf.write(full_path, arcname)

        # Display table
        table_data = combined_df.fillna("").to_dict(orient="records")

        return render_template(
            "results.html",
            table_data=table_data,
            download_link=url_for('main.download_file', filename="results_ALL.xlsx"),
            zip_link=url_for('main.download_file', filename="bills_only.zip"),
            result_summary=(
                f"Processed ALL DISCO bills: {len(combined_df)} records. "
                f"NET Carbon Footprint: {net_total:.6f} tonnes"
            ),
            error=None
        )

    except Exception as e:
        return render_template(
            "results.html",
            table_data=None,
            download_link=None,
            zip_link=None,
            result_summary=None,
            error=f"Error processing all discos: {e}"
        )

# -----------------------------
# Downloads
# -----------------------------
@bp.route('/downloads/<filename>')
@login_required
def download_file(filename):
    download_dir = _downloads_dir()
    return send_from_directory(download_dir, filename, as_attachment=True)