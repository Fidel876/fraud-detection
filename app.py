from flask import Flask, request, render_template, jsonify, session
import pandas as pd
import joblib
import io
import os
from datetime import timedelta
from functools import wraps
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════
#  REQUIRED SUPABASE SQL (run once in SQL Editor):
#
#  alter table user_profiles add column if not exists full_name text;
#
#  -- Allow self-registration (new users insert their own profile)
#  create policy "Users can insert own profile" on user_profiles
#    for insert with check (auth.uid() = id);
#
#  create policy "Users can read own profile" on user_profiles
#    for select using (auth.uid() = id);
# ══════════════════════════════════════════════════

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-this-in-production")
app.permanent_session_lifetime = timedelta(hours=8)

# ── Supabase ──
supabase = create_client(
    os.environ.get("SUPABASE_URL"),
    os.environ.get("SUPABASE_SERVICE_KEY")
)

# ── Model ──
model  = joblib.load("model/fraud_model.pkl")
scaler = joblib.load("model/scaler.pkl")


# ════════════════════════════════════════
#  AUTH DECORATORS
# ════════════════════════════════════════

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            return jsonify({"error": "Unauthorized. Please log in."}), 401
        return f(*args, **kwargs)
    return decorated


def role_required(*roles):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if "user" not in session:
                return jsonify({"error": "Unauthorized. Please log in."}), 401
            if session["user"]["role"] not in roles:
                return jsonify({"error": f"Forbidden. Required role: {', '.join(roles)}"}), 403
            return f(*args, **kwargs)
        return decorated
    return decorator


# ════════════════════════════════════════
#  ROUTES
# ════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


# ── Register ──
@app.route("/register", methods=["POST"])
def register():
    data      = request.get_json()
    email     = data.get("email", "").strip()
    password  = data.get("password", "")
    full_name = data.get("full_name", "").strip()
    role      = data.get("role", "analyst")

    if not email or not password or not full_name:
        return jsonify({"error": "All fields are required"}), 400

    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters"}), 400

    if role not in ["admin", "analyst", "investigator"]:
        return jsonify({"error": "Invalid role selected"}), 400

    try:
        # Create user in Supabase Auth
        response = supabase.auth.admin.create_user({
            "email":            email,
            "password":         password,
            "email_confirm":    True,   # auto-confirm so they can log in immediately
            "user_metadata":    {"full_name": full_name}
        })

        user = response.user
        if not user:
            return jsonify({"error": "Registration failed"}), 400

        # Save profile with role
        supabase.table("user_profiles").insert({
            "id":        user.id,
            "email":     email,
            "full_name": full_name,
            "role":      role
        }).execute()

        return jsonify({"message": "Account created successfully. You can now log in."})

    except Exception as e:
        error_msg = str(e)
        if "already registered" in error_msg.lower() or "already been registered" in error_msg.lower():
            return jsonify({"error": "This email is already registered"}), 400
        return jsonify({"error": "Registration failed. Please try again."}), 400


# ── Login ──
@app.route("/login", methods=["POST"])
def login():
    data     = request.get_json()
    email    = data.get("email", "").strip()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    try:
        response = supabase.auth.sign_in_with_password({
            "email":    email,
            "password": password
        })

        user = response.user
        if not user:
            return jsonify({"error": "Invalid credentials"}), 401

        # Fetch role from user_profiles
        profile = supabase.table("user_profiles") \
            .select("role") \
            .eq("id", user.id) \
            .single() \
            .execute()

        role = profile.data.get("role", "analyst") if profile.data else "analyst"

        # Start session
        session.permanent = True
        session["user"] = {
            "id":    user.id,
            "email": user.email,
            "role":  role
        }

        # Log login history
        try:
            supabase.table("login_history").insert({
                "email":      user.email,
                "role":       role,
                "ip_address": request.remote_addr
            }).execute()
        except Exception:
            pass  # Don't fail login if history insert fails

        return jsonify({
            "message": "Login successful",
            "email":   user.email,
            "role":    role
        })

    except Exception:
        return jsonify({"error": "Invalid email or password"}), 401


# ── Logout ──
@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"message": "Logged out successfully"})


# ── Password Reset ──
@app.route("/reset-password", methods=["POST"])
def reset_password():
    data  = request.get_json()
    email = data.get("email", "").strip()
    if not email:
        return jsonify({"error": "Email is required"}), 400
    try:
        supabase.auth.reset_password_email(email)
        return jsonify({"message": "Password reset email sent"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ── Current User ──
@app.route("/me")
@login_required
def me():
    return jsonify(session["user"])


# ── Test DB ──
@app.route("/test-db")
def test_db():
    try:
        result = supabase.table("predictions").select("*").limit(1).execute()
        return jsonify({"status": "✅ Connected to Supabase", "data": result.data})
    except Exception as e:
        return jsonify({"status": "❌ Connection failed", "error": str(e)})


# ── Predict ──
@app.route("/predict", methods=["POST"])
@role_required("admin", "analyst")
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        data = pd.read_csv(io.StringIO(file.read().decode("utf-8")))

        # Drop Class column if present
        if "Class" in data.columns:
            X = data.drop("Class", axis=1)
        else:
            X = data.copy()

        # Scale and predict
        X_scaled    = scaler.transform(X)
        predictions = model.predict(X_scaled)

        data["Fraud_Status"] = predictions
        data["Fraud_Status"] = data["Fraud_Status"].map({1: "Normal", -1: "Fraud"})

        summary    = data["Fraud_Status"].value_counts().to_dict()
        fraud_rows = data[data["Fraud_Status"] == "Fraud"].head(100).to_dict(orient="records")
        preview    = data.head(10).to_dict(orient="records")
        columns    = list(data.columns)

        # Save prediction record to Supabase
        pred_record = supabase.table("predictions").insert({
            "filename":      file.filename,
            "total_records": len(data),
            "fraud_count":   summary.get("Fraud", 0),
            "normal_count":  summary.get("Normal", 0),
            "uploaded_by":   session["user"]["email"]
        }).execute()

        prediction_id = pred_record.data[0]["id"]

        # Save fraud transactions
        if fraud_rows:
            fraud_inserts = [
                {
                    "prediction_id":    prediction_id,
                    "transaction_data": row,
                    "fraud_status":     "Fraud"
                }
                for row in fraud_rows
            ]
            supabase.table("fraud_transactions").insert(fraud_inserts).execute()

        return jsonify({
            "summary":    summary,
            "fraud_rows": fraud_rows,
            "preview":    preview,
            "columns":    columns
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── History (admin + analyst) ──
@app.route("/history", methods=["GET"])
@role_required("admin", "analyst")
def history():
    try:
        email = session["user"]["email"]
        role  = session["user"]["role"]

        # Admins see all, analysts see only their own
        if role == "admin":
            result = supabase.table("predictions") \
                .select("*") \
                .order("created_at", desc=True) \
                .limit(50) \
                .execute()
        else:
            result = supabase.table("predictions") \
                .select("*") \
                .eq("uploaded_by", email) \
                .order("created_at", desc=True) \
                .limit(50) \
                .execute()

        return jsonify(result.data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Login History (admin only) ──
@app.route("/login-history", methods=["GET"])
@role_required("admin")
def login_history():
    try:
        result = supabase.table("login_history") \
            .select("*") \
            .order("logged_in_at", desc=True) \
            .limit(100) \
            .execute()
        return jsonify(result.data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)