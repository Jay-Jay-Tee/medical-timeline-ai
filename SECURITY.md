# Security Policy

MediTrack handles patient health data. Security vulnerabilities in this project can affect real people in meaningful ways. We take reports seriously and will respond promptly.

---

## Supported Versions

| Version | Status |
|---------|--------|
| `main` (latest) | ✅ Actively maintained |
| Previous releases | ❌ No security patches |

We only backport security fixes to the current `main` branch. If you're running an older fork or pinned version, update to the latest.

---

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, report privately via one of these channels:

- **GitHub Private Security Advisories:** Use the "Report a vulnerability" button on the [Security tab](../../security/advisories/new) of this repository. This is the preferred method.
- **Email:** If you cannot use GitHub's advisory system, email the maintainer directly (see the GitHub profile for contact info). Encrypt sensitive details if possible.

### What to Include

A good vulnerability report helps us reproduce and fix the issue faster. Include:

- A clear description of the vulnerability and its potential impact
- Step-by-step reproduction instructions
- The environment where you found it (OS, Python version, browser if frontend)
- Any relevant logs, screenshots, or proof-of-concept code
- Whether you believe this is exploitable in the live demo at `meditrack.onrender.com`

### What Happens Next

1. **Acknowledgment** within 48 hours of receiving the report
2. **Initial assessment** within 5 business days — we'll confirm whether it's valid and assign a severity
3. **Fix development** — timeline depends on severity (see below)
4. **Coordinated disclosure** — we'll notify you before the fix goes public and credit you in the release notes (unless you prefer to remain anonymous)

We do not offer a bug bounty program at this time, but we will publicly credit all valid, responsibly disclosed reports.

---

## Severity and Response Timelines

| Severity | Definition | Target Fix Timeline |
|----------|-----------|---------------------|
| **Critical** | Unauthenticated access to patient records; remote code execution; credential leakage | 24–48 hours |
| **High** | Authentication bypass; IDOR allowing cross-patient data access; session fixation | 7 days |
| **Medium** | Reflected/stored XSS; insecure direct object reference with auth required; sensitive data in logs | 30 days |
| **Low** | Missing security headers; verbose error messages leaking stack traces; non-sensitive info disclosure | Next release cycle |

---

## Current Security Architecture

Understanding the existing security model helps contextualize what is and isn't in scope.

**Authentication**
- Passwords hashed with `bcrypt` (auto-salted, work factor default)
- Sessions managed via Flask-Login with HTTPOnly cookies
- CSRF protection enabled via Flask's session management

**Data Isolation**
- Medical records are filtered by `patient_id` at query time in Qdrant
- All record-modification endpoints require authentication via `@login_required`
- The public `/patient/<id>` endpoint is intentionally unauthenticated and read-only

**Transport**
- The app itself does not enforce HTTPS — this is delegated to the reverse proxy (nginx) or platform (Render)
- Running without HTTPS in production is a misconfiguration, not a code vulnerability

**Secrets Management**
- API keys and the Flask secret are loaded from environment variables via `python-dotenv`
- The `.env` file should never be committed (it's in `.gitignore`)

---

## Known Security Limitations

These are acknowledged design decisions or deferred features, not unaddressed vulnerabilities:

**In-memory user store** — User accounts are stored in a Python dict (`users_db`) that is wiped on server restart. This means there is no persistent password store at the database level. The risk is data loss, not unauthorized access.

**Shareable links are permanent** — The `/patient/<id>` URL cannot be revoked without changing the patient's ID (not currently implemented). This is security-through-obscurity and is explicitly called out in the README.

**No rate limiting** — There is no rate limiting on any endpoint, including login. Brute-force attacks against the `/login` endpoint are possible. Adding Flask-Limiter is an open contribution (see CONTRIBUTING.md).

**No RBAC** — All authenticated users have identical permissions. There is no distinction between patient, provider, and admin roles.

**Upload directory traversal** — Document download uses a UUID filename, not user-supplied input, for the `send_file` path. However, there is no allowlist validation on the `uploads/` directory. Reviewing this code path carefully before deploying in production is recommended.

**No HIPAA/GDPR compliance layer** — This system is not HIPAA-compliant out of the box. Deploying it for actual clinical use requires additional controls (audit logging, BAAs, access controls, data encryption at rest). The README medical disclaimer covers this.

---

## Credentials in the Repository

If you've cloned this repository and noticed that the `.env` file contains what appear to be real API keys and credentials — **they should be treated as compromised and rotated immediately.**

Credentials that appear in a public git history are compromised regardless of whether they've been "removed" in a later commit. If you are the maintainer, rotate:

- The Qdrant API key via the Qdrant Cloud dashboard
- The Groq API key via the Groq console
- The Flask `SECRET_KEY` by generating a new one: `python -c "import secrets; print(secrets.token_hex(32))"`

Going forward, use `.env.example` with placeholder values for documentation, and add the real `.env` to `.gitignore` before the first commit.

---

## Scope

### In Scope

- Authentication and session management flaws
- Cross-patient data access (IDOR)
- Injection vulnerabilities (SQL, NoSQL, command injection)
- Cross-site scripting (XSS) — stored or reflected
- Server-side request forgery (SSRF)
- Insecure file upload handling
- Credential or secret exposure
- Any vulnerability in the live demo at `meditrack.onrender.com`

### Out of Scope

- Denial of service attacks (no rate limiting is a known limitation, not a reportable vuln)
- Social engineering
- Physical security
- Vulnerabilities in third-party dependencies that are not directly exploitable in this application
- Issues that require the attacker to already have valid credentials and the same permissions as the victim
- Security issues in browsers or platforms (report those to the respective vendors)

---

## Disclosure Policy

We follow [coordinated vulnerability disclosure](https://cheatsheetseries.owasp.org/cheatsheets/Vulnerability_Disclosure_Cheat_Sheet.html). We ask researchers to:

1. Give us reasonable time to fix the issue before public disclosure
2. Not exploit the vulnerability beyond what's necessary to demonstrate it
3. Not access, modify, or delete patient data in the live demo
4. Not disrupt the availability of the service during testing

In return, we will:

1. Respond promptly and keep you informed throughout the process
2. Credit you publicly in the release notes (if you want credit)
3. Not pursue legal action against good-faith security researchers

---

*Last updated: March 2026*
