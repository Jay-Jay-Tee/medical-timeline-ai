# 🔧 Troubleshooting Guide

## 🎨 UI/UX Issues

### Issue: Text boxes are hard to read (black text on dark background)

**✅ FIXED!** The new version uses CSS variables for proper theming:
- Light mode: White input backgrounds with dark text
- Dark mode: Dark gray input backgrounds with light text
- All inputs now have proper contrast and borders
- Toggle dark mode with the moon/sun button

**How to verify the fix:**
1. Reload the page (Ctrl+F5 or Cmd+Shift+R)
2. Check input fields - should have clear borders
3. Try dark mode toggle - text should always be readable

---

### Issue: Voice input button not responding

**Problem:** Browser doesn't support Web Speech API or microphone blocked.

**Solution:**

1. **Use Chrome or Edge** (Firefox doesn't support it yet)
   ```
   ✅ Chrome: Full support
   ✅ Edge: Full support
   ⚠️ Safari: Partial support
   ❌ Firefox: Not supported
   ```

2. **Allow microphone permissions:**
   - Chrome: Click lock icon in address bar → Microphone → Allow
   - Edge: Same as Chrome
   - Safari: Settings → Websites → Microphone → Allow

3. **Test microphone:**
   ```javascript
   // Open browser console (F12) and run:
   navigator.mediaDevices.getUserMedia({ audio: true })
     .then(stream => console.log('Mic works!'))
     .catch(err => console.error('Mic error:', err))
   ```

4. **Check HTTPS requirement:**
   - Voice input requires HTTPS in production
   - Works on localhost for development
   - Use ngrok for testing: `ngrok http 5000`

**Still not working?**
- Check browser console for errors (F12 → Console tab)
- Try different browser
- Restart browser

---

### Issue: Document upload preview not showing

**Problem:** File type not supported or file too large.

**Solution:**

1. **Check file type:**
   ```
   ✅ Supported: .jpg, .jpeg, .png, .gif, .pdf
   ❌ Not supported: .docx, .txt, .zip
   ```

2. **Check file size:**
   ```
   Maximum: 10MB per file
   Recommended: Under 5MB
   ```

3. **Compress large images:**
   ```bash
   # Use online tools:
   - TinyPNG (https://tinypng.com)
   - Squoosh (https://squoosh.app)
   ```

4. **Try drag-and-drop:**
   - Sometimes file input button fails
   - Drag file directly to drop zone
   - Should see border turn blue

---

## 🔐 Authentication Issues

### Issue: "Email already registered" but I don't have an account

**Problem:** You tried to register before, or someone else used that email.

**Solution:**

1. **Try logging in instead:**
   - Click "Sign In"
   - Use the email and password you remember
   
2. **Use different email:**
   - Gmail allows aliases: `youremail+hospital@gmail.com`
   - Each alias is treated as unique

3. **Clear browser data:**
   ```
   Chrome: Settings → Privacy → Clear browsing data
   Select: Cookies and site data
   ```

---

### Issue: "Invalid credentials" when logging in

**Problem:** Email or password incorrect, or account doesn't exist.

**Solution:**

1. **Check email spelling:**
   - No spaces before/after
   - Correct domain (.com vs .org)
   
2. **Try creating new account:**
   - Click "Create one"
   - Use same email (will tell you if it exists)

3. **Password requirements:**
   - Current system: No specific requirements
   - Tip: Use strong password anyway!

---

### Issue: Not staying logged in / Session expires

**Problem:** Cookies not saving or session timeout.

**Solution:**

1. **Enable cookies:**
   ```
   Chrome: Settings → Privacy → Cookies → Allow all
   Edge: Same as Chrome
   ```

2. **Check third-party cookies:**
   - Some ad blockers block Flask sessions
   - Whitelist localhost:5000

3. **Session expires after:**
   - Default: Browser close
   - Future update: Remember me checkbox

---

## 🚀 Server Issues

### Issue: "Connection refused" or "Cannot connect to server"

**Problem:** Flask server not running or wrong port.

**Solution:**

1. **Check if server is running:**
   ```bash
   # Look for this output:
   * Running on http://0.0.0.0:5000
   * Running on http://127.0.0.1:5000
   ```

2. **Restart server:**
   ```bash
   # Stop: Ctrl+C
   # Start: python app.py
   ```

3. **Check port availability:**
   ```bash
   # On Linux/Mac:
   lsof -i :5000
   
   # On Windows:
   netstat -ano | findstr :5000
   ```

4. **Use different port:**
   ```bash
   # In app.py, change last line to:
   app.run(debug=True, host='0.0.0.0', port=8000)
   ```

---

### Issue: "ModuleNotFoundError: No module named 'qdrant_client'"

**Problem:** Dependencies not installed.

**Solution:**

```bash
# 1. Activate virtual environment
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 2. Upgrade pip
pip install --upgrade pip

# 3. Install requirements
pip install -r requirements.txt

# 4. Verify installation
pip list | grep qdrant
```

**Still not working?**
```bash
# Try installing individually:
pip install qdrant-client
pip install fastembed
pip install groq
pip install flask flask-cors flask-login
```

---

### Issue: "Failed to initialize application" on startup

**Problem:** Missing environment variables or invalid credentials.

**Solution:**

1. **Check .env file exists:**
   ```bash
   ls -la .env  # Should show the file
   ```

2. **Verify .env contents:**
   ```bash
   cat .env
   # Should show all three keys:
   # QDRANT_URL=...
   # QDRANT_API_KEY=...
   # GROQ_API_KEY=...
   # SECRET_KEY=...
   ```

3. **Test Qdrant connection:**
   ```python
   from qdrant_client import QdrantClient
   client = QdrantClient(url="your_url", api_key="your_key")
   print(client.get_collections())
   ```

4. **Test Groq connection:**
   ```python
   from groq import Groq
   client = Groq(api_key="your_key")
   response = client.chat.completions.create(
       model="llama-3.3-70b-versatile",
       messages=[{"role": "user", "content": "Hi"}],
       max_tokens=10
   )
   print(response.choices[0].message.content)
   ```

---

## 📊 Data & Timeline Issues

### Issue: "No events found for this patient"

**Problem:** Patient ID doesn't exist or no events added yet.

**Solution:**

1. **Check Patient ID format:**
   ```
   Correct: MED-A1B2C3D4
   Wrong: med-a1b2c3d4 (lowercase)
   Wrong: A1B2C3D4 (missing prefix)
   ```

2. **Add an event first:**
   - Sign in
   - Fill event type and details
   - Click "Save Event"
   - Wait for success message
   - Then analyze timeline

3. **Check if you're logged in:**
   - Patient ID should auto-fill after login
   - If empty, click "Sign In"

---

### Issue: Timeline shows wrong data or missing events

**Problem:** Caching or stale data.

**Solution:**

1. **Hard refresh:**
   ```
   Chrome/Edge/Firefox: Ctrl+Shift+R (Windows) or Cmd+Shift+R (Mac)
   Safari: Cmd+Option+R
   ```

2. **Clear localStorage:**
   ```javascript
   // Open console (F12) and run:
   localStorage.clear();
   location.reload();
   ```

3. **Check Qdrant data directly:**
   ```python
   from qdrant_client import QdrantClient
   client = QdrantClient(url="...", api_key="...")
   results = client.scroll(
       collection_name="medical_events",
       limit=10
   )
   print(results)
   ```

---

### Issue: AI summary says "AI analysis unavailable"

**Problem:** Groq API key invalid or rate limit hit.

**Solution:**

1. **Check API key:**
   ```bash
   # In .env file:
   GROQ_API_KEY=gsk_...your_key_here
   ```

2. **Get new API key:**
   - Go to https://console.groq.com/keys
   - Create new key
   - Replace in .env
   - Restart server

3. **Check rate limits:**
   ```
   Free tier: 14,400 requests/day, 30/minute
   
   If exceeded:
   - Wait for reset (midnight UTC)
   - Or upgrade plan
   ```

4. **Test Groq directly:**
   ```bash
   curl https://api.groq.com/openai/v1/chat/completions \
     -H "Authorization: Bearer $GROQ_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"model":"llama-3.3-70b-versatile","messages":[{"role":"user","content":"Hi"}]}'
   ```

---

## 📄 PDF Export Issues

### Issue: PDF download fails or is corrupted

**Problem:** ReportLab error or missing data.

**Solution:**

1. **Check ReportLab installed:**
   ```bash
   pip list | grep reportlab
   # Should show: reportlab 4.0.7 or newer
   ```

2. **Reinstall if needed:**
   ```bash
   pip uninstall reportlab
   pip install reportlab==4.0.7
   ```

3. **Check if patient has events:**
   - Need at least 1 event to export
   - Add test event if timeline empty

4. **Check browser download settings:**
   - Chrome: Check Downloads folder
   - May be blocked by popup blocker

---

### Issue: PDF is blank or missing data

**Problem:** Timeline empty or PDF generation error.

**Solution:**

1. **Check browser console:**
   ```javascript
   // Look for errors like:
   "Failed to generate PDF"
   "No events found"
   ```

2. **Check server logs:**
   ```bash
   # Look for errors in terminal:
   [ERROR] PDF export error: ...
   ```

3. **Verify data exists:**
   - Run timeline analysis first
   - Should show events in table
   - Then try PDF export

---

## 🌐 Browser-Specific Issues

### Chrome/Edge

**Issue: Microphone not working**
```
Solution:
1. chrome://settings/content/microphone
2. Check "Sites can ask to use your microphone"
3. Remove localhost from blocked list
```

**Issue: Downloads not starting**
```
Solution:
1. chrome://settings/downloads
2. Check "Ask where to save each file before downloading" is OFF
3. Clear download history
```

### Safari

**Issue: Voice input not starting**
```
Solution:
1. Safari → Preferences → Websites → Microphone
2. Allow for localhost
3. Note: May have limited support
4. Consider using Chrome instead
```

### Firefox

**Issue: Voice input not available**
```
Solution:
Firefox doesn't support Web Speech API yet.
Use Chrome or Edge for voice features.
Other features work fine in Firefox.
```

---

## 🔍 Advanced Debugging

### Enable Debug Mode

```python
# In app.py, last line:
app.run(debug=True, host='0.0.0.0', port=5000)
```

Benefits:
- Auto-reload on code changes
- Detailed error pages
- Interactive debugger

**⚠️ Never use in production!**

---

### Check Server Logs

```bash
# All logs appear in terminal where you ran:
python app.py

# Look for:
[INFO] - Success messages
[WARNING] - Potential issues
[ERROR] - Actual errors
```

---

### Browser Console Debugging

```javascript
// Open console: F12 → Console tab

// Check for errors:
// Red text = error
// Yellow text = warning

// Test API endpoint:
fetch('/health')
  .then(r => r.json())
  .then(d => console.log(d));

// Test authentication:
fetch('/me', { credentials: 'include' })
  .then(r => r.json())
  .then(d => console.log('User:', d));
```

---

### Network Tab Debugging

```
1. Open DevTools (F12)
2. Go to Network tab
3. Perform action (add event, etc.)
4. Click request to see:
   - Request headers
   - Request payload
   - Response data
   - Status code
```

**Common status codes:**
- 200 = Success
- 400 = Bad request (check your data)
- 401 = Not authenticated (login first)
- 404 = Not found (check endpoint URL)
- 500 = Server error (check server logs)

---

## 🆘 Getting More Help

### Still Having Issues?

1. **Check the README:**
   - Complete setup instructions
   - Feature explanations
   - Example workflows

2. **Search GitHub Issues:**
   - Someone may have had same problem
   - Check closed issues too

3. **Create New Issue:**
   - Include error messages
   - Share server logs
   - Describe steps to reproduce
   - Mention OS and browser

4. **Provide Debug Info:**
   ```bash
   # System info
   python --version
   pip list
   
   # Check .env (hide sensitive data!)
   cat .env
   
   # Test individual components
   python -c "from qdrant_client import QdrantClient; print('Qdrant OK')"
   python -c "from groq import Groq; print('Groq OK')"
   python -c "import flask; print('Flask OK')"
   ```

---

## 🎓 Learning Resources

**Python Debugging:**
- [Python Debugger Tutorial](https://docs.python.org/3/library/pdb.html)
- [Flask Debugging Guide](https://flask.palletsprojects.com/en/stable/debugging/)

**Web Development:**
- [Chrome DevTools Guide](https://developer.chrome.com/docs/devtools/)
- [Network Tab Tutorial](https://developer.chrome.com/docs/devtools/network/)

**API Debugging:**
- [Postman Tutorial](https://learning.postman.com/docs/getting-started/introduction/)
- [cURL Basics](https://curl.se/docs/manual.html)

---

## ✅ Pre-Deployment Checklist

Before going to production:

- [ ] Change SECRET_KEY to random value
- [ ] Set DEBUG=False
- [ ] Use production Qdrant instance
- [ ] Set up HTTPS/SSL
- [ ] Configure proper CORS origins
- [ ] Add rate limiting
- [ ] Set up logging to file
- [ ] Configure backup strategy
- [ ] Test all features
- [ ] Load test with 100+ users
- [ ] Set up monitoring (Sentry, etc.)

---

**Need immediate help? Check our GitHub Issues or Discord server!**