// Global state
let currentUser = null;
let voiceRecognition = null;
let isRecording = false;
let selectedFile = null;

// ==================== AUTH ====================

function showAuthModal() {
  document.getElementById('authModal').classList.add('active');
}

function hideAuthModal() {
  document.getElementById('authModal').classList.remove('active');
}

function showLogin() {
  document.getElementById('loginForm').style.display = 'block';
  document.getElementById('registerForm').style.display = 'none';
  document.getElementById('authTitle').textContent = 'Welcome Back';
}

function showRegister() {
  document.getElementById('loginForm').style.display = 'none';
  document.getElementById('registerForm').style.display = 'block';
  document.getElementById('authTitle').textContent = 'Create Account';
}

async function handleLogin() {
  const email = document.getElementById('loginEmail').value.trim();
  const password = document.getElementById('loginPassword').value;

  if (!email || !password) {
    showNotification('Please fill in all fields', 'error');
    return;
  }

  try {
    const res = await fetch('/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      body: JSON.stringify({ email, password })
    });

    const data = await res.json();

    if (res.ok) {
      currentUser = data;
      updateUIForLoggedInUser();
      hideAuthModal();
      
      document.getElementById('ingestPatientId').value = data.patient_id;
      document.getElementById('patientId').value = data.patient_id;
      
      showNotification(`Welcome back, ${data.username}!`, 'success');
      updateStats();
    } else {
      showNotification(data.error || 'Login failed', 'error');
    }
  } catch (err) {
    showNotification('Login failed: ' + err.message, 'error');
  }
}

async function handleRegister() {
  const username = document.getElementById('regUsername').value.trim();
  const email = document.getElementById('regEmail').value.trim();
  const password = document.getElementById('regPassword').value;

  if (!username || !email || !password) {
    showNotification('Please fill in all fields', 'error');
    return;
  }

  try {
    const res = await fetch('/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      body: JSON.stringify({ 
        username, 
        email, 
        password
      })
    });

    const data = await res.json();

    if (res.ok) {
      currentUser = data;
      updateUIForLoggedInUser();
      hideAuthModal();
      
      document.getElementById('ingestPatientId').value = data.patient_id;
      document.getElementById('patientId').value = data.patient_id;
      
      showNotification(`Account created! Your Patient ID: ${data.patient_id}`, 'success');
    } else {
      showNotification(data.error || 'Registration failed', 'error');
    }
  } catch (err) {
    showNotification('Registration failed: ' + err.message, 'error');
  }
}

async function handleLogout() {
  try {
    await fetch('/logout', { 
      method: 'POST',
      credentials: 'include'
    });
    currentUser = null;
    updateUIForLoggedOutUser();
    showNotification('Signed out successfully', 'success');
  } catch (err) {
    console.error('Logout failed:', err);
  }
}

function updateUIForLoggedInUser() {
  document.getElementById('welcomeText').textContent = `Welcome back, ${currentUser.username}!`;
  document.getElementById('authButton').textContent = 'Sign Out';
  document.getElementById('authButton').onclick = handleLogout;
  document.getElementById('profileBtn').classList.remove('hidden');
}

function updateUIForLoggedOutUser() {
  document.getElementById('welcomeText').textContent = 'Your Personal Health Timeline';
  document.getElementById('authButton').textContent = 'Sign In';
  document.getElementById('authButton').onclick = showAuthModal;
  document.getElementById('ingestPatientId').value = '';
  document.getElementById('patientId').value = '';
  document.getElementById('profileBtn').classList.add('hidden');
  
  const output = document.getElementById('output');
  output.innerHTML = `
    <div class="text-center" style="color: var(--text-muted);">
      <div class="text-4xl mb-3">📋</div>
      <p class="font-medium">Sign in and add events to view your timeline</p>
      <p class="text-sm mt-2">Your medical history will appear here</p>
    </div>
  `;
}

// ==================== DARK MODE ====================

function toggleDarkMode() {
  const html = document.documentElement;
  const icon = document.getElementById('darkModeIcon');
  
  if (html.getAttribute('data-theme') === 'dark') {
    html.removeAttribute('data-theme');
    icon.textContent = '🌙';
    localStorage.setItem('theme', 'light');
  } else {
    html.setAttribute('data-theme', 'dark');
    icon.textContent = '☀️';
    localStorage.setItem('theme', 'dark');
  }
}

// ==================== VOICE INPUT ====================

function initVoiceRecognition() {
  if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    voiceRecognition = new SpeechRecognition();
    voiceRecognition.continuous = true;
    voiceRecognition.interimResults = true;
    voiceRecognition.lang = 'en-US';

    let finalTranscript = '';

    voiceRecognition.onstart = () => {
      console.log('Voice recognition started');
      showNotification('🎤 Listening... Speak now', 'info');
    };

    voiceRecognition.onresult = (event) => {
      let interimTranscript = '';
      
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          finalTranscript += transcript + ' ';
        } else {
          interimTranscript += transcript;
        }
      }
      
      document.getElementById('content').value = finalTranscript + interimTranscript;
    };

    voiceRecognition.onerror = (event) => {
      console.error('Voice recognition error:', event.error);
      if (event.error === 'no-speech') {
        showNotification('No speech detected. Please try again.', 'warning');
      } else if (event.error === 'not-allowed') {
        showNotification('Microphone access denied. Please allow microphone access.', 'error');
      } else {
        showNotification('Voice recognition error: ' + event.error, 'error');
      }
      stopVoice();
    };

    voiceRecognition.onend = () => {
      if (isRecording) {
        // Restart if still recording
        try {
          voiceRecognition.start();
        } catch (e) {
          console.error('Failed to restart recognition:', e);
          stopVoice();
        }
      }
    };

    console.log('Voice recognition initialized');
  } else {
    console.warn('Voice recognition not supported');
  }
}

function toggleVoice() {
  if (!voiceRecognition) {
    showNotification('Voice recognition not supported in this browser. Please use Chrome or Edge.', 'error');
    return;
  }

  if (isRecording) {
    stopVoice();
  } else {
    startVoice();
  }
}

function startVoice() {
  try {
    isRecording = true;
    voiceRecognition.start();
    
    const btn = document.getElementById('voiceButton');
    btn.classList.add('pulse-recording');
    btn.style.backgroundColor = '#ef4444';
    document.getElementById('voiceIcon').textContent = '⏹️';
    
  } catch (e) {
    console.error('Failed to start voice recognition:', e);
    showNotification('Failed to start voice input', 'error');
    stopVoice();
  }
}

function stopVoice() {
  isRecording = false;
  if (voiceRecognition) {
    voiceRecognition.stop();
  }
  
  const btn = document.getElementById('voiceButton');
  btn.classList.remove('pulse-recording');
  btn.style.backgroundColor = '#3b82f6';
  document.getElementById('voiceIcon').textContent = '🎤';
  
  showNotification('Voice input stopped', 'info');
}

// ==================== FILE UPLOAD ====================

function handleFileSelect(event) {
  const file = event.target.files[0];
  handleFile(file);
}

function handleFile(file) {
  if (!file) return;
  
  if (!file.type.startsWith('image/') && file.type !== 'application/pdf') {
    showNotification('Please upload an image or PDF file', 'error');
    return;
  }
  
  if (file.size > 10 * 1024 * 1024) { // 10MB limit
    showNotification('File size must be less than 10MB', 'error');
    return;
  }
  
  selectedFile = file;
  
  if (file.type.startsWith('image/')) {
    const reader = new FileReader();
    reader.onload = (e) => {
      document.getElementById('previewImage').src = e.target.result;
      document.getElementById('uploadPreview').classList.remove('hidden');
    };
    reader.readAsDataURL(file);
  } else {
    // For PDFs, show a placeholder
    document.getElementById('previewImage').src = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200"><rect fill="%23f3f4f6" width="200" height="200"/><text x="50%" y="50%" font-size="60" text-anchor="middle" dy=".3em">📄</text></svg>';
    document.getElementById('uploadPreview').classList.remove('hidden');
  }
}

async function uploadDocument() {
  if (!selectedFile) {
    showNotification('Please select a file first', 'error');
    return;
  }

  const patientId = document.getElementById('ingestPatientId').value || 
                    document.getElementById('patientId').value;
  
  if (!patientId) {
    showNotification('Please sign in first', 'error');
    return;
  }

  const formData = new FormData();
  formData.append('file', selectedFile);
  formData.append('patient_id', patientId);
  
  const notes = document.getElementById('documentNotes').value.trim();
  if (notes) {
    formData.append('notes', notes);
  }

  const doctorName = document.getElementById('doctorName').value.trim() || 'Unknown';
  const hospitalName = document.getElementById('hospitalName').value.trim() || 'Unknown';
  formData.append('doctor_name', doctorName);
  formData.append('hospital_name', hospitalName);
  
  const output = document.getElementById('output');
  output.innerHTML = '<div class="text-center"><div class="text-4xl mb-3">📤</div><p class="font-medium">Uploading document...</p></div>';

  try {
    const res = await fetch('/upload-document', {
      method: 'POST',
      credentials: 'include',
      body: formData
    });

    const data = await res.json();

    if (res.ok) {
      output.innerHTML = `
        <div class="space-y-4">
          <div class="p-4 bg-green-50 dark:bg-green-900 rounded-lg border-2 border-green-200 dark:border-green-700">
            <p class="font-bold text-green-800 dark:text-green-200 flex items-center gap-2">
              <span class="text-2xl">✅</span> Document Uploaded Successfully
            </p>
            <p class="text-sm text-green-700 dark:text-green-300 mt-2">
              <strong>File:</strong> ${escapeHtml(selectedFile.name)}
            </p>
          </div>
          <div class="p-4 rounded-lg border-2" style="background-color: var(--input-bg); border-color: var(--input-border);">
            <p class="font-semibold mb-2">Stored Information:</p>
            <p class="text-sm">${escapeHtml(data.extracted_text)}</p>
            ${data.note ? `<p class="text-xs mt-3 text-yellow-600 dark:text-yellow-400">ℹ️ ${escapeHtml(data.note)}</p>` : ''}
          </div>
        </div>
      `;
      
      document.getElementById('uploadPreview').classList.add('hidden');
      document.getElementById('fileInput').value = '';
      document.getElementById('documentNotes').value = '';
      selectedFile = null;
      
      showNotification('Document uploaded successfully!', 'success');
      updateStats();
    } else {
      output.innerHTML = `<div class="text-center text-red-500"><p class="text-4xl mb-3">❌</p><p class="font-medium">${escapeHtml(data.error)}</p></div>`;
      showNotification(data.error, 'error');
    }
  } catch (err) {
    output.innerHTML = `<div class="text-center text-red-500"><p class="text-4xl mb-3">❌</p><p class="font-medium">Upload failed: ${escapeHtml(err.message)}</p></div>`;
    showNotification('Upload failed: ' + err.message, 'error');
  }
}

// ==================== INGEST EVENT ====================

async function ingestEvent() {
  const patientId = document.getElementById('ingestPatientId').value.trim();
  const eventType = document.getElementById('eventType').value;
  const content = document.getElementById('content').value.trim();
  const doctorName = document.getElementById('doctorName').value.trim() || 'Unknown';
  const hospitalName = document.getElementById('hospitalName').value.trim() || 'Unknown';

  if (!patientId) {
    showNotification('Please sign in first', 'error');
    return;
  }

  if (!content) {
    showNotification('Please enter event details', 'error');
    return;
  }

  try {
    const res = await fetch('/ingest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      body: JSON.stringify({
        patient_id: patientId,
        event_type: eventType,
        content: content,
        doctor_name: doctorName,
        hospital_name: hospitalName,
        timestamp: null
      })
    });

    const data = await res.json();

    if (res.ok) {
      showNotification(`Event added successfully!`, 'success');
      document.getElementById('content').value = '';
      document.getElementById('doctorName').value = '';
      document.getElementById('hospitalName').value = '';
      updateStats();
    } else {
      showNotification(data.error || 'Failed to add event', 'error');
    }
  } catch (err) {
    showNotification('Failed: ' + err.message, 'error');
  }
}

// ==================== TIMELINE SUMMARY ====================

async function runTimelineSummary(btn) {
  const originalText = btn.textContent;
  btn.disabled = true;
  btn.textContent = '⏳ Analyzing...';

  const patientId = document.getElementById('ingestPatientId').value.trim() ||
                    document.getElementById('patientId').value.trim();

  const output = document.getElementById('output');

  if (!patientId) {
    output.innerHTML = '<div class="text-center text-red-500"><p class="text-4xl mb-3">❌</p><p class="font-medium">Please sign in first</p></div>';
    btn.disabled = false;
    btn.textContent = originalText;
    return;
  }

  try {
    const res = await fetch('/timeline-summary', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      body: JSON.stringify({ patient_id: patientId })
    });

    const data = await res.json();

    if (data.error) {
      output.innerHTML = `<div class="text-center text-red-500"><p class="text-4xl mb-3">❌</p><p class="font-medium">${escapeHtml(data.error)}</p></div>`;
    } else {
      const qualityColor = 
        data.data_quality.label === 'Rich' ? 'text-green-600 dark:text-green-400' :
        data.data_quality.label === 'Moderate' ? 'text-yellow-600 dark:text-yellow-400' :
        'text-red-600 dark:text-red-400';

      let tableHTML = `
        <div class="overflow-x-auto">
          <table class="min-w-full text-sm border-2 rounded-lg" style="border-color: var(--input-border);">
            <thead style="background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%); color: white;">
              <tr>
                <th class="border px-4 py-3 text-left font-semibold">Time</th>
                <th class="border px-4 py-3 text-left font-semibold">Type</th>
                <th class="border px-4 py-3 text-left font-semibold">Content</th>
              </tr>
            </thead>
            <tbody>
      `;

      for (const row of data.timeline) {
        const date = new Date(row.timestamp).toLocaleString('en-US', {
          month: 'short',
          day: 'numeric',
          year: 'numeric',
          hour: '2-digit',
          minute: '2-digit'
        });
        tableHTML += `
          <tr class="hover:bg-blue-50 dark:hover:bg-gray-800 transition-colors">
            <td class="border px-4 py-3 font-medium" style="color: var(--text-muted);">${escapeHtml(date)}</td>
            <td class="border px-4 py-3">
              <span class="badge bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200">
                ${escapeHtml(row.event_type)}
              </span>
            </td>
            <td class="border px-4 py-3" style="color: var(--text);">${escapeHtml(row.content)}</td>
          </tr>
        `;
      }

      tableHTML += '</tbody></table></div>';

      output.innerHTML = `
        <h3 class="font-bold text-lg mb-4" style="color: var(--text);">📋 Patient Timeline</h3>
        ${tableHTML}
        <div class="mt-5 p-5 bg-blue-50 dark:bg-blue-900 rounded-lg border-2 border-blue-200 dark:border-blue-700">
          <p class="font-semibold mb-3 flex items-center gap-2" style="color: var(--text);">
            <span class="text-xl">🤖</span> AI Analysis (Powered by Groq)
          </p>
          <p class="text-sm leading-relaxed" style="color: var(--text);">${escapeHtml(data.overall_summary)}</p>
          <div class="mt-4 pt-4 border-t-2 border-blue-200 dark:border-blue-700">
            <p class="text-xs" style="color: var(--text-muted);">
              Semantic Shift: <strong class="text-purple-600 dark:text-purple-400">${data.semantic_shift}</strong>
            </p>
          </div>
        </div>
        <div class="mt-4 p-4 border-2 rounded-lg" style="border-color: var(--input-border); background-color: var(--input-bg);">
          <div class="flex items-center gap-3">
            <span class="font-semibold" style="color: var(--text);">Data Quality:</span>
            <span class="${qualityColor} font-bold">${data.data_quality.label}</span>
          </div>
          <p class="text-sm mt-2" style="color: var(--text-muted);">
            ${data.data_quality.description}
          </p>
        </div>
      `;

      updateStats();
      showNotification('Timeline analysis complete!', 'success');
    }
  } catch (err) {
    output.innerHTML = `<div class="text-center text-red-500"><p class="text-4xl mb-3">❌</p><p class="font-medium">Failed: ${escapeHtml(err.message)}</p></div>`;
    showNotification('Analysis failed: ' + err.message, 'error');
  }

  btn.disabled = false;
  btn.textContent = originalText;
}

// ==================== PDF EXPORT ====================

async function exportPDF() {
  const patientId = document.getElementById('ingestPatientId').value.trim() ||
                    document.getElementById('patientId').value.trim();

  if (!patientId) {
    showNotification('Please sign in first', 'error');
    return;
  }

  showNotification('Generating PDF...', 'info');

  try {
    const res = await fetch('/export-pdf', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      body: JSON.stringify({ patient_id: patientId })
    });

    if (res.ok) {
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `medical_timeline_${patientId}_${new Date().toISOString().split('T')[0]}.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
      
      showNotification('PDF downloaded successfully!', 'success');
    } else {
      const data = await res.json();
      showNotification(data.error || 'Failed to export PDF', 'error');
    }
  } catch (err) {
    showNotification('Export failed: ' + err.message, 'error');
  }
}

// ==================== SHARE PATIENT LINK ====================

function sharePatientLink() {
  const patientId = document.getElementById('ingestPatientId').value.trim() ||
                    document.getElementById('patientId').value.trim();

  if (!patientId) {
    showNotification('Please sign in first', 'error');
    return;
  }

  const link = `${window.location.origin}/patient/${patientId}`;
  
  navigator.clipboard.writeText(link).then(() => {
    showNotification('Shareable link copied to clipboard!', 'success');
  }).catch(() => {
    prompt('Copy this link:', link);
  });
}

// ==================== NOTIFICATIONS ====================

function showNotification(message, type = 'info') {
  const colors = {
    success: 'bg-green-500',
    error: 'bg-red-500',
    warning: 'bg-yellow-500',
    info: 'bg-blue-500'
  };

  const notification = document.createElement('div');
  notification.className = `fixed top-4 right-4 ${colors[type]} text-white px-6 py-3 rounded-lg shadow-lg z-[9999] transition-all transform translate-x-0 opacity-100`;
  notification.style.animation = 'slideIn 0.3s ease-out';
  notification.textContent = message;

  document.body.appendChild(notification);

  setTimeout(() => {
    notification.style.animation = 'slideOut 0.3s ease-out';
    setTimeout(() => notification.remove(), 300);
  }, 3000);
}

// Add CSS for animations
const style = document.createElement('style');
style.textContent = `
  @keyframes slideIn {
    from { transform: translateX(100%); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
  }
  @keyframes slideOut {
    from { transform: translateX(0); opacity: 1; }
    to { transform: translateX(100%); opacity: 0; }
  }
`;
document.head.appendChild(style);

// ==================== UTILS ====================

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

async function updateStats() {
  const patientId = document.getElementById('ingestPatientId').value.trim() ||
                    document.getElementById('patientId').value.trim();
  
  if (!patientId) return;

  try {
    const res = await fetch('/timeline-summary', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      body: JSON.stringify({ patient_id: patientId })
    });
    
    const data = await res.json();
    
    if (data.timeline) {
      document.getElementById('statEvents').textContent = data.timeline.length;
      if (data.timeline.length > 0) {
        const lastDate = new Date(data.timeline[data.timeline.length - 1].timestamp);
        document.getElementById('statLastUpdate').textContent = lastDate.toLocaleDateString('en-US', {
          month: 'short',
          day: 'numeric',
          year: 'numeric'
        });
      }
    }
  } catch (err) {
    console.error('Failed to update stats:', err);
  }
}

// ==================== INITIALIZATION ====================

window.addEventListener('DOMContentLoaded', () => {
  // Load saved theme
  const savedTheme = localStorage.getItem('theme');
  if (savedTheme === 'dark') {
    document.documentElement.setAttribute('data-theme', 'dark');
    document.getElementById('darkModeIcon').textContent = '☀️';
  }

  // Initialize voice recognition
  initVoiceRecognition();

  // Setup drag and drop
  const dropZone = document.getElementById('dropZone');
  
  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'var(--primary)';
    dropZone.style.backgroundColor = 'rgba(59, 130, 246, 0.05)';
  });
  
  dropZone.addEventListener('dragleave', () => {
    dropZone.style.borderColor = 'var(--input-border)';
    dropZone.style.backgroundColor = 'var(--input-bg)';
  });
  
  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'var(--input-border)';
    dropZone.style.backgroundColor = 'var(--input-bg)';
    
    const file = e.dataTransfer.files[0];
    if (file) {
      handleFile(file);
    }
  });

  // Check if user is logged in
  fetch('/me', { credentials: 'include' })
    .then(res => res.json())
    .then(data => {
      if (data.username) {
        currentUser = data;
        updateUIForLoggedInUser();
        document.getElementById('ingestPatientId').value = data.patient_id;
        document.getElementById('patientId').value = data.patient_id;
        updateStats();
      }
    })
    .catch(() => {
      // User not logged in, ignore
    });
});

// ==================== PROFILE MANAGEMENT ====================

function showProfile() {
  if (!currentUser) {
    showNotification('Please sign in first', 'error');
    return;
  }

  // Populate basic info
  document.getElementById('profileName').textContent = currentUser.username;
  document.getElementById('profileEmail').textContent = currentUser.email;
  document.getElementById('profilePatientId').textContent = currentUser.patient_id;

  // Populate health info if available
  if (currentUser.profile_data) {
    document.getElementById('profileAge').value = currentUser.profile_data.age || '';
    document.getElementById('profileGender').value = currentUser.profile_data.gender || '';
    document.getElementById('profileBloodType').value = currentUser.profile_data.blood_type || '';
    document.getElementById('profilePhone').value = currentUser.profile_data.phone || '';
    document.getElementById('profileAllergies').value = currentUser.profile_data.allergies || '';
    document.getElementById('profileConditions').value = currentUser.profile_data.chronic_conditions || '';
    document.getElementById('profileEmergency').value = currentUser.profile_data.emergency_contact || '';
    document.getElementById('profileAddress').value = currentUser.profile_data.address || '';
  }

  document.getElementById('profileModal').classList.add('active');
}

function hideProfile() {
  document.getElementById('profileModal').classList.remove('active');
}

async function saveProfile() {
  const profileData = {
    age: document.getElementById('profileAge').value,
    gender: document.getElementById('profileGender').value,
    blood_type: document.getElementById('profileBloodType').value,
    phone: document.getElementById('profilePhone').value,
    allergies: document.getElementById('profileAllergies').value,
    chronic_conditions: document.getElementById('profileConditions').value,
    emergency_contact: document.getElementById('profileEmergency').value,
    address: document.getElementById('profileAddress').value
  };

  try {
    const res = await fetch('/update-profile', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      body: JSON.stringify({ profile_data: profileData })
    });

    const data = await res.json();

    if (res.ok) {
      currentUser.profile_data = data.profile_data;
      showNotification('Profile updated successfully!', 'success');
    } else {
      showNotification(data.error || 'Failed to update profile', 'error');
    }
  } catch (err) {
    showNotification('Failed to update profile: ' + err.message, 'error');
  }
}

function viewMyTimeline() {
  if (!currentUser) {
    showNotification('Please sign in first', 'error');
    return;
  }

  // Open timeline view in new tab
  window.open(`/patient/${currentUser.patient_id}`, '_blank');
}