let ingestInProgress = false;
let analysisInProgress = false;

function localToUTC(datetimeLocalValue) {
  if (!datetimeLocalValue) return null;
  return new Date(datetimeLocalValue).toISOString();
}

function utcToLocal(isoString) {
  if (!isoString) return "N/A";
  return new Date(isoString).toLocaleString();
}

async function runTimelineSummary(btn) {
  btn.disabled = true;
  btn.innerText = "⏳ Summarizing...";

  const patientId = document.getElementById("ingestPatientId").value.trim();
  const patientName = document.getElementById("patientName").value.trim();
  const doctorName = document.getElementById("doctorName").value.trim();
  const eventType = document.getElementById("eventType").value;
  const content = document.getElementById("content").value.trim();


  try {
    const res = await fetch("/timeline-summary", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ patient_id: patientId, query })
    });

    const data = await res.json();

    if (data.error) {
      output.innerText = "❌ " + data.error;
    } else {
      output.innerText =
        "🧾 Patient Timeline Summary\n\n" +
        data.summary;
    }
  } catch {
    output.innerText = "❌ Failed to summarize timeline";
  }

  btn.disabled = false;
  btn.innerText = "Summarize Full Timeline";
}

async function ingestEvent() {
  if (ingestInProgress) return; // 🚫 spam block
  ingestInProgress = true;

  const button = event.target;
  button.disabled = true;
  button.innerText = "⏳ Adding...";

  const patientId = document.getElementById("ingestPatientId").value.trim();
  const eventType = document.getElementById("eventType").value.trim();
  const content = document.getElementById("content").value.trim();
  const timestampInput = document.getElementById("eventTime").value;
  const safePatientName =
  patientName && patientName.length > 0
    ? patientName
    : "Unknown";

const safeDoctorName =
  doctorName && doctorName.length > 0
    ? doctorName
    : "Self";

  if (!patientId || !eventType || !content) {
    alert("Please fill all required fields");
    resetButton(button);
    return;
  }

  const timestamp = localToUTC(timestampInput);

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 15000); // 15s

  try {
    const response = await fetch("/ingest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        patient_id: patientId,
        patient_name: safePatientName,
        doctor_name: safeDoctorName,
        event_type: eventType,
        content: content,
        timestamp: timestamp
      })
        ,
      signal: controller.signal
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new Error("Server error");
    }

    const data = await response.json();
    alert("✅ Event added!");

  } catch (err) {
    if (err.name === "AbortError") {
      alert("⚠️ Add event timed out. Try again.");
    } else {
      alert("❌ Failed to add event");
    }
  }

  resetButton(button);
}

function resetButton(button) {
  ingestInProgress = false;
  button.disabled = false;
  button.innerText = "Add Event";
}


async function runAnalysis() {
  if (analysisInProgress) return; // 🚫 spam block
  analysisInProgress = true;

  const button = event.target;
  button.disabled = true;
  button.innerText = "⏳ Analyzing...";

  const patientId = document.getElementById("patientId").value.trim();
  const query = document.getElementById("query").value.trim();
  const output = document.getElementById("output");

  if (!patientId || !query) {
    output.innerText = "❌ Please enter Patient ID and query";
    resetAnalysisButton(button);
    return;
  }

  output.innerText = "⏳ Running analysis...";

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 20000); // 20s

  try {
    const response = await fetch("/explain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        patient_id: patientId,
        query: query
      }),
      signal: controller.signal
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new Error("Server error");
    }

    const data = await response.json();

    if (data.error) {
      output.innerText = "❌ " + data.error;
      resetAnalysisButton(button);
      return;
    }

    // Optional: local time display
    const fromLocal = utcToLocal(data.difference.time_range.from);
    const toLocal = utcToLocal(data.difference.time_range.to);

    output.innerText = `
Change Level: ${data.difference.change_level}
Semantic Shift: ${data.difference.semantic_shift}

From: ${fromLocal}
To: ${toLocal}

Explanation:
${data.explanation}
`;

  } catch (err) {
    if (err.name === "AbortError") {
      output.innerText = "⚠️ Analysis timed out. Please try again.";
    } else {
      output.innerText = "❌ Failed to run analysis.";
    }
  }

  resetAnalysisButton(button);
}

function resetAnalysisButton(button) {
  analysisInProgress = false;
  button.disabled = false;
  button.innerText = "Analyze";
}


