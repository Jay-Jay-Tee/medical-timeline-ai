let ingestInProgress = false;
let summaryInProgress = false;
let analysisInProgress = false;

/* =========================
   TIME HELPERS
========================= */

function localToUTC(datetimeLocalValue) {
  if (!datetimeLocalValue) return null;
  return new Date(datetimeLocalValue).toISOString();
}

function utcToLocal(isoString) {
  if (!isoString) return "N/A";
  return new Date(isoString).toLocaleString();
}

/* =========================
   TIMELINE SUMMARY
========================= */

async function runTimelineSummary(btn) {
  if (summaryInProgress || analysisInProgress) return;

  summaryInProgress = true;

  const analyzeBtn = document.querySelector(
    'button[onclick="runAnalysis()"]'
  );

  btn.disabled = true;
  btn.innerText = "⏳ Summarizing...";
  if (analyzeBtn) analyzeBtn.disabled = true;

  const ingestId = document.getElementById("ingestPatientId").value.trim();
  const analysisId = document.getElementById("patientId").value.trim();
  const patientId = ingestId || analysisId;

  const output = document.getElementById("output");

  function cleanup() {
    summaryInProgress = false;
    btn.disabled = false;
    btn.innerText = "Summarize Full Timeline";
    if (analyzeBtn) analyzeBtn.disabled = false;
  }

  if (!patientId) {
    output.innerText = "❌ Please enter Patient ID (Ingest or Analyze section)";
    cleanup();
    return;
  }

  try {
    const res = await fetch("/timeline-summary", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ patient_id: patientId })
    });

    const data = await res.json();

    if (data.error) {
      output.innerText = "❌ " + data.error;
      cleanup();
      return;
    }

    /* ---------- Build Table ---------- */
    let tableHTML = `
      <div class="overflow-x-auto">
        <table class="min-w-full border border-gray-300 text-sm">
          <thead class="bg-gray-100">
            <tr>
              <th class="border px-3 py-2 text-left">Time (Local)</th>
              <th class="border px-3 py-2 text-left">Event Type</th>
              <th class="border px-3 py-2 text-left">Content</th>
            </tr>
          </thead>
          <tbody>
    `;

    for (const row of data.timeline) {
      tableHTML += `
        <tr class="hover:bg-gray-50">
          <td class="border px-3 py-2">${utcToLocal(row.timestamp)}</td>
          <td class="border px-3 py-2">${row.event_type}</td>
          <td class="border px-3 py-2">${row.content}</td>
        </tr>
      `;
    }

    tableHTML += `
          </tbody>
        </table>
      </div>
    `;

    const summaryText =
      data.overall_summary && data.overall_summary.trim().length > 0
        ? data.overall_summary
        : "The records show limited explicit textual differences over time.";

    output.innerHTML = `
      <h3 class="font-semibold mb-2">🗂 Patient Timeline</h3>
      ${tableHTML}

      <hr class="my-4">

      <h3 class="font-semibold mb-1">🧠 AI Patient Overview</h3>
      <p class="mb-2">${summaryText}</p>

      <p class="text-sm text-gray-600">
        Semantic Shift: <strong>${data.semantic_shift}</strong>
      </p>
    `;

  } catch {
    output.innerText = "❌ Failed to summarize timeline";
  }

  cleanup();
}

/* =========================
   INGEST EVENT
========================= */

async function ingestEvent() {
  if (ingestInProgress) return;
  ingestInProgress = true;

  const button = event.target;
  button.disabled = true;
  button.innerText = "⏳ Adding...";

  const patientId = document.getElementById("ingestPatientId").value.trim();
  const patientName = document.getElementById("patientName").value.trim();
  const doctorName = document.getElementById("doctorName").value.trim();
  const eventType = document.getElementById("eventType").value.trim();
  const content = document.getElementById("content").value.trim();
  const timestampInput = document.getElementById("eventTime").value;

  const safePatientName = patientName || "Unknown";
  const safeDoctorName = doctorName || "Self";

  if (!patientId || !eventType || !content) {
    alert("Please fill all required fields");
    resetIngestButton(button);
    return;
  }

  const timestamp = localToUTC(timestampInput);

  try {
    const response = await fetch("/ingest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        patient_id: patientId,
        patient_name: safePatientName,
        doctor_name: safeDoctorName,
        event_type: eventType,
        content,
        timestamp
      })
    });

    if (!response.ok) throw new Error();
    alert("✅ Event added");

  } catch {
    alert("❌ Failed to add event");
  }

  resetIngestButton(button);
}

function resetIngestButton(button) {
  ingestInProgress = false;
  button.disabled = false;
  button.innerText = "Add Event";
}

/* =========================
   ANALYSIS / EXPLAIN
========================= */

async function runAnalysis() {
  if (analysisInProgress || summaryInProgress) return;
  analysisInProgress = true;

  const button = event.target;
  const summaryBtn = document.querySelector(
    'button[onclick^="runTimelineSummary"]'
  );

  button.disabled = true;
  button.innerText = "⏳ Analyzing...";
  if (summaryBtn) summaryBtn.disabled = true;

  const patientId = document.getElementById("patientId").value.trim();
  const query = document.getElementById("query").value.trim();
  const output = document.getElementById("output");

  try {
    if (!patientId || !query) {
      output.innerText = "❌ Please enter Patient ID and query";
      resetAnalysisButton(button);
      return;
    }

    const response = await fetch("/explain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ patient_id: patientId, query })
    });

    const data = await response.json();

    if (data.error) {
      output.innerText = "❌ " + data.error;
    } else {
      output.innerText = `
Change Level: ${data.difference.change_level}
Semantic Shift: ${data.difference.semantic_shift}

From (Local): ${utcToLocal(data.difference.time_range.from)}
To (Local): ${utcToLocal(data.difference.time_range.to)}

Explanation:
${data.explanation || "The records show limited explicit textual differences over time."}
`;
    }
  } catch {
    output.innerText = "❌ Failed to run analysis";
  }

  resetAnalysisButton(button);
}

function resetAnalysisButton(button) {
  analysisInProgress = false;
  button.disabled = false;
  button.innerText = "Analyze";

  const summaryBtn = document.querySelector(
    'button[onclick^="runTimelineSummary"]'
  );
  if (summaryBtn) summaryBtn.disabled = false;
}
