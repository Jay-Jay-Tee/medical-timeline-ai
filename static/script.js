async function runAnalysis() {
  const patientId = document.getElementById("patientId").value;
  const query = document.getElementById("query").value;
  const output = document.getElementById("output");

  output.innerText = "⏳ Analyzing...";

  const response = await fetch("/explain", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      patient_id: patientId,
      query: query
    })
  });

  const data = await response.json();

  output.innerText = `
Change Level: ${data.difference.change_level}
Semantic Shift: ${data.difference.semantic_shift}

Explanation:
${data.explanation}
`;
}
