let currentFile = null;
let stats = window.INITIAL_STATS || {total:0,cats:0,dogs:0,history:[]};
let historyChart = null, ratioChart = null;

const dropZone    = document.getElementById("dropZone");
const fileInput   = document.getElementById("fileInput");
const previewImg  = document.getElementById("previewImg");
const placeholder = document.getElementById("dropPlaceholder");
const classifyBtn = document.getElementById("classifyBtn");
const classifyLbl = document.getElementById("classifyLabel");
const errorMsg    = document.getElementById("errorMsg");
const resultCard  = document.getElementById("resultCard");
const resultPh    = document.getElementById("resultPlaceholder");

document.addEventListener("DOMContentLoaded", () => {
  initCharts(); renderHistory(); updateKPIs();
});

// Drop zone
dropZone.addEventListener("dragover",  e => { e.preventDefault(); dropZone.classList.add("drag"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag"));
dropZone.addEventListener("drop", e => {
  e.preventDefault(); dropZone.classList.remove("drag");
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith("image/")) handleFile(f);
});
fileInput.addEventListener("change", e => handleFile(e.target.files[0]));

function handleFile(file) {
  if (!file) return;
  currentFile = file;
  previewImg.src = URL.createObjectURL(file);
  previewImg.style.display = "block";
  placeholder.style.display = "none";
  classifyBtn.disabled = false;
  hideError();
  resultCard.style.display = "none";
  resultPh.style.display = "flex";
}

function clearAll() {
  currentFile = null;
  previewImg.src = ""; previewImg.style.display = "none";
  placeholder.style.display = "flex";
  classifyBtn.disabled = true;
  classifyLbl.textContent = "Classify Image";
  fileInput.value = "";
  hideError();
  resultCard.style.display = "none";
  resultPh.style.display = "flex";
}

async function classify() {
  if (!currentFile) return;
  hideError();
  classifyBtn.disabled = true;
  classifyLbl.innerHTML = '<span class="dots"><span></span><span></span><span></span></span>';
  const fd = new FormData();
  fd.append("file", currentFile);
  try {
    const res  = await fetch("/predict", {method:"POST", body:fd});
    const data = await res.json();
    if (!res.ok || data.error) { showError(data.error || "Classification failed."); return; }
    showResult(data);
    stats = data.stats;
    updateKPIs(); renderHistory(); updateCharts();
  } catch(e) {
    showError("Network error — is the Flask server running?");
  } finally {
    classifyBtn.disabled = false;
    classifyLbl.textContent = "Classify Image";
  }
}

function showResult(data) {
  const isCat = data.label === "Cat";
  document.getElementById("resultAnimal").textContent = isCat ? "🐱" : "🐶";
  document.getElementById("resultLabel").textContent  = data.label;
  document.getElementById("resultLabel").style.color  = isCat ? "var(--cat)" : "var(--dog)";
  document.getElementById("resultSub").textContent    = `Confidence: ${data.confidence}%`;
  document.getElementById("resultThumb").src = "data:image/jpeg;base64," + data.image_b64;
  setTimeout(() => {
    document.getElementById("catBar").style.width = data.prob_cat + "%";
    document.getElementById("dogBar").style.width = data.prob_dog + "%";
    document.getElementById("catPct").textContent = data.prob_cat + "%";
    document.getElementById("dogPct").textContent = data.prob_dog + "%";
  }, 50);
  resultCard.style.display = "flex";
  resultPh.style.display = "none";
}

function updateKPIs() {
  document.getElementById("kpi-total").textContent = stats.total;
  document.getElementById("kpi-cats").textContent  = stats.cats;
  document.getElementById("kpi-dogs").textContent  = stats.dogs;
}

function renderHistory() {
  const tbody = document.getElementById("historyBody");
  const empty = document.getElementById("historyEmpty");
  const hist  = [...(stats.history||[])].reverse();
  if (!hist.length) { tbody.innerHTML=""; empty.style.display="block"; return; }
  empty.style.display = "none";
  tbody.innerHTML = hist.map((h,i) => {
    const isCat = h.label==="Cat";
    const color = isCat ? "var(--cat)" : "var(--dog)";
    return `<tr>
      <td style="color:var(--text-3)">${hist.length-i}</td>
      <td><span class="badge ${isCat?"badge-cat":"badge-dog"}">${isCat?"🐱":"🐶"} ${h.label}</span></td>
      <td>${h.confidence}%</td>
      <td><div class="mini-bar-track"><div class="mini-bar-fill" style="width:${h.confidence}%;background:${color}"></div></div></td>
    </tr>`;
  }).join("");
}

function clearHistory() { stats.history=[]; renderHistory(); updateCharts(); }

Chart.defaults.color = "#9090b8";
Chart.defaults.font  = {family:"-apple-system,sans-serif"};

function initCharts() {
  historyChart = new Chart(document.getElementById("historyChart").getContext("2d"), {
    type: "line",
    data: buildHistoryData(),
    options: {
      responsive:true,
      plugins:{legend:{display:false}},
      scales:{
        x:{display:false},
        y:{min:50,max:100,ticks:{callback:v=>v+"%"},grid:{color:"#1e1e3a"}},
      },
      elements:{line:{tension:0.4},point:{radius:3}},
    }
  });
  ratioChart = new Chart(document.getElementById("ratioChart").getContext("2d"), {
    type: "doughnut",
    data: buildRatioData(),
    options: {
      responsive:true,
      plugins:{legend:{position:"bottom",labels:{padding:12,usePointStyle:true,pointStyleWidth:10}}},
      cutout:"65%",
    }
  });
}

function buildHistoryData() {
  const h = stats.history||[];
  return {
    labels: h.map((_,i)=>i+1),
    datasets:[{
      data: h.map(x=>x.confidence),
      borderColor: h.map(x=>x.label==="Cat"?"#7f77dd":"#1d9e75"),
      borderWidth:2,
      pointBackgroundColor: h.map(x=>x.label==="Cat"?"#7f77dd":"#1d9e75"),
      fill:false,
    }]
  };
}

function buildRatioData() {
  const cats=stats.cats||0, dogs=stats.dogs||0, empty=cats+dogs===0;
  return {
    labels:["🐱 Cats","🐶 Dogs"],
    datasets:[{
      data: empty?[1,1]:[cats,dogs],
      backgroundColor: empty?["#2a2a4a","#2a2a4a"]:["#7f77dd","#1d9e75"],
      borderColor:"#13132b", borderWidth:3,
    }]
  };
}

function updateCharts() {
  if (historyChart) { historyChart.data=buildHistoryData(); historyChart.update(); }
  if (ratioChart)   { ratioChart.data=buildRatioData();     ratioChart.update();   }
}

function exportStats() {
  const rows=[["#","Label","Confidence (%)"]];
  (stats.history||[]).forEach((h,i)=>rows.push([i+1,h.label,h.confidence]));
  const blob = new Blob([rows.map(r=>r.join(",")).join("\n")],{type:"text/csv"});
  const a = Object.assign(document.createElement("a"),{href:URL.createObjectURL(blob),download:"predictions.csv"});
  a.click();
}

function showError(msg) { errorMsg.textContent=msg; errorMsg.style.display="block"; }
function hideError()    { errorMsg.style.display="none"; }
