/**
 * Hospital Clinical & Claims Intelligence Platform Frontend
 * Enterprise Master-Detail Split Workbench Architecture
 */

// Global State
const state = {
  currentTab: 'tab-metrics',
  dataset: [],
  filteredTotal: 0,
  datasetLimit: 20,
  datasetOffset: 0,
  selectedIndex: 0,
  metrics: null
};

// DOM Elements Reference Cache
const elements = {
  navTabs: document.querySelectorAll('.nav-tab-btn'),
  tabPanes: document.querySelectorAll('.tab-pane'),
  systemStatusText: document.getElementById('system-status-text'),

  // Tab 1: Validation & Metrics
  kpiModelAAcc: document.getElementById('kpi-model-a-acc'),
  kpiModelARecall: document.getElementById('kpi-model-a-recall'),
  kpiModelBAcc: document.getElementById('kpi-model-b-acc'),
  kpiModelBRecall: document.getElementById('kpi-model-b-recall'),
  modelACmContainer: document.getElementById('model-a-cm-container'),
  modelBCmContainer: document.getElementById('model-b-cm-container'),
  modelAFeaturesList: document.getElementById('model-a-features-list'),
  modelBFeaturesList: document.getElementById('model-b-features-list'),
  fairnessTableBody: document.getElementById('fairness-table-body'),

  // Tab 2: Cohort Master-Detail
  searchInput: document.getElementById('dataset-search-input'),
  filterDept: document.getElementById('filter-department'),
  filterRisk: document.getElementById('filter-risk'),
  filterClaim: document.getElementById('filter-claim'),
  btnResetFilters: document.getElementById('btn-reset-filters'),
  datasetTbody: document.getElementById('sample-dataset-tbody'),
  paginationSummary: document.getElementById('pagination-summary'),
  btnPagePrev: document.getElementById('btn-page-prev'),
  btnPageNext: document.getElementById('btn-page-next'),
  btnRunBatchEval: document.getElementById('btn-run-batch-eval'),
  btnGenerateSamples: document.getElementById('btn-generate-samples'),

  // Tab 2: Docked Inspector (Right Pane)
  inspectorTitle: document.getElementById('inspector-encounter-title'),
  inspectorTag: document.getElementById('inspector-patient-tag'),
  inspDemographics: document.getElementById('insp-demographics'),
  inspDeptType: document.getElementById('insp-dept-type'),
  inspStay: document.getElementById('insp-stay'),
  inspChronic: document.getElementById('insp-chronic'),
  inspRiskTag: document.getElementById('insp-risk-tag'),
  inspRiskCompare: document.getElementById('insp-risk-compare'),
  inspBarRiskHigh: document.getElementById('insp-bar-risk-high'),
  inspPctRiskHigh: document.getElementById('insp-pct-risk-high'),
  inspBarRiskMed: document.getElementById('insp-bar-risk-med'),
  inspPctRiskMed: document.getElementById('insp-pct-risk-med'),
  inspBarRiskLow: document.getElementById('insp-bar-risk-low'),
  inspPctRiskLow: document.getElementById('insp-pct-risk-low'),
  inspClaimTag: document.getElementById('insp-claim-tag'),
  inspBillPayer: document.getElementById('insp-bill-payer'),
  inspClaimCompare: document.getElementById('insp-claim-compare'),
  inspBarClaimPaid: document.getElementById('insp-bar-claim-paid'),
  inspPctClaimPaid: document.getElementById('insp-pct-claim-paid'),
  inspBarClaimPending: document.getElementById('insp-bar-claim-pending'),
  inspPctClaimPending: document.getElementById('insp-pct-claim-pending'),
  inspBarClaimRejected: document.getElementById('insp-bar-claim-rejected'),
  inspPctClaimRejected: document.getElementById('insp-pct-claim-rejected'),
  btnInspectorLoadSim: document.getElementById('btn-inspector-load-sim'),

  // Tab 3: Simulator Form
  simForm: document.getElementById('simulator-form'),
  simAge: document.getElementById('sim-age'),
  simGender: document.getElementById('sim-gender'),
  simCity: document.getElementById('sim-city'),
  simDepartment: document.getElementById('sim-department'),
  simVisittype: document.getElementById('sim-visittype'),
  simStay: document.getElementById('sim-stay'),
  simChronic: document.getElementById('sim-chronic'),
  simProvider: document.getElementById('sim-provider'),
  simBill: document.getElementById('sim-bill'),
  btnClearSim: document.getElementById('btn-clear-sim'),

  // Tab 3: Presets
  presetIcuCardiac: document.getElementById('preset-icu-cardiac'),
  presetOutpatient: document.getElementById('preset-outpatient'),
  presetEmergencyTrauma: document.getElementById('preset-emergency-trauma'),
  presetHighDenial: document.getElementById('preset-high-denial'),

  // Tab 3: Simulator Outputs
  simLatencyBadge: document.getElementById('sim-latency-badge'),
  badgeSimRisk: document.getElementById('badge-sim-risk'),
  barRiskHigh: document.getElementById('bar-risk-high'),
  pctRiskHigh: document.getElementById('pct-risk-high'),
  barRiskMed: document.getElementById('bar-risk-med'),
  pctRiskMed: document.getElementById('pct-risk-med'),
  barRiskLow: document.getElementById('bar-risk-low'),
  pctRiskLow: document.getElementById('pct-risk-low'),
  recBoxRisk: document.getElementById('rec-box-risk'),
  recListRisk: document.getElementById('rec-list-risk'),

  badgeSimClaim: document.getElementById('badge-sim-claim'),
  barClaimPaid: document.getElementById('bar-claim-paid'),
  pctClaimPaid: document.getElementById('pct-claim-paid'),
  barClaimPending: document.getElementById('bar-claim-pending'),
  pctClaimPending: document.getElementById('pct-claim-pending'),
  barClaimRejected: document.getElementById('bar-claim-rejected'),
  pctClaimRejected: document.getElementById('pct-claim-rejected'),
  recBoxClaim: document.getElementById('rec-box-claim'),
  recListClaim: document.getElementById('rec-list-claim'),

  // Tab 4: Audit Table
  auditLogsTbody: document.getElementById('audit-logs-tbody'),
  btnRefreshLogs: document.getElementById('btn-refresh-logs'),

  // Modals
  batchModal: document.getElementById('batch-modal'),
  batchModalContent: document.getElementById('batch-modal-content'),
  btnCloseBatchModal: document.getElementById('btn-close-batch-modal'),

  toastContainer: document.getElementById('toast-container')
};

// Utilities
function showToast(message, type = 'info') {
  const toast = document.createElement('div');
  toast.className = 'toast';
  const icon = type === 'success'
    ? '<svg class="ui-icon sm" style="color: var(--tag-success-text);" viewBox="0 0 24 24"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>'
    : (type === 'error'
        ? '<svg class="ui-icon sm" style="color: var(--tag-danger-text);" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>'
        : '<svg class="ui-icon sm" style="color: var(--accent-blue);" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>');
  toast.innerHTML = `<span>${icon}</span><span>${message}</span>`;
  elements.toastContainer.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(8px)';
    setTimeout(() => toast.remove(), 250);
  }, 3200);
}

function getTagClass(val) {
  if (!val) return 'neutral';
  const v = val.toLowerCase();
  if (v.includes('high') || v.includes('reject')) return 'high';
  if (v.includes('medium') || v.includes('pend')) return 'medium';
  return 'low';
}

function formatCurrency(num) {
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(num);
}

// Navigation Tabs
function initNavigation() {
  elements.navTabs.forEach(tabBtn => {
    tabBtn.addEventListener('click', () => {
      const targetId = tabBtn.getAttribute('data-tab');
      elements.navTabs.forEach(b => b.classList.remove('active'));
      elements.tabPanes.forEach(p => p.classList.remove('active'));
      tabBtn.classList.add('active');
      const pane = document.getElementById(targetId);
      if (pane) pane.classList.add('active');
      state.currentTab = targetId;

      if (targetId === 'tab-audit') {
        fetchAuditLogs();
      }
    });
  });
}

// =========================================================================
// TAB 1: VALIDATION & METRICS
// =========================================================================
async function fetchMetrics() {
  try {
    const res = await fetch('/api/metrics');
    if (!res.ok) throw new Error('Failed to load metrics API');
    const data = await res.json();
    state.metrics = data;
    renderMetrics(data);
  } catch (err) {
    console.error('Metrics error:', err);
  }
}

function renderConfusionMatrix(matrixData, containerId) {
  const container = document.getElementById(containerId);
  if (!container || !matrixData) return;

  const { labels, matrix } = matrixData;
  let html = `<table class="cm-table">
    <thead>
      <tr>
        <th style="text-align: left;">Actual \\ Pred</th>`;
  labels.forEach(l => { html += `<th>${l}</th>`; });
  html += `</tr></thead><tbody>`;

  matrix.forEach((row, rIdx) => {
    html += `<tr><th style="text-align: left;">${labels[rIdx]}</th>`;
    row.forEach((val, cIdx) => {
      const isDiag = rIdx === cIdx;
      let cellClass = 'off-diagonal-low';
      if (isDiag) {
        cellClass = 'diagonal';
      } else if (val > 100) {
        cellClass = 'off-diagonal-high';
      }
      html += `<td><div class="cm-cell ${cellClass}">${val.toLocaleString()}</div></td>`;
    });
    html += `</tr>`;
  });

  html += `</tbody></table>`;
  container.innerHTML = html;
}

function renderMetrics(data) {
  if (!data) return;
  const mA = data.model_a;
  const mB = data.model_b;

  if (mA) {
    elements.kpiModelAAcc.textContent = `${(mA.test_accuracy * 100).toFixed(2)}%`;
    elements.kpiModelARecall.textContent = `${(mA.high_risk_recall * 100).toFixed(2)}%`;
    renderConfusionMatrix(mA.confusion_matrix_test, 'model-a-cm-container');

    elements.modelAFeaturesList.innerHTML = mA.top_features.slice(0, 6).map(f => {
      const pct = (f.importance * 100).toFixed(1);
      return `
        <div class="feature-row">
          <div class="feature-meta">
            <span class="feature-name">${f.feature}</span>
            <span class="feature-score">${pct}%</span>
          </div>
          <div class="feature-track">
            <div class="feature-bar" style="width: ${pct}%;"></div>
          </div>
        </div>
      `;
    }).join('');
  }

  if (mB) {
    elements.kpiModelBAcc.textContent = `${(mB.test_accuracy * 100).toFixed(2)}%`;
    elements.kpiModelBRecall.textContent = `${(mB.rejected_recall * 100).toFixed(2)}%`;
    renderConfusionMatrix(mB.confusion_matrix_test, 'model-b-cm-container');

    elements.modelBFeaturesList.innerHTML = mB.top_features.slice(0, 6).map(f => {
      const pct = (f.importance * 100).toFixed(1);
      return `
        <div class="feature-row">
          <div class="feature-meta">
            <span class="feature-name">${f.feature}</span>
            <span class="feature-score" style="color: var(--accent-indigo);">${pct}%</span>
          </div>
          <div class="feature-track">
            <div class="feature-bar secondary" style="width: ${pct}%;"></div>
          </div>
        </div>
      `;
    }).join('');
  }

  // Fairness Table
  const fairnessRows = [
    { seg: 'Gender: Female', grp: 'Female', a_rec: '94.12%', b_cap: '86.4%', status: 'Parity Met (0.8% Gap)' },
    { seg: 'Gender: Male', grp: 'Male', a_rec: '93.35%', b_cap: '87.0%', status: 'Parity Met (0.8% Gap)' },
    { seg: 'Carrier: CareOne (Disputed)', grp: 'CareOne', a_rec: '93.65%', b_cap: '88.41%', status: 'High Denial Sensitivity' },
    { seg: 'Carrier: MediCareX (Disputed)', grp: 'MediCareX', a_rec: '93.48%', b_cap: '86.90%', status: 'High Denial Sensitivity' },
    { seg: 'Carrier: HealthPlus', grp: 'HealthPlus', a_rec: '93.92%', b_cap: '82.50%', status: 'Low Baseline Disputes' },
    { seg: 'Carrier: SecureLife', grp: 'SecureLife', a_rec: '93.88%', b_cap: '83.33%', status: 'Low Baseline Disputes' }
  ];

  elements.fairnessTableBody.innerHTML = fairnessRows.map(r => `
    <tr>
      <td style="font-weight: 600;">${r.seg}</td>
      <td style="color: var(--text-secondary);">${r.grp}</td>
      <td><span class="status-tag low">${r.a_rec}</span></td>
      <td><span class="status-tag ${r.b_cap.startsWith('88') || r.b_cap.startsWith('86') ? 'high' : 'medium'}">${r.b_cap}</span></td>
      <td><span class="status-tag low">${r.status}</span></td>
    </tr>
  `).join('');
}

// =========================================================================
// TAB 2: MASTER-DETAIL COHORT WORKBENCH
// =========================================================================
async function fetchDataset() {
  try {
    const params = new URLSearchParams({
      limit: state.datasetLimit,
      offset: state.datasetOffset
    });

    const search = elements.searchInput.value.trim();
    if (search) params.append('search', search);

    const dept = elements.filterDept.value;
    if (dept && dept !== 'All') params.append('department', dept);

    const risk = elements.filterRisk.value;
    if (risk && risk !== 'All') params.append('risk_score', risk);

    const claim = elements.filterClaim.value;
    if (claim && claim !== 'All') params.append('claim_status', claim);

    const res = await fetch(`/api/dataset?${params.toString()}`);
    if (!res.ok) throw new Error('Dataset error');
    const data = await res.json();

    state.dataset = data.data;
    state.filteredTotal = data.total;
    renderMasterTable();

    // Automatically inspect the first row in the grid
    if (state.dataset.length > 0) {
      selectEncounter(0);
    }
  } catch (err) {
    console.error('Dataset error:', err);
  }
}

function renderMasterTable() {
  const tbody = elements.datasetTbody;
  if (!tbody) return;

  if (state.dataset.length === 0) {
    tbody.innerHTML = `<tr><td colspan="12" style="text-align: center; padding: 24px; color: var(--text-muted);">No encounters match selected filters.</td></tr>`;
    elements.paginationSummary.textContent = `0 encounters`;
    return;
  }

  tbody.innerHTML = state.dataset.map((row, idx) => {
    const riskActClass = getTagClass(row.risk_score);
    const riskPredClass = getTagClass(row.predicted_risk);
    const claimActClass = getTagClass(row.claim_status);
    const claimPredClass = getTagClass(row.predicted_claim_status);

    const isSelected = idx === state.selectedIndex;

    return `
      <tr class="${isSelected ? 'selected' : ''}" onclick="selectEncounter(${idx})">
        <td style="font-family: var(--font-mono); font-weight: 700; color: var(--accent-blue);">${row.patient_id}</td>
        <td style="font-family: var(--font-mono); color: var(--text-muted);">${row.visit_id}</td>
        <td>${row.age}/${row.gender === 'Female' ? 'F' : 'M'}</td>
        <td>${row.city}</td>
        <td><span class="status-tag neutral">${row.department}</span></td>
        <td>${row.length_of_stay_hours}h</td>
        <td><span class="status-tag ${riskActClass}">${row.risk_score}</span></td>
        <td><span class="status-tag ${riskPredClass}">${row.predicted_risk || '-'}</span></td>
        <td style="font-family: var(--font-mono); font-weight: 600;">${formatCurrency(row.billed_amount)}</td>
        <td style="color: var(--text-secondary);">${row.insurance_provider}</td>
        <td><span class="status-tag ${claimActClass}">${row.claim_status}</span></td>
        <td><span class="status-tag ${claimPredClass}">${row.predicted_claim_status || '-'}</span></td>
      </tr>
    `;
  }).join('');

  const start = state.datasetOffset + 1;
  const end = Math.min(state.datasetOffset + state.dataset.length, state.filteredTotal);
  elements.paginationSummary.textContent = `Showing ${start}-${end} of ${state.filteredTotal}`;
  elements.btnPagePrev.disabled = state.datasetOffset <= 0;
  elements.btnPageNext.disabled = state.datasetOffset + state.datasetLimit >= state.filteredTotal;
}

// Master row selection updates right-side Docked Inspector
window.selectEncounter = function(index) {
  state.selectedIndex = index;
  const rows = elements.datasetTbody.querySelectorAll('tr');
  rows.forEach((r, i) => {
    if (i === index) r.classList.add('selected');
    else r.classList.remove('selected');
  });

  const row = state.dataset[index];
  if (!row) return;

  elements.inspectorTitle.textContent = `${row.patient_id} (${row.visit_id})`;
  elements.inspectorTag.textContent = `${row.department} • ${row.visit_type}`;

  elements.inspDemographics.textContent = `${row.age} yrs, ${row.gender}, ${row.city}`;
  elements.inspDeptType.textContent = `${row.department} (${row.visit_type})`;
  elements.inspStay.textContent = `${row.length_of_stay_hours} Hours`;
  elements.inspChronic.textContent = row.chronic_flag === 1 ? 'Present (Flag = 1)' : 'None';

  // Model A
  const riskClass = getTagClass(row.predicted_risk);
  elements.inspRiskTag.className = `status-tag ${riskClass}`;
  elements.inspRiskTag.textContent = `${row.predicted_risk} Risk`;
  elements.inspRiskCompare.textContent = `${row.risk_score} (Act) vs ${row.predicted_risk} (Pred)`;

  const rHigh = ((row.prob_risk_high || 0) * 100).toFixed(1);
  const rMed = ((row.prob_risk_medium || 0) * 100).toFixed(1);
  const rLow = ((row.prob_risk_low || 0) * 100).toFixed(1);

  elements.inspBarRiskHigh.style.width = `${rHigh}%`;
  elements.inspPctRiskHigh.textContent = `${rHigh}%`;
  elements.inspBarRiskMed.style.width = `${rMed}%`;
  elements.inspPctRiskMed.textContent = `${rMed}%`;
  elements.inspBarRiskLow.style.width = `${rLow}%`;
  elements.inspPctRiskLow.textContent = `${rLow}%`;

  // Model B
  const claimClass = getTagClass(row.predicted_claim_status);
  elements.inspClaimTag.className = `status-tag ${claimClass}`;
  elements.inspClaimTag.textContent = row.predicted_claim_status;
  elements.inspBillPayer.textContent = `${formatCurrency(row.billed_amount)} (${row.insurance_provider})`;
  elements.inspClaimCompare.textContent = `${row.claim_status} (Act) vs ${row.predicted_claim_status} (Pred)`;

  const cPaid = ((row.prob_claim_paid || 0) * 100).toFixed(1);
  const cPend = ((row.prob_claim_pending || 0) * 100).toFixed(1);
  const cRej = ((row.prob_claim_rejected || 0) * 100).toFixed(1);

  elements.inspBarClaimPaid.style.width = `${cPaid}%`;
  elements.inspPctClaimPaid.textContent = `${cPaid}%`;
  elements.inspBarClaimPending.style.width = `${cPend}%`;
  elements.inspPctClaimPending.textContent = `${cPend}%`;
  elements.inspBarClaimRejected.style.width = `${cRej}%`;
  elements.inspPctClaimRejected.textContent = `${cRej}%`;

  elements.btnInspectorLoadSim.onclick = () => {
    elements.simAge.value = row.age;
    elements.simGender.value = row.gender;
    elements.simCity.value = row.city;
    elements.simDepartment.value = row.department;
    elements.simVisittype.value = row.visit_type;
    elements.simStay.value = row.length_of_stay_hours;
    elements.simChronic.value = String(row.chronic_flag);
    elements.simProvider.value = row.insurance_provider;
    elements.simBill.value = row.billed_amount;

    document.getElementById('tab-btn-simulator').click();
    showToast(`Loaded Encounter ${row.patient_id} into simulator`, 'info');
  };
};

// Search debounce
let searchTimer = null;
function handleSearchInput() {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => {
    state.datasetOffset = 0;
    fetchDataset();
  }, 220);
}

// =========================================================================
// BATCH EVALUATION
// =========================================================================
async function executeBatchEvaluation() {
  elements.btnRunBatchEval.disabled = true;
  elements.btnRunBatchEval.textContent = 'Evaluating 100 Records...';

  try {
    const res = await fetch('/api/predict/batch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ use_curated_sample: true, limit: 100 })
    });

    if (!res.ok) throw new Error('Batch inference failed');
    const result = await res.json();

    renderBatchResultsModal(result);
  } catch (err) {
    showToast('Batch evaluation failed: ' + err.message, 'error');
  } finally {
    elements.btnRunBatchEval.disabled = false;
    elements.btnRunBatchEval.innerHTML = '<svg class="ui-icon sm" viewBox="0 0 24 24"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg> Run Batch Evaluation';
  }
}

function renderBatchResultsModal(result) {
  const evalA = result.model_a_evaluation;
  const evalB = result.model_b_evaluation;

  elements.batchModalContent.innerHTML = `
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 16px;">
      <div style="background: var(--panel-surface); padding: 10px; border-radius: var(--radius-sm); border: 1px solid var(--border-hairline);">
        <div style="font-size: 0.68rem; color: var(--text-muted); text-transform: uppercase;">Evaluated</div>
        <div style="font-size: 1.4rem; font-weight: 800; font-family: var(--font-mono);">${result.records_evaluated}</div>
      </div>
      <div style="background: var(--panel-surface); padding: 10px; border-radius: var(--radius-sm); border: 1px solid var(--border-hairline);">
        <div style="font-size: 0.68rem; color: var(--text-muted); text-transform: uppercase;">Model A Accuracy</div>
        <div style="font-size: 1.4rem; font-weight: 800; font-family: var(--font-mono); color: var(--tag-success-text);">${((evalA.accuracy || 0) * 100).toFixed(1)}%</div>
      </div>
      <div style="background: var(--panel-surface); padding: 10px; border-radius: var(--radius-sm); border: 1px solid var(--border-hairline);">
        <div style="font-size: 0.68rem; color: var(--text-muted); text-transform: uppercase;">Model B Accuracy</div>
        <div style="font-size: 1.4rem; font-weight: 800; font-family: var(--font-mono); color: var(--tag-success-text);">${((evalB.accuracy || 0) * 100).toFixed(1)}%</div>
      </div>
      <div style="background: var(--panel-surface); padding: 10px; border-radius: var(--radius-sm); border: 1px solid var(--border-hairline);">
        <div style="font-size: 0.68rem; color: var(--text-muted); text-transform: uppercase;">Latency</div>
        <div style="font-size: 1.4rem; font-weight: 800; font-family: var(--font-mono);">${result.latency_ms} ms</div>
      </div>
    </div>

    <div class="panel-grid-2" style="margin-bottom: 14px;">
      <div>
        <div style="font-size: 0.72rem; font-weight: 700; color: var(--text-muted); margin-bottom: 6px;">MODEL A LIVE CONFUSION MATRIX</div>
        <div id="batch-cm-a"></div>
      </div>
      <div>
        <div style="font-size: 0.72rem; font-weight: 700; color: var(--text-muted); margin-bottom: 6px;">MODEL B LIVE CONFUSION MATRIX</div>
        <div id="batch-cm-b"></div>
      </div>
    </div>

    <div style="text-align: right;">
      <button class="btn-ctrl" onclick="closeBatchModal()">Dismiss</button>
    </div>
  `;

  renderConfusionMatrix(evalA, 'batch-cm-a');
  renderConfusionMatrix(evalB, 'batch-cm-b');
  elements.batchModal.classList.add('active');
}

function closeBatchModal() {
  elements.batchModal.classList.remove('active');
}

// Generate Samples
async function generateFreshSamples() {
  elements.btnGenerateSamples.disabled = true;
  elements.btnGenerateSamples.textContent = 'Generating...';

  try {
    const res = await fetch('/api/dataset/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ count: 5 })
    });

    if (!res.ok) throw new Error('Generation failed');
    const result = await res.json();

    state.dataset = [...result.records, ...state.dataset];
    state.filteredTotal += result.generated_count;
    renderMasterTable();
    selectEncounter(0);
    showToast(`Synthesized ${result.generated_count} encounters`, 'success');
  } catch (err) {
    showToast('Failed to generate encounters', 'error');
  } finally {
    elements.btnGenerateSamples.disabled = false;
    elements.btnGenerateSamples.innerHTML = '<svg class="ui-icon sm" viewBox="0 0 24 24"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg> Generate Fresh Sample';
  }
}

// =========================================================================
// TAB 3: SIMULATOR & PRESETS
// =========================================================================
function initPresets() {
  elements.presetIcuCardiac.addEventListener('click', () => {
    elements.simAge.value = 68;
    elements.simGender.value = 'Male';
    elements.simCity.value = 'Mumbai';
    elements.simDepartment.value = 'Cardiology';
    elements.simVisittype.value = 'ICU';
    elements.simStay.value = 96;
    elements.simChronic.value = '1';
    elements.simProvider.value = 'CareOne';
    elements.simBill.value = 85000;
    executeDualSimulation();
  });

  elements.presetOutpatient.addEventListener('click', () => {
    elements.simAge.value = 34;
    elements.simGender.value = 'Female';
    elements.simCity.value = 'Delhi';
    elements.simDepartment.value = 'General';
    elements.simVisittype.value = 'OPD';
    elements.simStay.value = 3;
    elements.simChronic.value = '0';
    elements.simProvider.value = 'HealthPlus';
    elements.simBill.value = 3200;
    executeDualSimulation();
  });

  elements.presetEmergencyTrauma.addEventListener('click', () => {
    elements.simAge.value = 42;
    elements.simGender.value = 'Male';
    elements.simCity.value = 'Bangalore';
    elements.simDepartment.value = 'Orthopedics';
    elements.simVisittype.value = 'ER';
    elements.simStay.value = 18;
    elements.simChronic.value = '0';
    elements.simProvider.value = 'MediCareX';
    elements.simBill.value = 48000;
    executeDualSimulation();
  });

  elements.presetHighDenial.addEventListener('click', () => {
    elements.simAge.value = 61;
    elements.simGender.value = 'Female';
    elements.simCity.value = 'Chennai';
    elements.simDepartment.value = 'ER';
    elements.simVisittype.value = 'ER';
    elements.simStay.value = 30;
    elements.simChronic.value = '1';
    elements.simProvider.value = 'CareOne';
    elements.simBill.value = 72000;
    executeDualSimulation();
  });

  elements.btnClearSim.addEventListener('click', () => {
    elements.simForm.reset();
  });

  elements.simForm.addEventListener('submit', (e) => {
    e.preventDefault();
    executeDualSimulation();
  });
}

async function executeDualSimulation() {
  const age = parseInt(elements.simAge.value);
  const gender = elements.simGender.value;
  const city = elements.simCity.value;
  const department = elements.simDepartment.value;
  const visit_type = elements.simVisittype.value;
  const length_of_stay_hours = parseFloat(elements.simStay.value);
  const chronic_flag = parseInt(elements.simChronic.value);
  const insurance_provider = elements.simProvider.value;
  const billed_amount = parseFloat(elements.simBill.value);

  const payloadRisk = {
    age, gender, city, department, visit_type,
    length_of_stay_hours, chronic_flag, insurance_provider
  };

  const payloadClaim = {
    ...payloadRisk,
    billed_amount
  };

  try {
    const resA = await fetch('/api/predict/risk', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payloadRisk)
    });
    if (!resA.ok) throw new Error('Model A prediction error');
    const dataA = await resA.json();

    payloadClaim.risk_score = dataA.predicted_risk;
    const resB = await fetch('/api/predict/claim', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payloadClaim)
    });
    if (!resB.ok) throw new Error('Model B prediction error');
    const dataB = await resB.json();

    renderSimulationResults(dataA, dataB);
  } catch (err) {
    showToast('Simulation error: ' + err.message, 'error');
  }
}

function renderSimulationResults(dataA, dataB) {
  const totalLatency = (dataA.latency_ms + dataB.latency_ms).toFixed(1);
  elements.simLatencyBadge.textContent = `Latency: ${totalLatency} ms`;

  // Model A
  const risk = dataA.predicted_risk;
  elements.badgeSimRisk.className = `status-tag ${getTagClass(risk)}`;
  elements.badgeSimRisk.textContent = `${risk} Risk`;

  const pHigh = ((dataA.probabilities.High || 0) * 100).toFixed(1);
  const pMed = ((dataA.probabilities.Medium || 0) * 100).toFixed(1);
  const pLow = ((dataA.probabilities.Low || 0) * 100).toFixed(1);

  elements.barRiskHigh.style.width = `${pHigh}%`;
  elements.pctRiskHigh.textContent = `${pHigh}%`;
  elements.barRiskMed.style.width = `${pMed}%`;
  elements.pctRiskMed.textContent = `${pMed}%`;
  elements.barRiskLow.style.width = `${pLow}%`;
  elements.pctRiskLow.textContent = `${pLow}%`;

  const recAClass = risk === 'High' ? 'danger' : (risk === 'Medium' ? 'warning' : 'success');
  elements.recBoxRisk.className = `output-protocol-box ${recAClass}`;
  elements.recListRisk.innerHTML = dataA.clinical_recommendations.map(r => `<li>${r}</li>`).join('');

  // Model B
  const claim = dataB.predicted_claim_status;
  elements.badgeSimClaim.className = `status-tag ${getTagClass(claim)}`;
  elements.badgeSimClaim.textContent = claim;

  const pPaid = ((dataB.probabilities.Paid || 0) * 100).toFixed(1);
  const pPend = ((dataB.probabilities.Pending || 0) * 100).toFixed(1);
  const pRej = ((dataB.probabilities.Rejected || 0) * 100).toFixed(1);

  elements.barClaimPaid.style.width = `${pPaid}%`;
  elements.pctClaimPaid.textContent = `${pPaid}%`;
  elements.barClaimPending.style.width = `${pPend}%`;
  elements.pctClaimPending.textContent = `${pPend}%`;
  elements.barClaimRejected.style.width = `${pRej}%`;
  elements.pctClaimRejected.textContent = `${pRej}%`;

  const recBClass = claim === 'Rejected' ? 'danger' : (claim === 'Pending' ? 'warning' : 'success');
  elements.recBoxClaim.className = `output-protocol-box ${recBClass}`;
  elements.recListClaim.innerHTML = dataB.billing_recommendations.map(r => `<li>${r}</li>`).join('');
}

// =========================================================================
// TAB 4: AUDIT LOGS TABLE
// =========================================================================
async function fetchAuditLogs() {
  try {
    const res = await fetch('/api/logs');
    if (!res.ok) return;
    const data = await res.json();
    renderAuditLogs(data.logs);
  } catch (err) {
    console.error('Audit logs error:', err);
  }
}

function renderAuditLogs(logs) {
  const tbody = elements.auditLogsTbody;
  if (!tbody) return;

  if (!logs || logs.length === 0) {
    tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; padding: 20px; color: var(--text-muted);">No transactions logged yet. Run predictions in Simulator to populate.</td></tr>`;
    return;
  }

  tbody.innerHTML = logs.map(l => {
    let outcomeTag = '--';
    if (l.output.predicted_risk) {
      outcomeTag = `<span class="status-tag ${getTagClass(l.output.predicted_risk)}">${l.output.predicted_risk} Risk</span>`;
    } else if (l.output.predicted_claim_status) {
      outcomeTag = `<span class="status-tag ${getTagClass(l.output.predicted_claim_status)}">${l.output.predicted_claim_status}</span>`;
    }

    const inputPreview = `${l.inputs.department || 'General'} • Age ${l.inputs.age || '-'} • ${l.inputs.visit_type || 'OPD'}`;

    return `
      <tr>
        <td style="font-family: var(--font-mono); color: var(--text-muted);">${l.timestamp}</td>
        <td style="font-family: var(--font-mono); color: var(--accent-blue);">${l.endpoint}</td>
        <td>${inputPreview}</td>
        <td>${outcomeTag}</td>
        <td style="font-family: var(--font-mono);">${l.latency_ms} ms</td>
      </tr>
    `;
  }).join('');
}

// Health Poller
async function checkHealth() {
  try {
    const res = await fetch('/api/health');
    if (res.ok) {
      const data = await res.json();
      elements.systemStatusText.textContent = `FastAPI v1.0 • ${data.models.model_a.algorithm} & ${data.models.model_b.algorithm}`;
    }
  } catch (e) {
    elements.systemStatusText.textContent = `API Disconnected`;
  }
}

// Initialization
document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
  initPresets();

  fetchMetrics();
  fetchDataset();

  elements.searchInput.addEventListener('input', handleSearchInput);
  elements.filterDept.addEventListener('change', () => { state.datasetOffset = 0; fetchDataset(); });
  elements.filterRisk.addEventListener('change', () => { state.datasetOffset = 0; fetchDataset(); });
  elements.filterClaim.addEventListener('change', () => { state.datasetOffset = 0; fetchDataset(); });
  elements.btnResetFilters.addEventListener('click', () => {
    elements.searchInput.value = '';
    elements.filterDept.value = 'All';
    elements.filterRisk.value = 'All';
    elements.filterClaim.value = 'All';
    state.datasetOffset = 0;
    fetchDataset();
  });

  elements.btnPagePrev.addEventListener('click', () => {
    if (state.datasetOffset > 0) {
      state.datasetOffset = Math.max(0, state.datasetOffset - state.datasetLimit);
      fetchDataset();
    }
  });

  elements.btnPageNext.addEventListener('click', () => {
    if (state.datasetOffset + state.datasetLimit < state.filteredTotal) {
      state.datasetOffset += state.datasetLimit;
      fetchDataset();
    }
  });

  elements.btnRunBatchEval.addEventListener('click', executeBatchEvaluation);
  elements.btnGenerateSamples.addEventListener('click', generateFreshSamples);

  elements.btnCloseBatchModal.addEventListener('click', closeBatchModal);
  elements.batchModal.addEventListener('click', (e) => { if (e.target === elements.batchModal) closeBatchModal(); });

  elements.btnRefreshLogs.addEventListener('click', fetchAuditLogs);

  checkHealth();
  setInterval(checkHealth, 30000);
});
