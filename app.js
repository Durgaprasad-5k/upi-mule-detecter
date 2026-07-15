/* ═══════════════════════════════════════════════════════════════
   UPI Mule Detector — Dashboard Logic
   ═══════════════════════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {
    // ── Element References ──────────────────────────────────────
    const runBtn        = document.getElementById('run-btn');
    const btnText       = runBtn.querySelector('.btn-text');
    const btnIcon       = runBtn.querySelector('.btn-icon');
    const btnSpinner    = runBtn.querySelector('.btn-spinner');
    const liveIndicator = document.getElementById('live-indicator');
    const liveText      = document.getElementById('live-text');

    const f1ScoreEl     = document.getElementById('f1-score');
    const precisionEl   = document.getElementById('precision');
    const recallEl      = document.getElementById('recall');
    const mulesEl       = document.getElementById('mules-detected');
    const muleBreakdown = document.getElementById('mule-breakdown');
    const f1RingFill    = document.getElementById('f1-ring-fill');

    const cycleCountEl  = document.getElementById('cycle-count');
    const starCountEl   = document.getElementById('star-count');
    const bothCountEl   = document.getElementById('both-count');
    const cycleBar      = document.getElementById('cycle-bar');
    const starBar       = document.getElementById('star-bar');
    const bothBar       = document.getElementById('both-bar');

    const tpCell        = document.getElementById('tp-cell');
    const fpCell        = document.getElementById('fp-cell');
    const fnCell        = document.getElementById('fn-cell');
    const tnCell        = document.getElementById('tn-cell');

    const perfBadge     = document.getElementById('perf-badge');
    const gaugeCanvas   = document.getElementById('perf-gauge');
    const gaugeLabel    = document.getElementById('gauge-label');
    const phase2Time    = document.getElementById('phase2-time');
    const phase3Time    = document.getElementById('phase3-time');

    const resultsTbody  = document.getElementById('results-tbody');
    const searchInput   = document.getElementById('search-input');
    const filterSelect  = document.getElementById('filter-select');

    let allResults = [];

    // ── Init ────────────────────────────────────────────────────
    fetchResults();
    drawGauge(0, 6);

    // ── Run Pipeline ────────────────────────────────────────────
    runBtn.addEventListener('click', async () => {
        setLoading(true);
        try {
            const res = await fetch('/api/run', { method: 'POST' });
            const data = await res.json();
            if (data.status === 'success') {
                updateFullDashboard(data.results);
            } else {
                throw new Error(data.detail || 'Pipeline error');
            }
        } catch (err) {
            console.error(err);
            liveText.textContent = 'ERROR';
            liveIndicator.className = 'live-indicator';
        } finally {
            setLoading(false);
        }
    });

    // ── Search & Filter ─────────────────────────────────────────
    searchInput.addEventListener('input', () => renderTable(filterData()));
    filterSelect.addEventListener('change', () => renderTable(filterData()));

    function filterData() {
        let data = [...allResults];
        const query = searchInput.value.trim().toUpperCase();
        const source = filterSelect.value;
        if (query) data = data.filter(r => r.Account.toUpperCase().includes(query));
        if (source !== 'all') data = data.filter(r => r.DetectionSource === source);
        return data;
    }

    // ── Fetch Existing Results ──────────────────────────────────
    async function fetchResults() {
        try {
            const res = await fetch('/api/results');
            const data = await res.json();
            if (data.status === 'success') {
                allResults = data.results.ranked_df || [];
                mulesEl.textContent = data.results.total_mules;
                renderTable(allResults);
            }
        } catch (_) { /* first run, no data yet */ }
    }

    // ── Full Dashboard Update ───────────────────────────────────
    function updateFullDashboard(results) {
        const o = results.overall;

        // Metrics
        animateValue(f1ScoreEl, o.f1_score, 4);
        animateValue(precisionEl, o.precision, 4);
        animateValue(recallEl, o.recall, 4);

        // F1 Ring
        const pct = Math.round(o.f1_score * 100);
        f1RingFill.setAttribute('stroke-dasharray', `${pct}, 100`);

        // Mules
        mulesEl.textContent = o.tp;

        // Breakdown
        const ranked = results.ranked_df || [];
        allResults = ranked;
        const cycleOnly = ranked.filter(r => r.DetectionSource === 'CYCLE').length;
        const starOnly  = ranked.filter(r => r.DetectionSource === 'STAR').length;
        const both      = ranked.filter(r => r.DetectionSource === 'BOTH').length;
        const total     = ranked.length || 1;

        cycleCountEl.textContent = cycleOnly;
        starCountEl.textContent  = starOnly;
        bothCountEl.textContent  = both;
        muleBreakdown.textContent = `${cycleOnly} cycle · ${starOnly} star`;

        setTimeout(() => {
            cycleBar.style.width = `${(cycleOnly / total) * 100}%`;
            starBar.style.width  = `${(starOnly / total) * 100}%`;
            bothBar.style.width  = `${(both / total) * 100}%`;
        }, 100);

        // Confusion Matrix
        tpCell.textContent = o.tp;
        fpCell.textContent = o.fp;
        fnCell.textContent = o.fn;
        tnCell.textContent = o.tn;

        // Performance
        const totalTime = results.total_time;
        const p2 = results.total_time - (results.total_time * 0.15); // approximate
        const p3 = results.total_time * 0.15;

        // Use star/cycle metrics timing if available
        if (results.star_metrics && results.cycle_metrics) {
            // We don't have individual times from /api/run, but we can show total
        }

        gaugeLabel.textContent = totalTime.toFixed(3) + 's';
        drawGauge(totalTime, 6);

        if (totalTime < 2.0) {
            perfBadge.textContent = '✓ PASS';
            perfBadge.className = 'perf-badge pass';
        } else {
            perfBadge.textContent = '✗ FAIL';
            perfBadge.className = 'perf-badge fail';
        }

        if (results.phase2_time !== undefined && results.phase3_time !== undefined) {
            phase2Time.textContent = results.phase2_time.toFixed(3) + 's';
            phase3Time.textContent = results.phase3_time.toFixed(3) + 's';
        } else {
            phase2Time.textContent = '—';
            phase3Time.textContent = '—';
        }

        // Render table
        renderTable(allResults);
    }

    // ── Render Table ────────────────────────────────────────────
    function renderTable(data) {
        resultsTbody.innerHTML = '';

        if (!data || data.length === 0) {
            const emptyTr = document.createElement('tr');
            emptyTr.className = 'empty-row';
            const emptyTd = document.createElement('td');
            emptyTd.colSpan = 6;
            emptyTd.textContent = 'No results match your query.';
            emptyTr.appendChild(emptyTd);
            resultsTbody.appendChild(emptyTr);
            return;
        }

        data.forEach((row, i) => {
            const tr = document.createElement('tr');
            tr.style.animationDelay = `${i * 20}ms`;
            tr.classList.add('fade-in-row');

            const riskClass = row.RiskScore >= 1.0 ? 'risk-high' : 'risk-medium';
            const sourceClass = row.DetectionSource === 'CYCLE' ? 'source-cycle'
                              : row.DetectionSource === 'STAR'  ? 'source-star'
                              : 'source-both';
            const truthClass = row.GroundTruth === 'MULE' ? 'truth-mule' : 'truth-legit';

            // Build each cell safely with textContent (no innerHTML / XSS)
            const tdIndex = document.createElement('td');
            tdIndex.style.cssText = 'color: var(--text-muted); font-size: 0.8rem;';
            tdIndex.textContent = i + 1;

            const tdAccount = document.createElement('td');
            tdAccount.className = 'account-id';
            tdAccount.textContent = row.Account;

            const tdRisk = document.createElement('td');
            const riskSpan = document.createElement('span');
            riskSpan.className = `risk-badge ${riskClass}`;
            riskSpan.textContent = parseFloat(row.RiskScore).toFixed(1);
            tdRisk.appendChild(riskSpan);

            const tdSource = document.createElement('td');
            const sourceSpan = document.createElement('span');
            sourceSpan.className = `source-badge ${sourceClass}`;
            sourceSpan.textContent = row.DetectionSource;
            tdSource.appendChild(sourceSpan);

            const tdTruth = document.createElement('td');
            tdTruth.className = truthClass;
            tdTruth.textContent = row.GroundTruth;

            const tdPattern = document.createElement('td');
            tdPattern.style.fontSize = '0.82rem';
            tdPattern.textContent = row.PatternType;

            tr.append(tdIndex, tdAccount, tdRisk, tdSource, tdTruth, tdPattern);
            resultsTbody.appendChild(tr);
        });
    }

    // ── Loading State ───────────────────────────────────────────
    function setLoading(on) {
        runBtn.disabled = on;
        btnText.textContent = on ? 'Running...' : 'Run Detection';
        btnIcon.classList.toggle('hidden', on);
        btnSpinner.classList.toggle('hidden', !on);
        liveIndicator.classList.toggle('running', on);
        liveText.textContent = on ? 'PROCESSING' : 'COMPLETE';
    }

    // ── Animate Number ──────────────────────────────────────────
    function animateValue(el, target, decimals) {
        const duration = 800;
        const start = performance.now();
        const from = 0;

        function tick(now) {
            const elapsed = now - start;
            const progress = Math.min(elapsed / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
            const current = from + (target - from) * eased;
            el.textContent = current.toFixed(decimals);
            if (progress < 1) requestAnimationFrame(tick);
        }
        requestAnimationFrame(tick);
    }

    // ── Gauge Drawing ───────────────────────────────────────────
    function drawGauge(value, max) {
        const ctx = gaugeCanvas.getContext('2d');
        const w = gaugeCanvas.width;
        const h = gaugeCanvas.height;
        const cx = w / 2;
        const cy = h - 10;
        const radius = 80;

        ctx.clearRect(0, 0, w, h);

        // Background arc
        ctx.beginPath();
        ctx.arc(cx, cy, radius, Math.PI, 2 * Math.PI);
        ctx.strokeStyle = 'rgba(255,255,255,0.06)';
        ctx.lineWidth = 10;
        ctx.lineCap = 'round';
        ctx.stroke();

        // Value arc
        const pct = Math.min(value / max, 1);
        const endAngle = Math.PI + pct * Math.PI;

        const grad = ctx.createLinearGradient(cx - radius, cy, cx + radius, cy);
        if (value < 2.0) {
            grad.addColorStop(0, '#34d399');
            grad.addColorStop(1, '#22d3ee');
        } else {
            grad.addColorStop(0, '#fbbf24');
            grad.addColorStop(1, '#f87171');
        }

        ctx.beginPath();
        ctx.arc(cx, cy, radius, Math.PI, endAngle);
        ctx.strokeStyle = grad;
        ctx.lineWidth = 10;
        ctx.lineCap = 'round';
        ctx.stroke();

        // Threshold marker at 2.0s
        const thresholdAngle = Math.PI + (2.0 / max) * Math.PI;
        const mx = cx + radius * Math.cos(thresholdAngle);
        const my = cy + radius * Math.sin(thresholdAngle);
        ctx.beginPath();
        ctx.arc(mx, my, 3, 0, 2 * Math.PI);
        ctx.fillStyle = '#f87171';
        ctx.fill();
    }
});

/* ── Row fade-in animation (injected via JS) ─────────────────── */
const style = document.createElement('style');
style.textContent = `
    @keyframes fadeInRow {
        from { opacity: 0; transform: translateY(6px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .fade-in-row {
        animation: fadeInRow 0.3s ease forwards;
        opacity: 0;
    }
`;
document.head.appendChild(style);
