
let currentResult = null;

function quickDiagnose(code) {
    document.getElementById('stockInput').value = code;
    startDiagnosis();
}

function startDiagnosis() {
    const stockCode = document.getElementById('stockInput').value.trim();
    if (!stockCode || stockCode.length !== 6) {
        alert('请输入6位股票代码');
        return;
    }

    document.getElementById('loadingArea').style.display = 'flex';
    document.getElementById('resultArea').style.display = 'none';
    document.getElementById('streamArea').style.display = 'none';

    fetch('/api/agent/diagnose', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ stock_code: stockCode })
    })
    .then(r => r.json())
    .then(data => {
        document.getElementById('loadingArea').style.display = 'none';
        if (data.error) {
            alert('诊断失败: ' + data.error);
            return;
        }
        currentResult = data;
        renderFullResult(data);
    })
    .catch(err => {
        document.getElementById('loadingArea').style.display = 'none';
        alert('请求失败: ' + err.message);
    });
}

function renderFullResult(data) {
    document.getElementById('resultArea').style.display = 'block';

    const fd = data.final_decision || {};
    const sigMap = {
        '-1': { text: '卖出', class: 'mass-signal-sell', color: '#ef4444' },
        '0': { text: '观望', class: 'mass-signal-hold', color: '#eab308' },
        '1': { text: '买入', class: 'mass-signal-buy', color: '#22c55e' }
    };
    const sig = sigMap[String(fd.decision)] || sigMap['0'];

    // Banner
    document.getElementById('bannerStock').textContent = `${data.stock_name || data.stock_code} (${data.stock_code})`;
    document.getElementById('bannerPrice').textContent = `当前价: ¥${(data.current_price || 0).toFixed(2)} | 市场周期: ${data.market_cycle || '--'}`;
    document.getElementById('bannerSignal').innerHTML = `<span class="mass-signal ${sig.class}">${sig.text}</span>`;
    document.getElementById('bannerConfidence').textContent = `${((fd.confidence || 0) * 100).toFixed(0)}%`;
    document.getElementById('bannerPosition').textContent = `${((fd.position_pct || 0) * 100).toFixed(1)}%`;
    document.getElementById('bannerReturn').textContent = `${(fd.expected_return_pct || 0).toFixed(1)}%`;
    document.getElementById('bannerTime').textContent = `${data.processing_time_seconds || '--'}s`;
    document.getElementById('decisionBanner').style.borderLeftColor = sig.color;

    // K-line
    renderKlineChart('klineChart', data);

    // 3D Radar
    renderRadar3D('radar3DChart', data);

    // Agent Result Grid (2×3, merged view)
    renderAgentResultGrid(data);

    // Scenario Gauges
    renderScenarioGauges('scenarioGaugeChart', data);

    // Trade Plan + prices in banner already
    renderTradePlan('tradePlanContent', data);
    document.getElementById('planTarget').textContent = fd.target_price ? `¥${fd.target_price.toFixed(2)}` : '--';
    document.getElementById('planStop').textContent = fd.stop_loss ? `¥${fd.stop_loss.toFixed(2)}` : '--';
    document.getElementById('planHorizon').textContent = fd.time_horizon || '--';

    // Capital Flow
    renderCapitalFlow('capitalFlowChart', data);

    // Sentiment Cloud
    renderSentimentCloud('sentimentCloudChart', data);

    // Risk Matrix
    renderRiskMatrix('riskMatrixChart', data);

    // Chairman Reasoning
    document.getElementById('chairmanReasoning').innerHTML = formatReasoning(fd.reasoning || '');
    renderDissentingViews('dissentingViews', fd.dissenting_views || []);

    // Consensus & Risk
    renderTags('consensusFactors', fd.consensus_factors || [], 'mass-tag-success');
    const allRisks = [];
    Object.values(data.opinions || {}).forEach(op => {
        (op.risk_flags || []).forEach(f => { if (!allRisks.includes(f)) allRisks.push(f); });
    });
    renderTags('riskFlagsAll', allRisks, 'mass-tag-risk');

    // K-line status badges
    const ind = data.data_summary?.indicator_names || [];
    document.getElementById('klineMaStatus').textContent = ind.includes('ma_alignment') ? 'MA排列已计算' : 'MA: --';
    document.getElementById('klineMacdStatus').textContent = ind.includes('macd_golden_cross') ? 'MACD已计算' : 'MACD: --';
}

function formatReasoning(text) {
    return text.replace(/\n/g, '<br>').replace(/([0-9]+\.[0-9]+)/g, '<span style="color:var(--accent-cyan); font-family:var(--font-mono);">$1</span>');
}

function renderTags(containerId, items, className) {
    const el = document.getElementById(containerId);
    if (!items.length) {
        el.innerHTML = '<span style="color:var(--text-muted); font-size:12px;">无</span>';
        return;
    }
    el.innerHTML = items.map(f => `<span class="mass-tag ${className}">${f}</span>`).join('');
}

function renderDissentingViews(containerId, views) {
    const el = document.getElementById(containerId);
    if (!views || !views.length) {
        el.style.display = 'none';
        return;
    }
    el.style.display = 'block';
    el.innerHTML = '<div style="font-size:11px; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.5px; margin-bottom:8px;">异议记录</div>' +
        views.map(v => `
            <div style="padding:10px; background:var(--bg-surface); border-radius:var(--radius-sm); margin-bottom:8px; border-left:3px solid var(--accent-orange);">
                <div style="font-size:12px; color:var(--agent-sa); font-weight:600;">${v.agent}: ${v.view}</div>
                <div style="font-size:11px; color:var(--text-secondary); margin-top:4px;">→ ${v.chairman_response}</div>
            </div>
        `).join('');
}

function exportReport(format) {
    if (!currentResult) { alert('请先进行诊断'); return; }
    const fd = currentResult.final_decision || {};
    const ops = currentResult.opinions || {};
    const sigText = fd.decision === 1 ? '买入' : (fd.decision === -1 ? '卖出' : '观望');

    let md = `# MASS投研报告: ${currentResult.stock_name || currentResult.stock_code}\n\n`;
    md += `**生成时间**: ${currentResult.decision_date} ${currentResult.decision_time}\n\n`;
    md += `## 一、综合决策\n\n`;
    md += `- **决策**: ${sigText} (置信度: ${(fd.confidence*100).toFixed(1)}%)\n`;
    md += `- **建议仓位**: ${(fd.position_pct*100).toFixed(1)}%\n`;
    md += `- **目标价**: ${fd.target_price || '--'}\n`;
    md += `- **止损价**: ${fd.stop_loss || '--'}\n`;
    md += `- **预期收益**: ${fd.expected_return_pct || 0}%\n\n`;
    md += `## 二、各Agent观点\n\n`;
    Object.entries(ops).forEach(([id, op]) => {
        const s = op.signal === 1 ? '买入' : (op.signal === -1 ? '卖出' : '观望');
        md += `### ${id}\n- 信号: ${s} (置信度: ${(op.confidence*100).toFixed(0)}%)\n`;
        md += `- 理由: ${op.reasoning}\n- 关键因子: ${(op.key_factors || []).join(', ')}\n\n`;
    });
    md += `## 三、Chairman推理\n\n${fd.reasoning || ''}\n\n`;
    md += `---\n*免责声明: 本报告仅供参考，不构成投资建议。*`;

    if (format === 'markdown') {
        document.getElementById('reportContent').textContent = md;
        document.getElementById('reportModal').style.display = 'flex';
    } else {
        const element = document.createElement('div');
        element.innerHTML = `<pre style="background:#0a0e1a;color:#e8ecf4;padding:40px;font-family:sans-serif;line-height:1.8;white-space:pre-wrap;">${md.replace(/\n/g, '<br>')}</pre>`;
        html2pdf().set({ margin: 10, filename: `MASS_${currentResult.stock_code}_${currentResult.decision_date}.pdf` }).from(element).save();
    }
}

function closeReport() {
    document.getElementById('reportModal').style.display = 'none';
}

function saveToPortfolio() {
    if (!currentResult) { alert('请先进行诊断'); return; }
    const fd = currentResult.final_decision || {};
    fetch('/api/agent/positions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            stock_code: currentResult.stock_code,
            stock_name: currentResult.stock_name,
            entry_price: currentResult.current_price,
            shares: 100,
            position_pct: fd.position_pct || 0.1,
            target_price: fd.target_price,
            stop_loss: fd.stop_loss,
        })
    })
    .then(r => r.json())
    .then(d => {
        if (d.success) alert('已加入模拟持仓');
        else alert('保存失败: ' + (d.error || '未知错误'));
    });
}

// ========== Global Tooltip ==========
const massTooltipEl = document.createElement('div');
massTooltipEl.id = 'mass-tooltip';
massTooltipEl.innerHTML = '<div class="mass-tooltip-inner"><div class="mass-tooltip-arrow"></div></div>';
document.body.appendChild(massTooltipEl);

function showTooltip(targetEl, htmlContent) {
    const tooltip = document.getElementById('mass-tooltip');
    if (!tooltip) return;
    const inner = tooltip.querySelector('.mass-tooltip-inner');
    inner.innerHTML = '<div class="mass-tooltip-arrow"></div>' + htmlContent;

    const rect = targetEl.getBoundingClientRect();
    const tooltipRect = tooltip.getBoundingClientRect();

    // Position: above the element, centered horizontally
    let left = rect.left + rect.width / 2 - 160;
    let top = rect.top - 12;

    // Boundary checks
    if (left < 10) left = 10;
    if (left + 340 > window.innerWidth - 10) left = window.innerWidth - 350;
    if (top < 10) top = rect.bottom + 12; // flip to below if not enough space above

    tooltip.style.left = left + 'px';
    tooltip.style.top = top + 'px';
    tooltip.classList.add('visible');
}

function hideTooltip() {
    const tooltip = document.getElementById('mass-tooltip');
    if (tooltip) tooltip.classList.remove('visible');
}

// Enter key support
document.getElementById('stockInput')?.addEventListener('keypress', e => {
    if (e.key === 'Enter') startStreamDiagnosis();
});

// Auto-reconnect: resume active task on page load (cross-page persistence)
(function() {
    // Priority 1: Quick-diagnose/predict from history page
    const quickCode = sessionStorage.getItem('quickDiagnoseCode') || sessionStorage.getItem('quickPredictCode');
    // Priority 2: Active task reconnection
    const activeTask = (function() {
        try { const r = sessionStorage.getItem('mass_active_task'); return r ? JSON.parse(r) : null; } catch(e) { return null; }
    })();

    if (quickCode) {
        sessionStorage.removeItem('quickDiagnoseCode');
        sessionStorage.removeItem('quickPredictCode');
        const input = document.getElementById('stockInput');
        if (input) { input.value = quickCode; setTimeout(() => startStreamDiagnosis(), 600); }
    } else if (activeTask && activeTask.stock_code) {
        // Reconnect to ongoing task
        const input = document.getElementById('stockInput');
        if (input) { input.value = activeTask.stock_code; setTimeout(() => startStreamDiagnosis(activeTask.task_id), 600); }
    }
})();
