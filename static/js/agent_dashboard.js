/**
 * MASS Agent Dashboard JS
 * 多智能体投研结果可视化
 */

let radarChart = null;

function startDiagnosis() {
    const stockCode = document.getElementById('stockInput').value.trim();
    if (!stockCode || stockCode.length !== 6) {
        alert('请输入6位股票代码');
        return;
    }

    // 显示加载
    document.getElementById('loadingArea').style.display = 'block';
    document.getElementById('resultArea').style.display = 'none';

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
        renderResult(data);
    })
    .catch(err => {
        document.getElementById('loadingArea').style.display = 'none';
        alert('请求失败: ' + err.message);
    });
}

function renderResult(data) {
    document.getElementById('resultArea').style.display = 'block';

    // 头部信息
    document.getElementById('stockTitle').textContent = `${data.stock_name || data.stock_code} (${data.stock_code})`;
    document.getElementById('stockPrice').textContent = `当前价格: ¥${data.current_price?.toFixed(2) || '--'}`;
    document.getElementById('processingTime').textContent = `${data.processing_time_seconds || '--'}s`;

    // 最终决策
    const fd = data.final_decision || {};
    const signalMap = { '-1': { text: '卖出', class: 'signal-sell' }, '0': { text: '观望', class: 'signal-hold' }, '1': { text: '买入', class: 'signal-buy' } };
    const sig = signalMap[String(fd.decision)] || signalMap['0'];
    const sigEl = document.getElementById('finalSignal');
    sigEl.textContent = sig.text;
    sigEl.className = 'signal-badge ' + sig.class;

    document.getElementById('finalConfidence').textContent = (fd.confidence * 100).toFixed(1) + '%';
    document.getElementById('finalPosition').textContent = ((fd.position_pct || 0) * 100).toFixed(1) + '%';
    document.getElementById('finalReturn').textContent = (fd.expected_return_pct || 0).toFixed(1) + '%';

    // 雷达图
    renderRadarChart(data);

    // Agent卡片
    renderAgentCards(data);

    // 情景分析
    renderScenarios(fd.scenario_analysis || {});

    // 交易计划
    renderTradePlan(fd);

    // 推理链
    renderReasoning(fd);

    // 数据摘要
    renderDataSummary(data.data_summary || {});

    // 滚动到结果
    document.getElementById('resultArea').scrollIntoView({ behavior: 'smooth' });
}

function renderRadarChart(data) {
    const opinions = data.opinions || {};
    const dimensions = [
        { name: '技术面', key: 'TA-Agent', field: 'confidence' },
        { name: '基本面', key: 'FA-Agent', field: 'confidence' },
        { name: '资金面', key: 'CA-Agent', field: 'confidence' },
        { name: '情绪面', key: 'SA-Agent', field: 'confidence' },
        { name: '宏观匹配', key: 'MA-Agent', field: 'style_alignment' },
        { name: '风险可控', key: 'RA-Agent', field: 'confidence' },
    ];

    const indicatorData = dimensions.map(d => {
        const op = opinions[d.key];
        let value = 50;
        if (op && op.raw_data) {
            if (d.field === 'confidence') {
                value = (op.raw_data.confidence || 0.5) * 100;
            } else {
                value = (op.raw_data[d.field] || 0.5) * 100;
            }
            // 风险Agent反向：风险越高分数越低
            if (d.key === 'RA-Agent') {
                const rl = op.raw_data.risk_level || 3;
                value = (6 - rl) / 5 * 100;
            }
        }
        return { name: d.name, max: 100, value: Math.round(value) };
    });

    if (!radarChart) {
        radarChart = echarts.init(document.getElementById('radarChart'));
    }

    radarChart.setOption({
        color: ['#6c5ce7'],
        radar: {
            indicator: indicatorData.map(d => ({ name: d.name, max: d.max })),
            shape: 'polygon',
            splitNumber: 4,
            axisName: { color: '#a0a0a0' },
            splitLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } },
            splitArea: { show: false },
            axisLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } }
        },
        series: [{
            type: 'radar',
            data: [{
                value: indicatorData.map(d => d.value),
                name: '综合评分',
                areaStyle: { color: 'rgba(108, 92, 231, 0.3)' },
                lineStyle: { color: '#6c5ce7', width: 2 },
                itemStyle: { color: '#6c5ce7' }
            }]
        }]
    });
}

function renderAgentCards(data) {
    const container = document.getElementById('agentCards');
    const opinions = data.opinions || {};

    const agentMeta = {
        'TA-Agent': { name: '技术面分析师', icon: 'bi-graph-up-arrow', color: '#00b894' },
        'FA-Agent': { name: '基本面分析师', icon: 'bi-file-earmark-bar-graph', color: '#0984e3' },
        'CA-Agent': { name: '资金面分析师', icon: 'bi-cash-coin', color: '#6c5ce7' },
        'SA-Agent': { name: '情绪面分析师', icon: 'bi-megaphone', color: '#e17055' },
        'MA-Agent': { name: '宏观策略师', icon: 'bi-globe', color: '#fdcb6e' },
        'RA-Agent': { name: '风险控制官', icon: 'bi-shield-exclamation', color: '#d63031' },
    };

    let html = '';
    for (const [agentId, meta] of Object.entries(agentMeta)) {
        const op = opinions[agentId];
        if (!op) continue;

        const signalMap = { '-1': '卖出', '0': '观望', '1': '买入' };
        const signalClass = { '-1': 'text-danger', '0': 'text-warning', '1': 'text-success' };
        const signalText = signalMap[String(op.signal)] || '观望';
        const sClass = signalClass[String(op.signal)] || 'text-warning';

        const conf = Math.round((op.confidence || 0) * 100);
        const confColor = conf > 70 ? '#00b894' : (conf > 50 ? '#fdcb6e' : '#d63031');

        html += `
        <div class="agent-card mb-3 p-3" style="border-radius:10px; background:rgba(255,255,255,0.03);">
            <div class="d-flex justify-content-between align-items-center mb-2">
                <div>
                    <i class="bi ${meta.icon}" style="font-size:1.1rem; margin-right:8px; color:${meta.color};"></i>
                    <strong style="color:${meta.color}">${meta.name}</strong>
                    <span class="${sClass} ms-2" style="font-weight:bold;">${signalText}</span>
                </div>
                <div style="font-size:0.85rem; color:var(--text-secondary)">
                    置信度 ${conf}%
                </div>
            </div>
            <div class="confidence-bar mb-2">
                <div class="confidence-fill" style="width:${conf}%; background:${confColor};"></div>
            </div>
            <div style="font-size:0.85rem; color:var(--text-secondary); margin-bottom:6px;">
                ${op.reasoning?.substring(0, 80) || ''}...
            </div>
            <div>
                ${(op.key_factors || []).slice(0, 3).map(f => `<span class="factor-tag">${f}</span>`).join('')}
                ${(op.risk_flags || []).slice(0, 2).map(f => `<span class="factor-tag risk-tag"><i class="bi bi-exclamation-triangle" style="margin-right:3px;"></i>${f}</span>`).join('')}
            </div>
        </div>
        `;
    }
    container.innerHTML = html;
}

function renderScenarios(scenarios) {
    const container = document.getElementById('scenarioArea');
    const names = { bull: '乐观情景', base: '基准情景', bear: '悲观情景' };
    const classes = { bull: 'scenario-bull', base: 'scenario-base', bear: 'scenario-bear' };

    let html = '';
    for (const [key, scenario] of Object.entries(scenarios)) {
        const prob = Math.round((scenario.probability || 0) * 100);
        const ret = (scenario.return_pct || 0);
        const retColor = ret >= 0 ? '#00b894' : '#d63031';
        html += `
        <div class="col-md-4">
            <div class="scenario-box ${classes[key]}">
                <h6>${names[key]}</h6>
                <div style="font-size:1.5rem; font-weight:bold; color:${retColor}">${ret > 0 ? '+' : ''}${ret.toFixed(1)}%</div>
                <div style="font-size:0.85rem; color:var(--text-secondary)">概率 ${prob}%</div>
            </div>
        </div>
        `;
    }
    container.innerHTML = html;
}

function renderTradePlan(fd) {
    const container = document.getElementById('tradePlan');
    const plans = fd.execution_plan || [];

    let html = '<div class="timeline-item">';
    plans.forEach((plan, i) => {
        html += `
        <div class="timeline-item mb-3">
            <div class="timeline-dot"></div>
            <div style="font-size:0.9rem;">${plan}</div>
        </div>
        `;
    });
    html += '</div>';

    // 关键价位
    html += `
    <div class="row mt-3">
        <div class="col-4 text-center">
            <div style="font-size:0.8rem; color:var(--text-secondary)">目标价</div>
            <div style="font-size:1.2rem; color:#00b894; font-weight:bold">¥${(fd.target_price || 0).toFixed(2)}</div>
        </div>
        <div class="col-4 text-center">
            <div style="font-size:0.8rem; color:var(--text-secondary)">止损价</div>
            <div style="font-size:1.2rem; color:#d63031; font-weight:bold">¥${(fd.stop_loss || 0).toFixed(2)}</div>
        </div>
        <div class="col-4 text-center">
            <div style="font-size:0.8rem; color:var(--text-secondary)">时间周期</div>
            <div style="font-size:1.2rem; font-weight:bold">${fd.time_horizon || '--'}</div>
        </div>
    </div>
    `;

    container.innerHTML = html;

    // 共识因子
    const cf = document.getElementById('consensusFactors');
    const factors = fd.consensus_factors || [];
    if (factors.length > 0) {
        cf.innerHTML = '<h6 class="mb-2">共识因子</h6>' +
            factors.map(f => `<span class="factor-tag"><i class="bi bi-check-lg" style="margin-right:3px;"></i>${f}</span>`).join('');
    } else {
        cf.innerHTML = '';
    }
}

function renderReasoning(fd) {
    const container = document.getElementById('reasoningChain');

    let html = `
    <div class="mb-3" style="line-height:1.8;">
        <p>${fd.reasoning || '暂无详细推理'}</p>
    </div>
    `;

    // 异议记录
    const dissents = fd.dissenting_views || [];
    if (dissents.length > 0) {
        html += '<h6 class="mb-2">异议与回应</h6>';
        dissents.forEach(d => {
            html += `
            <div class="p-2 mb-2" style="border-radius:8px; background:rgba(214,48,49,0.1); border-left:3px solid #d63031;">
                <div style="font-size:0.85rem; color:#ff7675;"><strong>${d.agent}:</strong> ${d.view}</div>
                <div style="font-size:0.8rem; color:var(--text-secondary); margin-top:4px;">→ ${d.chairman_response}</div>
            </div>
            `;
        });
    }

    container.innerHTML = html;
}

function renderDataSummary(summary) {
    const container = document.getElementById('dataSummary');

    const sections = [
        { title: '技术指标', items: summary.indicator_names || [] },
        { title: '基本面', items: summary.fundamental_keys || [] },
        { title: '资金流向', items: summary.fund_flow_keys || [] },
        { title: '情绪', items: summary.sentiment_keys || [] },
        { title: '宏观', items: summary.macro_keys || [] },
        { title: '风险', items: summary.risk_keys || [] },
    ];

    let html = '';
    sections.forEach(sec => {
        html += `
        <div class="col-md-2 col-4 mb-2">
            <div style="font-size:0.75rem; color:var(--text-secondary); margin-bottom:4px;">${sec.title}</div>
            <div style="font-size:0.8rem;">
                ${sec.items.slice(0, 5).map(i => `<span class="factor-tag" style="font-size:0.7rem; padding:2px 6px;">${i}</span>`).join('')}
            </div>
        </div>
        `;
    });
    container.innerHTML = html;
}

// 回车搜索
document.getElementById('stockInput')?.addEventListener('keypress', function(e) {
    if (e.key === 'Enter') startDiagnosis();
});

// 窗口resize时重绘图表
window.addEventListener('resize', function() {
    if (radarChart) radarChart.resize();
});
