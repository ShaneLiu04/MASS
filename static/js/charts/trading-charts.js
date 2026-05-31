/**
 * MASS Trading Charts Engine v2.0
 * K线 / 3D雷达 / 桑基图 / 词云 / 仪表盘 / 散点图 / 时间轴
 */

// ========== K-Line Chart ==========
function renderKlineChart(domId, data) {
    const chart = echarts.init(document.getElementById(domId));
    const rawKline = data.data_summary?.kline_preview || generateMockKline(data.current_price || 15);
    const dates = rawKline.map(d => d.date);
    const values = rawKline.map(d => [d.open, d.close, d.low, d.high]);
    const volumes = rawKline.map(d => d.volume);

    // Calculate MA
    function calcMA(dayCount, data) {
        const result = [];
        for (let i = 0; i < data.length; i++) {
            if (i < dayCount - 1) { result.push('-'); continue; }
            let sum = 0;
            for (let j = 0; j < dayCount; j++) sum += data[i - j][1];
            result.push((sum / dayCount).toFixed(2));
        }
        return result;
    }

    const ma5 = calcMA(5, values);
    const ma20 = calcMA(20, values);
    const ma60 = calcMA(60, values);

    const option = {
        backgroundColor: 'transparent',
        animation: true,
        tooltip: {
            trigger: 'axis',
            axisPointer: { type: 'cross' },
            backgroundColor: 'rgba(22, 31, 58, 0.95)',
            borderColor: 'rgba(100,130,200,0.2)',
            textStyle: { color: '#e8ecf4', fontSize: 11 },
            formatter: function (params) {
                let res = `<div style="font-weight:bold;margin-bottom:4px;">${params[0].axisValue}</div>`;
                params.forEach(p => {
                    const color = p.color;
                    res += `<div style="display:flex;align-items:center;gap:6px;">
                        <span style="width:8px;height:8px;background:${color};border-radius:50%;"></span>
                        <span>${p.seriesName}: <b>${p.value}</b></span>
                    </div>`;
                });
                return res;
            }
        },
        grid: [
            { left: '8%', right: '6%', top: '10%', height: '55%' },
            { left: '8%', right: '6%', top: '70%', height: '20%' }
        ],
        xAxis: [
            {
                type: 'category', data: dates, scale: true,
                boundaryGap: false, axisLine: { lineStyle: { color: 'rgba(100,130,200,0.2)' } },
                axisLabel: { color: '#4a5a78', fontSize: 10 },
                splitLine: { show: false }
            },
            {
                type: 'category', data: dates, gridIndex: 1, scale: true,
                boundaryGap: false, axisLine: { show: false },
                axisLabel: { show: false }, splitLine: { show: false }
            }
        ],
        yAxis: [
            {
                scale: true,
                axisLine: { lineStyle: { color: 'rgba(100,130,200,0.2)' } },
                axisLabel: { color: '#4a5a78', fontSize: 10, fontFamily: 'monospace' },
                splitLine: { lineStyle: { color: 'rgba(100,130,200,0.06)' } }
            },
            {
                scale: true, gridIndex: 1, splitNumber: 2,
                axisLine: { show: false }, axisLabel: { show: false },
                axisTick: { show: false }, splitLine: { show: false }
            }
        ],
        dataZoom: [
            { type: 'inside', xAxisIndex: [0, 1], start: 50, end: 100 },
            { show: true, xAxisIndex: [0, 1], type: 'slider', bottom: '2%', start: 50, end: 100,
              height: 16, borderColor: 'transparent', backgroundColor: 'rgba(100,130,200,0.05)',
              fillerColor: 'rgba(59,130,246,0.15)', handleStyle: { color: '#3b82f6' },
              textStyle: { color: '#4a5a78', fontSize: 10 } }
        ],
        series: [
            {
                name: 'K线', type: 'candlestick', data: values,
                itemStyle: {
                    color: '#ef4444', color0: '#22c55e',
                    borderColor: '#ef4444', borderColor0: '#22c55e'
                }
            },
            { name: 'MA5', type: 'line', data: ma5, smooth: true, showSymbol: false,
              lineStyle: { color: '#eab308', width: 1 } },
            { name: 'MA20', type: 'line', data: ma20, smooth: true, showSymbol: false,
              lineStyle: { color: '#a855f7', width: 1 } },
            { name: 'MA60', type: 'line', data: ma60, smooth: true, showSymbol: false,
              lineStyle: { color: '#06b6d4', width: 1 } },
            {
                name: '成交量', type: 'bar', xAxisIndex: 1, yAxisIndex: 1, data: volumes,
                itemStyle: {
                    color: function(p) {
                        const v = values[p.dataIndex];
                        return v[1] >= v[0] ? 'rgba(34,197,94,0.5)' : 'rgba(239,68,68,0.5)';
                    }
                }
            }
        ]
    };
    chart.setOption(option);
    window.addEventListener('resize', () => chart.resize());
}

function generateMockKline(basePrice) {
    const kline = [];
    let price = basePrice;
    const now = new Date();
    for (let i = 59; i >= 0; i--) {
        const d = new Date(now);
        d.setDate(d.getDate() - i);
        const change = (Math.random() - 0.48) * 0.04;
        const open = price;
        const close = price * (1 + change);
        const high = Math.max(open, close) * (1 + Math.random() * 0.02);
        const low = Math.min(open, close) * (1 - Math.random() * 0.02);
        price = close;
        kline.push({
            date: d.toISOString().split('T')[0],
            open: parseFloat(open.toFixed(2)),
            close: parseFloat(close.toFixed(2)),
            high: parseFloat(high.toFixed(2)),
            low: parseFloat(low.toFixed(2)),
            volume: Math.floor(Math.random() * 5000000 + 1000000)
        });
    }
    return kline;
}

// ========== 3D Radar Chart ==========
function renderRadar3D(domId, data) {
    const chart = echarts.init(document.getElementById(domId));
    const opinions = data.opinions || {};
    const dims = [
        { name: '技术面', key: 'TA-Agent', color: '#06b6d4' },
        { name: '基本面', key: 'FA-Agent', color: '#3b82f6' },
        { name: '资金面', key: 'CA-Agent', color: '#a855f7' },
        { name: '情绪面', key: 'SA-Agent', color: '#f97316' },
        { name: '宏观匹配', key: 'MA-Agent', color: '#eab308' },
        { name: '风险可控', key: 'RA-Agent', color: '#22c55e' }
    ];

    const values = dims.map(d => {
        const op = opinions[d.key];
        let v = 50;
        if (op && op.raw_data) {
            if (d.key === 'RA-Agent') {
                const rl = op.raw_data.risk_level || 3;
                v = (6 - rl) / 5 * 100;
            } else {
                v = (op.raw_data.confidence || 0.5) * 100;
            }
        }
        return { name: d.name, value: Math.round(v), itemStyle: { color: d.color } };
    });

    const option = {
        backgroundColor: 'transparent',
        tooltip: {
            trigger: 'item',
            backgroundColor: 'rgba(22, 31, 58, 0.95)',
            borderColor: 'rgba(100,130,200,0.2)',
            textStyle: { color: '#e8ecf4' }
        },
        radar: {
            indicator: values.map(v => ({ name: v.name, max: 100 })),
            shape: 'polygon',
            splitNumber: 4,
            axisName: { color: '#8b9bb4', fontSize: 11 },
            splitLine: { lineStyle: { color: 'rgba(100,130,200,0.1)' } },
            splitArea: { areaStyle: { color: ['rgba(59,130,246,0.02)', 'rgba(59,130,246,0.04)'] } },
            axisLine: { lineStyle: { color: 'rgba(100,130,200,0.1)' } }
        },
        series: [{
            type: 'radar',
            data: [{
                value: values.map(v => v.value),
                name: '综合评分',
                symbol: 'circle',
                symbolSize: 6,
                lineStyle: { color: '#3b82f6', width: 2 },
                areaStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                    { offset: 0, color: 'rgba(59,130,246,0.3)' },
                    { offset: 1, color: 'rgba(59,130,246,0.05)' }
                ]) },
                itemStyle: { color: '#3b82f6', borderWidth: 2, borderColor: '#0a0e1a' }
            }]
        }]
    };
    chart.setOption(option);
    window.addEventListener('resize', () => chart.resize());
}

// ========== Agent Mini Cards ==========
// ========== Agent Result Grid — 6 cards in 2×3 layout ==========
function renderAgentResultGrid(data) {
    const el = document.getElementById('agentResultGrid');
    if (!el) return;
    const opinions = data.opinions || {};
    const sigMap = { '-1': '卖出', '0': '观望', '1': '买入' };
    const sigColorMap = { '-1': '#ef4444', '0': '#eab308', '1': '#22c55e' };

    const agents = [
        { id: 'TA-Agent', name: '技术面', role: 'K线形态 · 均线系统 · MACD/RSI/KDJ', icon: 'graph-up-arrow', color: '#06b6d4' },
        { id: 'FA-Agent', name: '基本面', role: '财务健康度 · 估值水平 · 盈利能力', icon: 'bar-chart-line', color: '#3b82f6' },
        { id: 'CA-Agent', name: '资金面', role: '主力资金 · 北向资金 · 筹码分布', icon: 'cash-stack', color: '#a855f7' },
        { id: 'SA-Agent', name: '情绪面', role: '市场情绪 · 舆情热度 · crowd行为', icon: 'chat-square-text', color: '#f97316' },
        { id: 'MA-Agent', name: '宏观', role: '经济周期 · 行业景气 · 政策环境', icon: 'bank', color: '#eab308' },
        { id: 'RA-Agent', name: '风险', role: '波动率 · 最大回撤 · 仓位管理', icon: 'shield-exclamation', color: '#ef4444' }
    ];

    el.innerHTML = agents.map(a => {
        const op = opinions[a.id];
        if (!op) return '';
        const sigText = sigMap[String(op.signal)] || '观望';
        const sigColor = sigColorMap[String(op.signal)] || '#eab308';
        const conf = Math.round((op.confidence || 0) * 100);
        const raw = op.raw_data || {};

        // Key-value indicator pairs
        const kv = _getAgentKV(a.id, raw);
        const kvHtml = kv.length ? `<div style="display:flex; flex-wrap:wrap; gap:4px; margin:6px 0;">${kv.map(([k,v]) => `<span class="result-kv"><b>${k}</b> ${escapeHtml(v)}</span>`).join('')}</div>` : '';

        const factors = (op.key_factors || []).map(f => `<span class="mass-tag mass-tag-factor">${escapeHtml(f)}</span>`).join('');
        const risks = (op.risk_flags || []).slice(0, 3).map(f => `<span class="mass-tag mass-tag-risk">${escapeHtml(f)}</span>`).join('');

        return `
        <div class="result-agent-card" style="background:var(--bg-surface); border-radius:var(--radius-md); border:1px solid var(--border-subtle); border-top:3px solid ${a.color}; padding:11px 13px; display:flex; flex-direction:column;">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
                <span style="width:28px;height:28px;border-radius:var(--radius-sm);background:${a.color}15;color:${a.color};display:flex;align-items:center;justify-content:center;font-size:13px;"><i class="bi bi-${a.icon}"></i></span>
                <span style="font-weight:600;color:${a.color};font-size:13px;">${a.name}</span>
                <span style="font-size:10px;color:var(--text-muted);flex:1;">${a.role}</span>
                <span style="font-weight:700;font-size:15px;color:${sigColor};">${sigText}</span>
                <span style="font-size:11px;color:var(--text-muted);">${conf}%</span>
            </div>
            <div class="mass-progress" style="height:3px;margin-bottom:6px;"><div class="mass-progress-fill" style="width:${conf}%;background:${a.color};"></div></div>
            ${kvHtml}
            <div style="font-size:11px;color:var(--text-secondary);line-height:1.5;flex:1;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden;">${escapeHtml(op.reasoning || '')}</div>
            <div style="display:flex;gap:4px;flex-wrap:wrap;margin-top:6px;">${factors}${risks}</div>
        </div>`;
    }).join('');

    // Update consensus bar
    let buy = 0, hold = 0, sell = 0;
    Object.values(opinions).forEach(o => { if (o.signal === 1) buy++; else if (o.signal === -1) sell++; else hold++; });
    const t = buy + hold + sell;
    if (t > 0) {
        ['rConsensusBuy','rConsensusHold','rConsensusSell'].forEach((id,i) => { const el2 = document.getElementById(id); if (el2) el2.style.width = ([buy,hold,sell][i]/t*100).toFixed(1)+'%'; });
        ['rConsensusBuyNum','rConsensusHoldNum','rConsensusSellNum'].forEach((id,i) => { const el2 = document.getElementById(id); if (el2) el2.textContent = [buy,hold,sell][i]; });
    }
}

function _getAgentKV(agentId, raw) {
    switch (agentId) {
        case 'TA-Agent': return [
            ['形态', (raw.chart_patterns||[]).slice(0,2).join(', ')||'--'],
            ['趋势', raw.trend_direction||'--'],
            ['目标', raw.target_price_low&&raw.target_price_high?`¥${raw.target_price_low}-${raw.target_price_high}`:'--']
        ];
        case 'FA-Agent': return [
            ['评分', raw.fundamental_score!=null?raw.fundamental_score+'/100':'--'],
            ['估值', raw.valuation_gap||'--']
        ];
        case 'CA-Agent': return [
            ['资金分', raw.capital_score!=null?raw.capital_score:'--'],
            ['主力', raw.smart_money_direction||'--']
        ];
        case 'SA-Agent': return [
            ['情绪', raw.sentiment_index!=null?raw.sentiment_index.toFixed(2):'--'],
            ['crowd', raw.crowd_behavior||'--']
        ];
        case 'MA-Agent': return [
            ['周期', raw.market_cycle||'--'],
            ['行业', raw.sector_outlook||'--']
        ];
        case 'RA-Agent': return [
            ['风险', raw.risk_level!=null?raw.risk_level+'/5':'--'],
            ['仓位', raw.max_position_pct!=null?'≤'+(raw.max_position_pct*100).toFixed(0)+'%':'--']
        ];
        default: return [];
    }
}

// ========== Decision Timeline ==========
function renderTimeline(domId, data) {
    const el = document.getElementById(domId);
    const ops = data.opinions || {};
    const meta = [
        { id: 'TA-Agent', name: '技术面分析', icon: '<i class="bi bi-graph-up-arrow" style="font-size:12px;"></i>', color: '#06b6d4' },
        { id: 'FA-Agent', name: '基本面分析', icon: '<i class="bi bi-bar-chart-line" style="font-size:12px;"></i>', color: '#3b82f6' },
        { id: 'CA-Agent', name: '资金面分析', icon: '<i class="bi bi-cash-stack" style="font-size:12px;"></i>', color: '#a855f7' },
        { id: 'SA-Agent', name: '情绪面分析', icon: '<i class="bi bi-chat-square-text" style="font-size:12px;"></i>', color: '#f97316' },
        { id: 'MA-Agent', name: '宏观策略', icon: '<i class="bi bi-bank" style="font-size:12px;"></i>', color: '#eab308' },
        { id: 'RA-Agent', name: '风险评估', icon: '<i class="bi bi-shield-exclamation" style="font-size:12px;"></i>', color: '#ef4444' },
        { id: 'Chairman', name: '综合决策', icon: '<i class="bi bi-award" style="font-size:12px;"></i>', color: '#e8ecf4' }
    ];

    el.innerHTML = '<div class="mass-timeline">' + meta.map((m, i) => {
        const op = ops[m.id];
        const isDone = !!op || m.id === 'Chairman';
        const isLast = i === meta.length - 1;
        const sigMap = { '-1': '卖出', '0': '观望', '1': '买入' };

        let detail = '';
        if (op) {
            detail = `<div style="font-size:11px; color:var(--text-muted); margin-top:4px;">
                ${sigMap[String(op.signal)] || '观望'} | 置信度 ${Math.round(op.confidence * 100)}%
            </div>`;
        } else if (m.id === 'Chairman' && data.final_decision) {
            const fd = data.final_decision;
            detail = `<div style="font-size:11px; color:var(--text-muted); margin-top:4px;">
                ${sigMap[String(fd.decision)] || '观望'} | 置信度 ${Math.round(fd.confidence * 100)}%
            </div>`;
        }

        return `
        <div class="mass-timeline-item">
            <div class="mass-timeline-dot ${isDone ? 'completed' : ''}" style="border-color:${m.color};"></div>
            <div style="font-size:12px; font-weight:600; color:${isDone ? m.color : 'var(--text-muted)'};">
                <span style="margin-right:4px;">${m.icon}</span> ${m.name}
            </div>
            ${detail}
        </div>`;
    }).join('') + '</div>';
}

// ========== Scenario Gauges ==========
function renderScenarioGauges(domId, data) {
    const chart = echarts.init(document.getElementById(domId));
    const scenarios = (data.final_decision?.scenario_analysis) || {};
    const bull = scenarios.bull || { probability: 0.25, return_pct: 20 };
    const base = scenarios.base || { probability: 0.5, return_pct: 10 };
    const bear = scenarios.bear || { probability: 0.25, return_pct: -8 };

    const option = {
        backgroundColor: 'transparent',
        tooltip: {
            backgroundColor: 'rgba(22, 31, 58, 0.95)',
            borderColor: 'rgba(100,130,200,0.2)',
            textStyle: { color: '#e8ecf4' }
        },
        series: [
            {
                type: 'gauge', center: ['20%', '55%'], radius: '70%',
                startAngle: 200, endAngle: -20,
                min: -30, max: 50,
                splitNumber: 8,
                axisLine: { lineStyle: { width: 8, color: [[0.4, '#ef4444'], [0.6, '#eab308'], [1, '#22c55e']] } },
                pointer: { itemStyle: { color: '#22c55e' }, width: 4 },
                axisTick: { distance: -12, length: 4, lineStyle: { color: 'rgba(100,130,200,0.3)' } },
                splitLine: { distance: -16, length: 8, lineStyle: { color: 'rgba(100,130,200,0.3)' } },
                axisLabel: { distance: -28, color: '#4a5a78', fontSize: 9 },
                detail: { valueAnimation: true, formatter: '{value}%', color: '#22c55e', fontSize: 18, fontFamily: 'monospace', offsetCenter: [0, '60%'] },
                title: { offsetCenter: [0, '85%'], fontSize: 11, color: '#8b9bb4' },
                data: [{ value: bull.return_pct, name: `乐观 ${Math.round(bull.probability * 100)}%` }]
            },
            {
                type: 'gauge', center: ['50%', '55%'], radius: '70%',
                startAngle: 200, endAngle: -20,
                min: -30, max: 50,
                splitNumber: 8,
                axisLine: { lineStyle: { width: 8, color: [[0.4, '#ef4444'], [0.6, '#eab308'], [1, '#22c55e']] } },
                pointer: { itemStyle: { color: '#eab308' }, width: 4 },
                axisTick: { distance: -12, length: 4, lineStyle: { color: 'rgba(100,130,200,0.3)' } },
                splitLine: { distance: -16, length: 8, lineStyle: { color: 'rgba(100,130,200,0.3)' } },
                axisLabel: { distance: -28, color: '#4a5a78', fontSize: 9 },
                detail: { valueAnimation: true, formatter: '{value}%', color: '#eab308', fontSize: 18, fontFamily: 'monospace', offsetCenter: [0, '60%'] },
                title: { offsetCenter: [0, '85%'], fontSize: 11, color: '#8b9bb4' },
                data: [{ value: base.return_pct, name: `基准 ${Math.round(base.probability * 100)}%` }]
            },
            {
                type: 'gauge', center: ['80%', '55%'], radius: '70%',
                startAngle: 200, endAngle: -20,
                min: -30, max: 50,
                splitNumber: 8,
                axisLine: { lineStyle: { width: 8, color: [[0.4, '#ef4444'], [0.6, '#eab308'], [1, '#22c55e']] } },
                pointer: { itemStyle: { color: '#ef4444' }, width: 4 },
                axisTick: { distance: -12, length: 4, lineStyle: { color: 'rgba(100,130,200,0.3)' } },
                splitLine: { distance: -16, length: 8, lineStyle: { color: 'rgba(100,130,200,0.3)' } },
                axisLabel: { distance: -28, color: '#4a5a78', fontSize: 9 },
                detail: { valueAnimation: true, formatter: '{value}%', color: '#ef4444', fontSize: 18, fontFamily: 'monospace', offsetCenter: [0, '60%'] },
                title: { offsetCenter: [0, '85%'], fontSize: 11, color: '#8b9bb4' },
                data: [{ value: bear.return_pct, name: `悲观 ${Math.round(bear.probability * 100)}%` }]
            }
        ]
    };
    chart.setOption(option);
    window.addEventListener('resize', () => chart.resize());
}

// ========== Trade Plan ==========
function renderTradePlan(domId, data) {
    const el = document.getElementById(domId);
    const plans = (data.final_decision?.execution_plan) || [];
    if (!plans.length) {
        el.innerHTML = '<span style="color:var(--text-muted);">暂无具体交易计划</span>';
        return;
    }
    el.innerHTML = plans.map((plan, i) => `
        <div style="display:flex; gap:10px; align-items:flex-start; padding:8px 0; ${i < plans.length - 1 ? 'border-bottom:1px solid var(--border-subtle);' : ''}">
            <div style="width:20px; height:20px; border-radius:50%; background:linear-gradient(135deg, var(--accent-blue), var(--accent-cyan)); color:white; display:flex; align-items:center; justify-content:center; font-size:10px; font-weight:700; flex-shrink:0;">${i + 1}</div>
            <div style="font-size:13px; color:var(--text-secondary); line-height:1.6;">${plan}</div>
        </div>
    `).join('');
}

// ========== Capital Flow Sankey ==========
function renderCapitalFlow(domId, data) {
    const chart = echarts.init(document.getElementById(domId));
    const ff = data.opinions?.['CA-Agent']?.raw_data || {};
    const mainFlow = ff.main_net_inflow_10d || 0;
    const north = ff.north_bound_30d || 0;

    const option = {
        backgroundColor: 'transparent',
        tooltip: { trigger: 'item', triggerOn: 'mousemove',
            backgroundColor: 'rgba(22, 31, 58, 0.95)', borderColor: 'rgba(100,130,200,0.2)', textStyle: { color: '#e8ecf4' } },
        series: [{
            type: 'sankey', layout: 'none', emphasis: { focus: 'adjacency' },
            nodeAlign: 'left',
            data: [
                { name: '主力资金', itemStyle: { color: '#a855f7' } },
                { name: '北向资金', itemStyle: { color: '#3b82f6' } },
                { name: '散户资金', itemStyle: { color: '#f97316' } },
                { name: '融资资金', itemStyle: { color: '#eab308' } },
                { name: '买入', itemStyle: { color: '#22c55e' } },
                { name: '卖出', itemStyle: { color: '#ef4444' } },
                { name: '观望', itemStyle: { color: '#8b9bb4' } }
            ],
            links: [
                { source: '主力资金', target: mainFlow > 0 ? '买入' : '卖出', value: Math.abs(mainFlow) || 3000 },
                { source: '北向资金', target: north > 0 ? '买入' : '卖出', value: Math.abs(north) * 1000 || 1500 },
                { source: '散户资金', target: '卖出', value: 2000 },
                { source: '融资资金', target: '买入', value: 800 },
                { source: '主力资金', target: '观望', value: 500 },
            ],
            lineStyle: { color: 'gradient', curveness: 0.5, opacity: 0.4 },
            label: { color: '#8b9bb4', fontSize: 11 }
        }]
    };
    chart.setOption(option);
    window.addEventListener('resize', () => chart.resize());
}

// ========== Sentiment Word Cloud ==========
function renderSentimentCloud(domId, data) {
    const chart = echarts.init(document.getElementById(domId));
    const words = [
        { name: '业绩预增', value: 90 }, { name: '机构增持', value: 80 },
        { name: '政策支持', value: 75 }, { name: '技术突破', value: 70 },
        { name: '订单饱满', value: 65 }, { name: '估值修复', value: 60 },
        { name: '板块轮动', value: 55 }, { name: '情绪回暖', value: 50 },
        { name: '北向流入', value: 45 }, { name: '主力吸筹', value: 40 },
        { name: '短期震荡', value: 35 }, { name: '套牢盘', value: 30 },
        { name: '解禁压力', value: 25 }, { name: '业绩不及预期', value: 20 }
    ];

    const option = {
        backgroundColor: 'transparent',
        tooltip: { show: true,
            backgroundColor: 'rgba(22, 31, 58, 0.95)', borderColor: 'rgba(100,130,200,0.2)', textStyle: { color: '#e8ecf4' } },
        series: [{
            type: 'wordCloud', shape: 'circle',
            left: 'center', top: 'center', width: '90%', height: '90%',
            right: null, bottom: null,
            sizeRange: [10, 28],
            rotationRange: [-30, 30],
            rotationStep: 15,
            gridSize: 8,
            drawOutOfBound: false,
            textStyle: {
                fontFamily: 'sans-serif',
                fontWeight: 'bold',
                color: function() {
                    const colors = ['#3b82f6', '#06b6d4', '#22c55e', '#eab308', '#f97316', '#a855f7', '#8b9bb4'];
                    return colors[Math.floor(Math.random() * colors.length)];
                }
            },
            emphasis: { focus: 'self', textStyle: { shadowBlur: 10, shadowColor: 'rgba(59,130,246,0.5)' } },
            data: words
        }]
    };
    chart.setOption(option);
    window.addEventListener('resize', () => chart.resize());
}

// ========== Risk Matrix Scatter ==========
function renderRiskMatrix(domId, data) {
    const chart = echarts.init(document.getElementById(domId));
    const fd = data.final_decision || {};
    const ra = data.opinions?.['RA-Agent']?.raw_data || {};
    const riskLevel = ra.risk_level || 3;
    const expReturn = fd.expected_return_pct || 0;

    const option = {
        backgroundColor: 'transparent',
        tooltip: {
            backgroundColor: 'rgba(22, 31, 58, 0.95)', borderColor: 'rgba(100,130,200,0.2)', textStyle: { color: '#e8ecf4' }
        },
        grid: { left: '15%', right: '8%', top: '10%', bottom: '15%' },
        xAxis: {
            name: '风险等级', nameLocation: 'middle', nameGap: 25,
            type: 'value', min: 0.5, max: 5.5, interval: 1,
            axisLine: { lineStyle: { color: 'rgba(100,130,200,0.2)' } },
            axisLabel: { color: '#4a5a78', fontSize: 10, formatter: v => v <= 5 ? v.toFixed(0) : '' },
            splitLine: { lineStyle: { color: 'rgba(100,130,200,0.06)' } }
        },
        yAxis: {
            name: '预期收益 %', nameLocation: 'middle', nameGap: 35,
            type: 'value',
            axisLine: { lineStyle: { color: 'rgba(100,130,200,0.2)' } },
            axisLabel: { color: '#4a5a78', fontSize: 10, fontFamily: 'monospace' },
            splitLine: { lineStyle: { color: 'rgba(100,130,200,0.06)' } }
        },
        series: [
            {
                type: 'scatter',
                symbolSize: 40,
                data: [[riskLevel, expReturn]],
                itemStyle: {
                    color: expReturn > 0 ? 'rgba(34,197,94,0.6)' : 'rgba(239,68,68,0.6)',
                    borderColor: expReturn > 0 ? '#22c55e' : '#ef4444',
                    borderWidth: 2,
                    shadowBlur: 10,
                    shadowColor: expReturn > 0 ? 'rgba(34,197,94,0.3)' : 'rgba(239,68,68,0.3)'
                },
                label: { show: true, formatter: '当前', color: '#e8ecf4', fontSize: 10, fontWeight: 'bold' }
            },
            {
                type: 'scatter',
                symbolSize: 12,
                data: [[1, 15], [1, 8], [2, 12], [2, 5], [3, 10], [3, 3], [4, 8], [4, -2], [5, 5], [5, -8]],
                itemStyle: { color: 'rgba(100,130,200,0.15)', borderColor: 'rgba(100,130,200,0.3)', borderWidth: 1 },
                silent: true
            }
        ],
        graphic: [
            { type: 'rect', left: '15%', bottom: '15%', shape: { width: 200, height: 120 }, style: { fill: 'rgba(34,197,94,0.03)' } },
            { type: 'rect', left: '15%', top: '10%', shape: { width: 200, height: 100 }, style: { fill: 'rgba(239,68,68,0.03)' } }
        ]
    };
    chart.setOption(option);
    window.addEventListener('resize', () => chart.resize());
}

// ========== Stream Diagnosis (Real SSE) ==========
let streamAbortController = null;
let _activeTaskId = null;
let _streamRetryCount = 0;
const STREAM_MAX_RETRY = 5;
const STREAM_BASE_DELAY_MS = 1000;
let _streamReconnectTimer = null;

const _AGENT_META = {
    'TA-Agent': { color: '#06b6d4', waitText: '等待技术面分析师完成K线形态与指标研判...' },
    'FA-Agent': { color: '#3b82f6', waitText: '等待基本面分析师完成财务数据评估...' },
    'CA-Agent': { color: '#a855f7', waitText: '等待资金面分析师完成资金流向追踪...' },
    'SA-Agent': { color: '#f97316', waitText: '等待情绪面分析师完成舆情与情绪评估...' },
    'MA-Agent': { color: '#eab308', waitText: '等待宏观策略师完成经济周期研判...' },
    'RA-Agent': { color: '#ef4444', waitText: '等待风险控制官完成风险等级评估...' },
};

function _showReconnectingOverlay(text) {
    const overlay = document.getElementById('streamReconnectOverlay');
    const txtEl = document.getElementById('streamReconnectText');
    if (overlay) overlay.style.display = 'flex';
    if (txtEl && text) txtEl.textContent = text;
}

function _hideReconnectingOverlay() {
    const overlay = document.getElementById('streamReconnectOverlay');
    if (overlay) overlay.style.display = 'none';
}

function _clearStreamRetryState() {
    _streamRetryCount = 0;
    if (_streamReconnectTimer) {
        clearTimeout(_streamReconnectTimer);
        _streamReconnectTimer = null;
    }
}

function startStreamDiagnosis(taskId) {
    const stockCode = document.getElementById('stockInput').value.trim();
    if (!stockCode || stockCode.length !== 6) {
        alert('请输入6位股票代码');
        return;
    }

    // 取消之前的请求和重连定时器
    if (streamAbortController) { streamAbortController.abort(); }
    if (_streamReconnectTimer) { clearTimeout(_streamReconnectTimer); _streamReconnectTimer = null; }
    streamAbortController = new AbortController();

    document.getElementById('streamArea').style.display = 'block';
    document.getElementById('resultArea').style.display = 'none';
    document.getElementById('loadingArea').style.display = 'none';
    document.getElementById('streamStatus').textContent = '连接中...';
    document.getElementById('streamAgentCount').textContent = '0/6';

    // 重置进度条
    document.querySelectorAll('.stream-stage').forEach(el => el.className = 'stream-stage');
    document.querySelectorAll('.stage-connector').forEach(el => el.className = 'stage-connector');
    document.getElementById('stage-init').classList.add('active');

    // 重置共识条
    document.getElementById('streamConsensusMini').style.display = 'none';
    ['cMiniBuy','cMiniHold','cMiniSell'].forEach(id => document.getElementById(id).style.width = '0%');

    // 重置全部 6 张 Agent 卡片
    _streamAgentsCompleted = 0;
    _streamAgentResults = {};
    ['TA-Agent','FA-Agent','CA-Agent','SA-Agent','MA-Agent','RA-Agent'].forEach(id => {
        const card = document.getElementById('stream-agent-' + id);
        if (!card) return;
        card.style.opacity = '0.4';
        card.className = 'agent-card';
        const meta = _AGENT_META[id];
        const sigEl = card.querySelector('.ach-signal');
        const confEl = card.querySelector('.ach-conf');
        const barEl = card.querySelector('.ach-progress-fill');
        const kvEl = card.querySelector('.ach-kv');
        const reasonEl = card.querySelector('.ach-reasoning');
        const tagsEl = card.querySelector('.ach-tags');
        if (sigEl) { sigEl.textContent = '--'; sigEl.style.color = 'var(--text-muted)'; }
        if (confEl) confEl.textContent = '--%';
        if (barEl) { barEl.style.width = '0%'; barEl.style.background = meta.color; }
        if (kvEl) kvEl.innerHTML = '';
        if (reasonEl) { reasonEl.textContent = meta.waitText; reasonEl.style.color = 'var(--text-muted)'; }
        if (tagsEl) tagsEl.innerHTML = '';
    });

    // Build body — support reconnection with task_id
    const reqBody = { stock_code: stockCode };
    if (taskId) reqBody.task_id = taskId;

    fetch('/api/agent/diagnose/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(reqBody),
        signal: streamAbortController.signal,
    }).then(response => {
        if (!response.ok) throw new Error('HTTP ' + response.status);

        // 连接成功：重置重连状态并隐藏遮罩
        _clearStreamRetryState();
        _hideReconnectingOverlay();

        // Persist task ID from response header for cross-page reconnection
        const respTaskId = response.headers.get('X-Task-ID');
        if (respTaskId && !taskId) {
            _activeTaskId = respTaskId;
            try { sessionStorage.setItem('mass_active_task', JSON.stringify({task_id:respTaskId,stock_code:stockCode,task_type:'diagnosis',started_at:Date.now()})); } catch(e) {}
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '', resultData = null;
        document.getElementById('streamStatus').textContent = taskId ? '重连中...' : '推理中...';

        function readChunk() {
            return reader.read().then(({ done, value }) => {
                if (done) {
                    document.getElementById('streamStatus').textContent = '完成';
                    _activateStage('result');
                    _clearActiveTask();
                    if (resultData) { currentResult = resultData; renderFullResult(resultData); }
                    return;
                }
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop();
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const event = JSON.parse(line.slice(6));
                            // Reconnect replay marker: skip, already counted
                            if (event.stage === 'reconnect') {
                                document.getElementById('streamStatus').textContent = '回放 ' + (event.replay_count||0) + ' 个事件...';
                                continue;
                            }
                            // Reconnect done marker
                            if (event.stage === 'done') {
                                document.getElementById('streamStatus').textContent = '完成（重连）';
                                _clearActiveTask();
                                if (resultData) { currentResult = resultData; renderFullResult(resultData); }
                                return;
                            }
                            handleStreamEventV2(event);
                            if (event.stage === 'result') resultData = event.data;
                            if (event.stage === 'error') {
                                document.getElementById('streamStatus').textContent = '出错';
                                _clearActiveTask();
                                return;
                            }
                        } catch (e) {}
                    }
                }
                return readChunk();
            });
        }
        return readChunk();
    }).catch(err => {
        if (err.name === 'AbortError') return;

        // 非手动取消：启动指数退避重连
        document.getElementById('streamStatus').textContent = '断开';

        if (_streamRetryCount < STREAM_MAX_RETRY) {
            const delay = Math.min(
                STREAM_BASE_DELAY_MS * Math.pow(2, _streamRetryCount),
                30000
            );
            _showReconnectingOverlay(
                `连接已断开，${(delay / 1000).toFixed(1)}秒后重连 (${_streamRetryCount + 1}/${STREAM_MAX_RETRY})...`
            );
            _streamReconnectTimer = setTimeout(() => {
                _streamRetryCount++;
                startStreamDiagnosis(_activeTaskId);
            }, delay);
        } else {
            _showReconnectingOverlay('连接失败，请刷新页面重试');
            document.getElementById('streamStatus').textContent = '连接失败';
            _clearActiveTask();
        }
    });
}

function _clearActiveTask() {
    _activeTaskId = null;
    try { sessionStorage.removeItem('mass_active_task'); } catch(e) {}
}

function _getActiveTask() {
    try { const r = sessionStorage.getItem('mass_active_task'); return r ? JSON.parse(r) : null; } catch(e) { return null; }
}

function _activateStage(name) {
    const order = ['init','data','agents','engine','chairman','result'];
    const idx = order.indexOf(name);
    for (let i = 0; i < order.length; i++) {
        const el = document.getElementById('stage-' + order[i]);
        if (el) el.className = 'stream-stage' + (i <= idx ? ' done' : '');
    }
    for (let i = 0; i < order.length - 1; i++) {
        const el = document.getElementById('conn-' + i);
        if (el) el.className = 'stage-connector' + (i < idx ? ' done' : '');
    }
}

function handleStreamEventV2(event) {
    const stage = event.stage;
    if (stage === 'init') {
        _activateStage('init');
    } else if (stage === 'data') {
        _activateStage('data');
    } else if (stage === 'agent_start') {
        _activateStage('agents');
        document.getElementById('streamConsensusMini').style.display = 'flex';
    } else if (stage === 'agent') {
        const res = event.agent_result || {};
        const sigMap = { '-1': '卖出', '0': '观望', '1': '买入' };
        const sigColorMap = { '-1': '#ef4444', '0': '#eab308', '1': '#22c55e' };
        const sig = sigMap[String(res.signal)] || '观望';
        const sigColor = sigColorMap[String(res.signal)] || '#eab308';
        const agentColorMap = {
            'TA-Agent': '#06b6d4', 'FA-Agent': '#3b82f6', 'CA-Agent': '#a855f7',
            'SA-Agent': '#f97316', 'MA-Agent': '#eab308', 'RA-Agent': '#ef4444'
        };
        const agentColor = agentColorMap[event.agent_id] || '#8b9bb4';
        const conf = Math.round((res.confidence || 0) * 100);
        updateStreamAgentCardV2(event.agent_id, res, agentColor, sig, sigColor, conf);
    } else if (stage === 'engine') {
        _activateStage('engine');
    } else if (stage === 'chairman') {
        _activateStage('chairman');
    } else if (stage === 'result') {
        _activateStage('result');
    }
}

// ========== Stream Agent Cards v2.3 — Full-Screen Update ==========

let _streamAgentsCompleted = 0;
let _streamAgentResults = {};

function updateStreamAgentCardV2(agentId, res, agentColor, sig, sigColor, conf) {
    const cardEl = document.getElementById('stream-agent-' + agentId);
    const countEl = document.getElementById('streamAgentCount');
    if (!cardEl) return;

    _streamAgentsCompleted++;
    _streamAgentResults[agentId] = res;
    if (countEl) countEl.textContent = _streamAgentsCompleted + '/6';

    // Highlight card
    cardEl.style.opacity = '1';
    cardEl.className = 'agent-card complete';

    // Header: signal + conf
    const sigEl = cardEl.querySelector('.ach-signal');
    if (sigEl) { sigEl.textContent = sig; sigEl.style.color = sigColor; }
    const confEl = cardEl.querySelector('.ach-conf');
    if (confEl) { confEl.textContent = conf + '%'; confEl.style.color = sigColor; }

    // Progress bar
    const barEl = cardEl.querySelector('.ach-progress-fill');
    if (barEl) { barEl.style.width = conf + '%'; barEl.style.background = agentColor; }

    // Key-value indicators
    const kvEl = cardEl.querySelector('.ach-kv');
    if (kvEl && res.indicators) {
        const items = _getAgentIndicatorTags(agentId, res.indicators);
        kvEl.innerHTML = items.map(t => '<span class="ach-kv-item">' + escapeHtml(t) + '</span>').join('');
    }

    // Reasoning
    const reasonEl = cardEl.querySelector('.ach-reasoning');
    if (reasonEl) {
        reasonEl.textContent = res.reasoning || '';
        reasonEl.style.color = 'var(--text-secondary)';
    }

    // Tags: key_factors + risk_flags
    const tagsEl = cardEl.querySelector('.ach-tags');
    if (tagsEl) {
        const factors = (res.key_factors || []).map(f => '<span class="mass-tag mass-tag-factor">' + escapeHtml(f) + '</span>').join('');
        const risks = (res.risk_flags || []).slice(0, 3).map(f => '<span class="mass-tag mass-tag-risk">' + escapeHtml(f) + '</span>').join('');
        tagsEl.innerHTML = factors + risks;
    }

    // Update consensus mini bar
    updateConsensusMiniBar();
}

function _getAgentIndicatorTags(agentId, ind) {
    switch (agentId) {
        case 'TA-Agent': return [
            ind.trend_direction || '趋势--',
            (ind.target_price_low && ind.target_price_high) ?
                ('目标 ' + ind.target_price_low + '-' + ind.target_price_high) : '目标--',
            (ind.chart_patterns && ind.chart_patterns.length) ? ind.chart_patterns[0] : '形态--'
        ];
        case 'FA-Agent':
            const ss = ind.sub_scores || {};
            return [
                '评分 ' + (ind.fundamental_score || '--'),
                'ROE ' + (ss.profitability || '--'),
                ind.valuation_gap || '估值--'
            ];
        case 'CA-Agent':
            return [
                ind.smart_money_direction || '主力--',
                ind.retail_vs_institutional || '机构--',
                '评分 ' + (ind.capital_score || '--')
            ];
        case 'SA-Agent':
            return [
                '情绪 ' + (ind.sentiment_index != null ? ind.sentiment_index : '--'),
                ind.crowd_behavior || 'crowd--',
                '分位 ' + (ind.sentiment_percentile != null ? ind.sentiment_percentile + '%' : '--')
            ];
        case 'MA-Agent':
            return [
                ind.market_cycle || '周期--',
                ind.sector_outlook || '行业--',
                '风格 ' + (ind.style_alignment || '--')
            ];
        case 'RA-Agent':
            return [
                '风险等级 ' + (ind.risk_level || '--'),
                '仓位上限 ' + (ind.max_position_pct != null ? (ind.max_position_pct * 100).toFixed(0) + '%' : '--'),
                'RR比 ' + (ind.risk_reward_ratio || '--')
            ];
        default: return [];
    }
}

function updateConsensusMiniBar() {
    let buy = 0, hold = 0, sell = 0;
    Object.values(_streamAgentResults).forEach(r => {
        if (r.signal === 1) buy++;
        else if (r.signal === -1) sell++;
        else hold++;
    });
    const total = buy + hold + sell;
    if (total === 0) return;
    document.getElementById('cMiniBuy').style.width = (buy / total * 100).toFixed(1) + '%';
    document.getElementById('cMiniHold').style.width = (hold / total * 100).toFixed(1) + '%';
    document.getElementById('cMiniSell').style.width = (sell / total * 100).toFixed(1) + '%';
    document.getElementById('streamConsensusMini').style.display = 'flex';
}

function appendStreamLine(container, label, color, text) {
    const line = document.createElement('div');
    line.style.marginBottom = '6px';
    line.style.fontSize = '13px';
    line.style.lineHeight = '1.6';
    line.innerHTML = `<span style="color:${color}; font-weight:600; font-size:11px; display:inline-block; min-width:90px;">[${label}]</span> <span style="color:var(--text-secondary);">${escapeHtml(text)}</span>`;
    container.appendChild(line);
    container.scrollTop = container.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
