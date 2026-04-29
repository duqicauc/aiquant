/**
 * Indicator help data: what each indicator means and how to read it.
 * Extracted from docs/guides/TECHNICAL_ANALYSIS_GUIDE.md
 */

export interface IndicatorHelp {
  title: string
  what: string
  how: string
  bullishSignal: string
  bearishSignal: string
  tips?: string
}

/** Static help dictionary for all indicators. */
export const INDICATOR_HELP: Record<string, IndicatorHelp> = {
  // ─── 趋势方向 ───
  adx_dmi: {
    title: 'ADX/DMI',
    what: 'ADX（平均趋向指数）衡量趋势强度，+DI 与 -DI 判断多空方向。',
    how: 'ADX > 25 表示趋势较强，适合顺势操作；ADX < 20 表示震荡市，减少操作。+DI 上穿 -DI 为多头启动信号。',
    bullishSignal: 'ADX > 25 且 +DI 在 -DI 上方，趋势强劲偏多。',
    bearishSignal: 'ADX > 25 且 -DI 在 +DI 上方，趋势强劲偏空。',
    tips: 'ADX 本身不指示方向，只指示强度；需配合 DI 线判断多空。',
  },
  supertrend: {
    title: 'SuperTrend',
    what: '基于 ATR 通道的趋势跟踪指标，A股效果极佳的趋势工具。',
    how: '指标线翻绿（价格在 SuperTrend 线上方）= 多头持仓；翻红（价格在下方）= 空头或空仓。顺势持仓，翻转即操作信号。',
    bullishSignal: '价格站上 SuperTrend 线，显示绿色，趋势转多。',
    bearishSignal: '价格跌破 SuperTrend 线，显示红色，趋势转空。',
    tips: '默认参数 period=10, multiplier=3；A股波动大，可调大 multiplier 减少假信号。',
  },
  ichimoku: {
    title: '一目均衡（Ichimoku）',
    what: '日本综合技术分析指标，通过云层、转换线、基准线判断趋势与支撑压力。',
    how: '价格在云上方 + 转换线 > 基准线 = 强烈看涨；价格在云中 = 震荡整理；价格在云下方 = 看跌。云层本身构成支撑/压力带。',
    bullishSignal: '价格在云上方，Tenkan（转换线）> Kijun（基准线），强烈看涨。',
    bearishSignal: '价格在云下方，Tenkan < Kijun，强烈看跌。',
    tips: '一目均衡是「一眼看均衡」的系统，单图即可判断趋势、支撑、压力、信号。',
  },
  sar: {
    title: 'SAR（抛物线转向）',
    what: '抛物线 Stop And Reverse，给出趋势反转点与建议止损位。',
    how: 'SAR 点从价格下方翻到上方 = 多头转空头；从上方翻到下方 = 空头转多头。SAR 点也可作为移动止损位参考。',
    bullishSignal: 'SAR 点翻转至 K线下方，趋势转多。',
    bearishSignal: 'SAR 点翻转至 K线上方，趋势转空。',
    tips: '在强趋势中 SAR 效果很好，但在震荡市中会频繁翻转产生假信号。',
  },
  atr_channel: {
    title: 'ATR Channel（ATR通道）',
    what: '基于 ATR（平均真实波幅）构建的波动率通道，判断价格波动区间。',
    how: '突破上轨 = 波动率扩张，可能出现 breakout；接近下轨 = 低吸机会；通道收窄 = 即将变盘。',
    bullishSignal: '价格突破上轨，波动率扩张，可能启动上涨。',
    bearishSignal: '价格跌破下轨，波动率扩张，可能加速下跌。',
    tips: '通道收窄后的首次突破往往意味着新一轮趋势开始。',
  },

  // ─── 量价验证 ───
  vwap: {
    title: 'VWAP（成交量加权平均价）',
    what: '机构平均成本线，反映大资金的实际持仓成本。',
    how: '价格 > VWAP = 强于机构成本，持仓信心足；价格 < VWAP = 弱于成本，需警惕。偏离幅度越大，回归动力越强。',
    bullishSignal: '价格稳定在 VWAP 上方，说明机构浮盈，趋势偏多。',
    bearishSignal: '价格跌破 VWAP 且持续在下方运行，说明机构被套或出货。',
    tips: 'VWAP 日内交易常用，日线级别同样有效，是判断机构成本的核心指标。',
  },
  cmf: {
    title: 'CMF（Chaikin Money Flow）',
    what: 'Chaikin 资金流向指标，衡量资金流向强度，范围 -1 ~ +1。',
    how: 'CMF > 0.05 资金净流入，市场偏强；CMF < -0.05 资金净流出，市场偏弱。重点关注底背离：价格创新低但 CMF 上升。',
    bullishSignal: 'CMF > 0.05，资金持续流入，或出现底背离。',
    bearishSignal: 'CMF < -0.05，资金持续流出，或出现顶背离。',
    tips: '背离信号比绝对值更可靠，是预判转折的利器。',
  },
  mfi: {
    title: 'MFI（Money Flow Index）',
    what: '量价结合的 RSI，将成交量纳入超买超卖判断，比 RSI 更可靠。',
    how: 'MFI > 80 量价超买，警惕回调；MFI < 20 量价超卖，关注反弹机会。与 RSI 不同，MFI 考虑了成交量权重。',
    bullishSignal: 'MFI < 20 进入超卖区，或从低位回升，买入机会。',
    bearishSignal: 'MFI > 80 进入超买区，或从高位回落，卖出警惕。',
    tips: 'MFI 在 20-80 之间震荡时信号不明确，需配合趋势指标使用。',
  },
  pvo: {
    title: 'PVO（Percentage Volume Oscillator）',
    what: '量能版 MACD，衡量成交量趋势的强弱变化。',
    how: 'PVO 金叉 = 成交量趋势转强，适合追涨；PVO 死叉 = 量能萎缩，上涨动力不足，需警惕。',
    bullishSignal: 'PVO 金叉，成交量趋势转强，资金活跃度提升。',
    bearishSignal: 'PVO 死叉，成交量趋势萎缩，市场参与度下降。',
    tips: '价格上涨但 PVO 死叉 = 量价背离，可能是顶部信号。',
  },
  ad_line: {
    title: 'A/D Line（累积/派发线）',
    what: 'Accumulation/Distribution Line，追踪资金累积与派发的累计值。',
    how: 'A/D Line 与价格同步上升 = 资金在累积，趋势健康；价格新高但 A/D Line 不新高 = 顶背离，危险信号。',
    bullishSignal: 'A/D Line 持续上升，资金在累积，趋势得到确认。',
    bearishSignal: '价格新高但 A/D Line 走低，出现顶背离，主力可能在出货。',
    tips: 'A/D Line 的背离信号是判断主力建仓/出货的重要依据。',
  },
  volume_profile: {
    title: 'Volume Profile（成交量分布）',
    what: '展示不同价格区间的成交量分布，识别主力成本区与筹码密集带。',
    how: 'POC（最大成交量价）= 主力成本区；Value Area（70%筹码区）= 正常波动区间。价格突破 VA 上下轨 = 变盘信号。',
    bullishSignal: '价格从 VA 下轨向上突破 POC，筹码松动后重新聚集。',
    bearishSignal: '价格从 VA 上轨向下跌破 POC，支撑变压力。',
    tips: 'POC 附近往往是强支撑/压力，回调到 POC 附近是较好的低吸/高抛位置。',
  },

  // ─── 形态识别 ───
  harmonic: {
    title: '谐波形态（Harmonic Patterns）',
    what: '基于斐波那契比例的反转形态（加特利、蝴蝶、螃蟹、蝙蝠），给出精确入场位。',
    how: '在 D 点完成形态后入场，预设目标位和止损位，风险回报比清晰。看涨形态在下跌末端，看跌形态在上涨末端。',
    bullishSignal: '识别到看涨谐波形态（如看涨蝴蝶/蝙蝠），D点即为潜在反转位。',
    bearishSignal: '识别到看跌谐波形态（如看跌螃蟹/加特利），D点即为潜在反转位。',
    tips: '谐波形态有明确的数学比例规则，比艾略特波浪更适合程序化识别。',
  },
  fractals: {
    title: '分形（Fractals）',
    what: '比尔·威廉姆斯提出的 5 根 K线 分形结构，标识潜在支撑与压力。',
    how: '分形高点 = 潜在压力区；分形低点 = 潜在支撑区。价格突破分形高点 = 趋势延续信号；跌破分形低点 = 趋势走弱。',
    bullishSignal: '价格突破最近分形高点，趋势延续向上。',
    bearishSignal: '价格跌破最近分形低点，趋势延续向下。',
    tips: '分形常与其他指标（如 Alligator）配合使用，单独使用信号较多。',
  },
}

/** MTFA sub-indicator help */
export const MTFA_SUB_HELP: Record<string, IndicatorHelp> = {
  rsi: {
    title: 'RSI（相对强弱指数）',
    what: '衡量价格变动的速度和幅度，判断超买超卖。',
    how: 'RSI > 70 超买，< 30 超卖。在强势市场中，RSI 可在 50-80 之间反复，不必死板等待 70/30。',
    bullishSignal: 'RSI 从 30 以下回升，或在 50 上方运行。',
    bearishSignal: 'RSI 从 70 以上回落，或在 50 下方运行。',
  },
  macd: {
    title: 'MACD',
    what: '指数平滑异同平均线，判断趋势方向和动量变化。',
    how: 'DIF 上穿 DEA（金叉）= 多头信号；DIF 下穿 DEA（死叉）= 空头信号。MACD 柱状图反映动量强度。',
    bullishSignal: 'MACD 金叉，或 DIF 在零轴上方运行。',
    bearishSignal: 'MACD 死叉，或 DIF 在零轴下方运行。',
  },
  ma_alignment: {
    title: '均线排列',
    what: '多条均线的多头排列或空头排列状态。',
    how: '短期均线在长期均线上方且向上发散 = 多头排列，强势；反之 = 空头排列，弱势。',
    bullishSignal: 'MA5 > MA10 > MA20 > MA60，多头排列。',
    bearishSignal: 'MA5 < MA10 < MA20 < MA60，空头排列。',
  },
  bollinger: {
    title: '布林带（Bollinger Bands）',
    what: '由中轨（MA20）和上下轨（±2σ）组成的波动通道。',
    how: '价格触及上轨 = 可能超买/突破；触及下轨 = 可能超卖/反弹。通道收窄 = 即将选择方向。',
    bullishSignal: '价格从下轨反弹，或突破上轨持续运行。',
    bearishSignal: '价格从上轨回落，或跌破下轨持续运行。',
  },
  price_vs_ma20: {
    title: '价 vs MA20',
    what: '当前价格与 20 日均线的偏离程度。',
    how: '价格在 MA20 上方 = 短期偏多；在下方 = 短期偏空。偏离幅度过大有回归均值需求。',
    bullishSignal: '价格在 MA20 上方，短期趋势向上。',
    bearishSignal: '价格在 MA20 下方，短期趋势向下。',
  },
}

/** Moneyflow sub-indicator help */
export const MONEYFLOW_HELP: Record<string, IndicatorHelp> = {
  main_force: {
    title: '主力动向',
    what: '特大单、大单的资金净流入情况，反映机构真实意图。',
    how: '特大单连续净流入 = 机构建仓/加仓，跟随；特大单净流出 = 机构减仓，警惕。',
    bullishSignal: '主力连续净流入，资金在主动买入。',
    bearishSignal: '主力连续净流出，资金在主动卖出。',
  },
  retail_contrarian: {
    title: '散户反向',
    what: '散户行为往往是反向指标：恐慌时割肉（底部）、狂热时追涨（顶部）。',
    how: '小单净买入 + 大涨 = 散户狂热，警惕顶部；小单净卖出 + 大跌 = 散户恐慌，可能是底部。',
    bullishSignal: '散户恐慌割肉（小单净卖出 + 大跌），反向看多。',
    bearishSignal: '散户狂热追涨（小单净买入 + 大涨），反向看空。',
    tips: 'A股是散户主导市场，资金流向比技术指标更有预测力。',
  },
  capital_trend: {
    title: '资金趋势',
    what: '资金连续流入/流出的天数与趋势强度。',
    how: '连续流入天数越多，趋势越稳固；突然中断需警惕。趋势强度量化资金变化的速率。',
    bullishSignal: '连续多日资金净流入，趋势稳固。',
    bearishSignal: '连续多日资金净流出，趋势疲弱。',
  },
}

/**
 * Build a dynamic signal interpretation string for a given indicator result.
 */
export function buildSignalInterpretation(
  key: string,
  data: { value?: number | string; signal?: string; strength?: number; detail?: Record<string, any>; count?: number }
): string {
  const signal = data.signal || '无信号'
  const strength = data.strength

  if (key === 'vwap' && data.detail?.distance_pct !== undefined) {
    const pct = data.detail.distance_pct
    return `价格${pct >= 0 ? '高于' : '低于'} VWAP ${Math.abs(pct).toFixed(2)}%，${pct >= 0 ? '机构当前浮盈' : '机构当前浮亏'}。`
  }
  if (key === 'cmf' && typeof data.value === 'number') {
    return `CMF = ${data.value.toFixed(3)}，${data.value > 0.05 ? '资金净流入' : data.value < -0.05 ? '资金净流出' : '资金流平衡'}。`
  }
  if (key === 'mfi' && typeof data.value === 'number') {
    return `MFI = ${data.value.toFixed(1)}，${data.value > 80 ? '量价超买' : data.value < 20 ? '量价超卖' : '处于中性区'}。`
  }
  if (key === 'adx_dmi' && data.detail?.adx !== undefined) {
    const adx = data.detail.adx
    return `ADX = ${adx.toFixed(1)}，${adx > 25 ? '趋势较强' : adx < 20 ? '震荡市' : '趋势中等'}。`
  }
  if (key === 'supertrend') {
    return signal
  }
  if (key === 'volume_profile' && data.detail?.poc !== undefined) {
    return `POC（主力成本区）= ${data.detail.poc}，当前价格${data.detail.position || '—'}。`
  }
  if (key === 'harmonic' && data.count !== undefined) {
    return data.count === 0 ? '当前未识别到谐波形态。' : `识别到 ${data.count} 个谐波形态，关注 D 点入场机会。`
  }
  if (key === 'fractals' && data.count !== undefined) {
    return `识别到 ${data.count} 个分形结构，可作为支撑/压力参考。`
  }

  // Generic fallback
  if (strength !== undefined) {
    return `信号强度 ${strength}/10，${signal}。`
  }
  return signal
}
