"""
Tushare Pro 数据获取器

封装 Tushare Pro API，提供统一的数据获取接口。
支持：股票列表、日线数据、周线数据、技术因子、每日指标等。
"""

import os
from datetime import datetime
from typing import List, Optional

import pandas as pd
from dotenv import load_dotenv

from src.data.fetcher.base_fetcher import BaseFetcher
from src.utils.logger import log
from src.utils.rate_limiter import get_api_limiter, init_rate_limiter

# 加载环境变量
load_dotenv()


class TushareFetcher(BaseFetcher):
    """Tushare Pro 数据获取器"""

    def __init__(self, token: str = None, points: int = None):
        """
        初始化 Tushare 数据获取器

        Args:
            token: Tushare Pro Token，如果为None则从环境变量读取
            points: Tushare 积分，用于设置限流级别
        """
        # 先设置 token，因为父类初始化会调用 _init_connection
        self.token = token or os.getenv("TUSHARE_TOKEN")

        if not self.token or self.token == "YOUR_TUSHARE_TOKEN":
            raise ValueError(
                "请设置有效的 TUSHARE_TOKEN！\n"
                "1. 在 https://tushare.pro/register 注册账号\n"
                "2. 在 .env 文件中设置 TUSHARE_TOKEN=你的token"
            )

        # 初始化限流器（根据积分设置）
        if points is None:
            points = int(os.getenv("TUSHARE_POINTS", "120"))
        init_rate_limiter(points)
        self.rate_limiter = get_api_limiter()

        # 调用父类初始化（会触发 _init_connection）
        super().__init__(source_name="tushare")

        log.info(f"TushareFetcher 初始化成功 (积分级别: {points})")

    def _init_connection(self):
        """初始化 Tushare Pro 连接"""
        import tushare as ts

        ts.set_token(self.token)
        self.pro = ts.pro_api()

    def get_stock_list(self, list_status: str = "L", exchange: str = None, **kwargs) -> pd.DataFrame:
        """
        获取股票列表

        Args:
            list_status: 上市状态 ('L'上市, 'D'退市, 'P'暂停上市)
            exchange: 交易所 ('SSE'上交所, 'SZSE'深交所, 'BSE'北交所)

        Returns:
            股票列表DataFrame，包含 ts_code, name, list_date 等字段
        """
        self.rate_limiter.wait_if_needed()

        try:
            df = self.pro.stock_basic(
                list_status=list_status,
                exchange=exchange,
                fields="ts_code,symbol,name,area,industry,list_date,market,is_hs",
            )
            log.info(f"获取股票列表成功: {len(df)} 只")
            return df
        except Exception as e:
            log.error(f"获取股票列表失败: {e}")
            raise

    def get_daily_data(
        self, stock_code: str, start_date: str, end_date: Optional[str] = None, adjust: str = "qfq"
    ) -> pd.DataFrame:
        """
        获取日线数据

        Args:
            stock_code: 股票代码 (如 '000001.SZ' 或 '000001')
            start_date: 开始日期 (YYYYMMDD 或 YYYY-MM-DD)
            end_date: 结束日期 (YYYYMMDD 或 YYYY-MM-DD)，如果为None则使用今天
            adjust: 复权类型 ('qfq'前复权, 'hfq'后复权, ''不复权)

        Returns:
            日线数据DataFrame
        """
        # 格式化股票代码和日期
        ts_code = self.format_stock_code(stock_code)
        start_date = self.format_date(start_date)

        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        else:
            end_date = self.format_date(end_date)

        self.rate_limiter.wait_if_needed()

        try:
            import tushare as ts

            # 使用 pro_bar 获取复权数据
            df = ts.pro_bar(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                adj=adjust,
                factors=["tor", "vr"],  # 换手率、量比
            )

            if df is not None and not df.empty:
                # 转换日期格式并排序
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df = df.sort_values("trade_date").reset_index(drop=True)

            return df if df is not None else pd.DataFrame()

        except Exception as e:
            log.warning(f"获取日线数据失败 {ts_code} ({start_date}~{end_date}): {e}")
            return pd.DataFrame()

    def get_weekly_data(self, ts_code: str, start_date: str, end_date: str, adjust: str = "qfq") -> pd.DataFrame:
        """
        获取周线数据

        Args:
            ts_code: 股票代码
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            adjust: 复权类型

        Returns:
            周线数据DataFrame
        """
        self.rate_limiter.wait_if_needed()

        try:
            df = self.pro.weekly(ts_code=ts_code, start_date=start_date, end_date=end_date, adj=adjust)

            if df is not None and not df.empty:
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df = df.sort_values("trade_date").reset_index(drop=True)

            return df if df is not None else pd.DataFrame()

        except Exception as e:
            log.warning(f"获取周线数据失败 {ts_code}: {e}")
            return pd.DataFrame()

    def get_daily_basic(
        self,
        stock_code: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        获取每日指标（市值、市盈率、换手率等）

        Args:
            stock_code: 股票代码（可选，如果为None则获取所有股票）
            start_date: 开始日期 (YYYYMMDD 或 YYYY-MM-DD)，可选
            end_date: 结束日期 (YYYYMMDD 或 YYYY-MM-DD)，可选
            **kwargs: 其他参数（trade_date等）

        Returns:
            每日指标DataFrame
        """
        self.rate_limiter.wait_if_needed()

        try:
            params = {
                "fields": "ts_code,trade_date,close,turnover_rate,turnover_rate_f,"
                "volume_ratio,pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,"
                "total_share,float_share,free_share,total_mv,circ_mv"
            }

            # 处理trade_date参数（优先使用）
            if "trade_date" in kwargs:
                params["trade_date"] = self.format_date(kwargs["trade_date"])
            else:
                # 处理日期范围
                if start_date:
                    start_date = self.format_date(start_date)
                    params["start_date"] = start_date

                if end_date:
                    end_date = self.format_date(end_date)
                    params["end_date"] = end_date
                elif start_date:
                    # 如果只有start_date，使用它作为end_date
                    params["end_date"] = start_date

            # 处理股票代码
            if stock_code:
                ts_code = self.format_stock_code(stock_code)
                params["ts_code"] = ts_code

            df = self.pro.daily_basic(**params)

            if df is not None and not df.empty:
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df = df.sort_values("trade_date").reset_index(drop=True)

            return df if df is not None else pd.DataFrame()

        except Exception as e:
            log.warning(f"获取每日指标失败: {e}")
            return pd.DataFrame()

    def get_stk_factor(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        fields: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        获取技术因子（MACD、KDJ、RSI、DMI、MFI、TAQ等）

        Args:
            ts_code: 股票代码（ETF也可用，如 510300.SH）
            start_date: 开始日期
            end_date: 结束日期
            fields: 可选，自定义字段列表；默认返回完整常用因子

        Returns:
            技术因子DataFrame
        """
        self.rate_limiter.wait_if_needed()

        if fields is None:
            # 完整常用因子字段（覆盖趋势/动量/量价/波动/超买超卖）
            fields = (
                "ts_code,trade_date,close,"
                # 趋势
                "macd_dif,macd_dea,macd,"
                "ma5,ma10,ma_20d,ma30,ma60,ma90,"
                "ema_5,ema_10,ema_20,ema_60,"
                "expma_12,expma_50,"
                "trix,trma,"
                "dfma_dif,dfma_difma,"
                # 动量/超买超卖
                "kdj_k,kdj_d,kdj_j,"
                "rsi_6,rsi_12,rsi_24,"
                "wr,wr1,"
                "cci,"
                "psy,psyma,"
                # 波动/通道
                "boll_upper,boll_mid,boll_lower,"
                "atr,"
                "ktn_upper,ktn_mid,ktn_down,"
                "taq_up,taq_mid,taq_down,"
                # 量价
                "obv,"
                "mfi,"
                "emv,maemv,"
                "vr,"
                "volume_ratio,"
                # 趋势强度
                "dmi_pdi,dmi_mdi,dmi_adx,dmi_adxr,"
                # 偏离
                "bias_short,bias_mid,bias_long,"
                # 其他
                "roc,maroc,"
                "cr,"
                "brar_br,brar_ar,"
                "bbi,"
                "dpo,madpo,"
                "asi,asit,"
                "mass,ma_mass,"
                "mtm,mtmma,"
                "xsii_td1,xsii_td2,xsii_td3,xsii_td4"
            )

        try:
            df = self.pro.stk_factor(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                fields=fields,
            )

            if df is not None and not df.empty:
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df = df.sort_values("trade_date").reset_index(drop=True)

            return df if df is not None else pd.DataFrame()

        except Exception as e:
            log.warning(f"获取技术因子失败 {ts_code}: {e}")
            return pd.DataFrame()

    def get_etf_daily_basic(
        self,
        ts_code: Optional[str] = None,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        获取 ETF 每日指标（换手率、量比等）
        复用 daily_basic 接口，ETF 作为场内品种同样适用

        Args:
            ts_code: ETF 代码，如 '510300.SH'
            trade_date: 交易日期（YYYYMMDD）
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            DataFrame 含 turnover_rate, turnover_rate_f, volume_ratio 等
        """
        self.rate_limiter.wait_if_needed()
        try:
            params = {
                "fields": "ts_code,trade_date,close,turnover_rate,turnover_rate_f,volume_ratio,"
                "total_share,float_share,total_mv,circ_mv"
            }
            if trade_date:
                params["trade_date"] = self.format_date(trade_date)
            if start_date:
                params["start_date"] = self.format_date(start_date)
            if end_date:
                params["end_date"] = self.format_date(end_date)
            if ts_code:
                params["ts_code"] = ts_code

            df = self.pro.daily_basic(**params)
            if df is not None and not df.empty:
                if "trade_date" in df.columns:
                    df["trade_date"] = pd.to_datetime(df["trade_date"])
                    df = df.sort_values("trade_date").reset_index(drop=True)
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            log.warning(f"获取ETF每日指标失败 {ts_code}: {e}")
            return pd.DataFrame()

    def get_suspend_info(self, ts_code: str = None, trade_date: str = None, suspend_type: str = None) -> pd.DataFrame:
        """
        获取停牌信息

        Args:
            ts_code: 股票代码（可选）
            trade_date: 交易日期（可选）
            suspend_type: 停牌类型 ('S'停牌, 'R'复牌)

        Returns:
            停牌信息DataFrame
        """
        self.rate_limiter.wait_if_needed()

        try:
            params = {}
            if ts_code:
                params["ts_code"] = ts_code
            if trade_date:
                params["trade_date"] = trade_date
            if suspend_type:
                params["suspend_type"] = suspend_type

            df = self.pro.suspend_d(**params)
            return df if df is not None else pd.DataFrame()

        except Exception as e:
            log.warning(f"获取停牌信息失败: {e}")
            return pd.DataFrame()

    def get_trade_calendar(self, start_date: str, end_date: str, exchange: str = "SSE") -> pd.DataFrame:
        """
        获取交易日历

        Args:
            start_date: 开始日期
            end_date: 结束日期
            exchange: 交易所（默认上交所）

        Returns:
            交易日历DataFrame
        """
        self.rate_limiter.wait_if_needed()

        try:
            df = self.pro.trade_cal(
                exchange=exchange,
                start_date=start_date,
                end_date=end_date,
                fields="exchange,cal_date,is_open,pretrade_date",
            )

            if df is not None and not df.empty:
                df["cal_date"] = pd.to_datetime(df["cal_date"])
                df = df.sort_values("cal_date").reset_index(drop=True)

            return df if df is not None else pd.DataFrame()

        except Exception as e:
            log.warning(f"获取交易日历失败: {e}")
            return pd.DataFrame()

    def get_minute_data(
        self, stock_code: str, freq: str = "5min", start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        获取分钟数据（Tushare Pro 暂不支持，返回空DataFrame）

        Args:
            stock_code: 股票代码
            freq: 频率（1min, 5min, 15min, 30min, 60min）
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            分钟数据DataFrame（Tushare Pro 暂不支持，返回空DataFrame）
        """
        log.warning("Tushare Pro 暂不支持分钟数据获取")
        return pd.DataFrame()

    def get_fundamental_data(
        self, stock_code: str, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        获取基本面数据（通过财务指标接口实现）

        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            基本面数据DataFrame
        """
        # Tushare Pro 的基本面数据需要通过其他接口获取
        # 这里返回空DataFrame，具体实现可以根据需要添加
        log.warning("基本面数据获取功能待实现")
        return pd.DataFrame()

    def batch_get_daily_basic(self, trade_date: str, stock_codes: List[str] = None) -> pd.DataFrame:
        """
        批量获取某日所有股票的每日指标

        Args:
            trade_date: 交易日期 (YYYYMMDD)
            stock_codes: 股票代码列表（可选，为None则获取所有）

        Returns:
            每日指标DataFrame
        """
        # 使用新的接口，传入trade_date参数
        df = self.get_daily_basic(stock_code=None, trade_date=trade_date)

        if df.empty:
            return df

        if stock_codes:
            # 格式化股票代码列表
            formatted_codes = [self.format_stock_code(code) for code in stock_codes]
            df = df[df["ts_code"].isin(formatted_codes)]

        return df

    def get_index_daily(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取指数日线数据

        Args:
            ts_code: 指数代码（如 '000001.SH' 上证指数）
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)

        Returns:
            指数日线数据DataFrame
        """
        self.rate_limiter.wait_if_needed()

        try:
            df = self.pro.index_daily(
                ts_code=ts_code,
                start_date=self.format_date(start_date),
                end_date=self.format_date(end_date),
                fields="ts_code,trade_date,close,open,high,low,pre_close,change,pct_chg,vol,amount",
            )

            if df is not None and not df.empty:
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df = df.sort_values("trade_date").reset_index(drop=True)

            return df if df is not None else pd.DataFrame()

        except Exception as e:
            log.warning(f"获取指数日线数据失败 {ts_code}: {e}")
            return pd.DataFrame()

    def get_limit_list(self, trade_date: str) -> pd.DataFrame:
        """
        获取涨跌停统计

        Args:
            trade_date: 交易日期 (YYYYMMDD)

        Returns:
            涨跌停统计DataFrame
        """
        self.rate_limiter.wait_if_needed()

        try:
            df = self.pro.limit_list_d(trade_date=self.format_date(trade_date))

            return df if df is not None else pd.DataFrame()

        except Exception as e:
            log.warning(f"获取涨跌停统计失败: {e}")
            return pd.DataFrame()

    def get_margin_data(self, trade_date: str = None, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        获取融资融券数据

        Args:
            trade_date: 交易日期 (YYYYMMDD)
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            融资融券数据DataFrame
        """
        self.rate_limiter.wait_if_needed()

        try:
            params = {}
            if trade_date:
                params["trade_date"] = self.format_date(trade_date)
            if start_date:
                params["start_date"] = self.format_date(start_date)
            if end_date:
                params["end_date"] = self.format_date(end_date)

            df = self.pro.margin(**params)

            if df is not None and not df.empty:
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df = df.sort_values("trade_date").reset_index(drop=True)

            return df if df is not None else pd.DataFrame()

        except Exception as e:
            log.warning(f"获取融资融券数据失败: {e}")
            return pd.DataFrame()

    def get_moneyflow(self, trade_date: str, ts_codes: Optional[list] = None) -> pd.DataFrame:
        """
        获取个股资金流向数据（主力净流入等）

        接口说明：https://tushare.pro/document/2?doc_id=170
        用户积分≥2000可调取

        Args:
            trade_date: 交易日期 (YYYYMMDD)
            ts_codes: 股票代码列表，若为None则拉取当日全市场

        Returns:
            DataFrame 含 ts_code, trade_date, net_mf_amount(净流入额万元),
            buy_elg_amount(特大单买入), sell_elg_amount(特大单卖出) 等
        """
        self.rate_limiter.wait_if_needed()
        try:
            params = {"trade_date": self.format_date(trade_date)}
            df = self.pro.moneyflow(**params)
            if df is None or df.empty:
                return pd.DataFrame()
            # 可选：只保留指定股票
            if ts_codes:
                ts_set = set(self.format_stock_code(c) for c in ts_codes)
                df = df[df["ts_code"].isin(ts_set)].copy()
            return df
        except Exception as e:
            log.warning(f"获取资金流向失败: {e}")
            return pd.DataFrame()

    def get_sector_moneyflow(self, trade_date: str, top_n: int = 30) -> pd.DataFrame:
        """
        获取板块资金流向（东方财富行业/概念板块主力净流入）

        接口说明：https://tushare.pro/document/2?doc_id=291
        用户积分≥5000可调取（moneyflow_dc）

        Args:
            trade_date: 交易日期 (YYYYMMDD)
            top_n: 返回主力净流入最多的 TopN 板块

        Returns:
            DataFrame，含 ts_code, ts_name, net_mf_amount(万元, 正数=净流入), pct_chg 等；
            若接口不可用则返回空 DataFrame
        """
        self.rate_limiter.wait_if_needed()
        try:
            df = self.pro.moneyflow_dc(trade_date=self.format_date(trade_date))
            if df is None or df.empty:
                return pd.DataFrame()
            # 只保留有净流入量字段的行
            if "net_mf_amount" not in df.columns:
                # 尝试兼容字段名差异
                for col in ("net_buy_amount", "net_amount"):
                    if col in df.columns:
                        df = df.rename(columns={col: "net_mf_amount"})
                        break
            if "net_mf_amount" not in df.columns:
                return pd.DataFrame()
            df = df[df["net_mf_amount"].notna()].copy()
            # 按净流入降序，取 TopN（净流入为正的板块）
            df = df.sort_values("net_mf_amount", ascending=False).head(top_n)
            return df
        except Exception as e:
            log.warning(f"获取板块资金流向失败: {e}")
            return pd.DataFrame()

    def get_stock_industry_map(self, ts_codes: Optional[list] = None) -> dict:
        """
        获取股票与行业映射（用于热点行业匹配）

        Args:
            ts_codes: 股票代码列表，若为None则返回全市场

        Returns:
            dict: {ts_code: industry_name}
        """
        self.rate_limiter.wait_if_needed()
        try:
            df = self.pro.stock_basic(list_status="L", fields="ts_code,industry")
            if df is None or df.empty:
                return {}
            df = df.dropna(subset=["industry"])
            out = df.set_index("ts_code")["industry"].astype(str).to_dict()
            if ts_codes:
                ts_set = set(self.format_stock_code(c) for c in ts_codes)
                out = {k: v for k, v in out.items() if k in ts_set}
            return out
        except Exception as e:
            log.warning(f"获取股票行业映射失败: {e}")
            return {}

    def concept_detail(self, ts_code: str = None) -> pd.DataFrame:
        """
        获取概念板块成分股

        Args:
            ts_code: 股票代码，如果提供则获取该股票所属的概念板块

        Returns:
            概念板块成分股DataFrame，包含 concept_name, ts_code 等字段
        """
        self.rate_limiter.wait_if_needed()

        try:
            df = self.pro.concept_detail(ts_code=ts_code)
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            log.debug(f"获取概念板块成分股失败: {e}")
            return pd.DataFrame()

    def get_hot_concepts(self, trade_date: str, top_n: int = 20, min_up_nums: int = 3) -> pd.DataFrame:
        """
        获取热门概念板块（基于涨停股票数）

        Args:
            trade_date: 交易日期 (YYYYMMDD)
            top_n: 返回TopN热门板块
            min_up_nums: 最少涨停股票数

        Returns:
            热门概念板块DataFrame，包含 ts_code, name, up_nums, pct_chg, rank 等字段
        """
        self.rate_limiter.wait_if_needed()

        try:
            # 使用limit_cpt_list获取最强板块统计
            df = self.pro.limit_cpt_list(trade_date=self.format_date(trade_date))

            if df is not None and not df.empty:
                # 过滤最少涨停股票数
                df = df[df["up_nums"] >= min_up_nums].copy()
                # 按涨停数排序
                df = df.sort_values("up_nums", ascending=False).head(top_n)
                # 添加热度得分（涨停数 + 涨幅加权）
                df["heat_score"] = df["up_nums"] * 0.7 + df["pct_chg"].abs() * 0.3
                df = df.sort_values("heat_score", ascending=False)

            return df if df is not None else pd.DataFrame()
        except Exception as e:
            log.warning(f"获取热门概念板块失败: {e}")
            return pd.DataFrame()

    def get_hot_industries(self, trade_date: str, top_n: int = 20, min_pct_chg: float = 1.0) -> pd.DataFrame:
        """
        获取热门行业板块（基于申万行业涨幅）

        Args:
            trade_date: 交易日期 (YYYYMMDD)
            top_n: 返回TopN热门行业
            min_pct_chg: 最小涨幅要求（%）

        Returns:
            热门行业板块DataFrame，包含 ts_code, name, pct_chg 等字段
        """
        self.rate_limiter.wait_if_needed()

        try:
            # 使用sw_daily获取申万行业日线行情
            df = self.pro.sw_daily(trade_date=self.format_date(trade_date))

            if df is not None and not df.empty:
                # 过滤最小涨幅
                df = df[df["pct_change"] >= min_pct_chg].copy()
                # 按涨幅排序
                df = df.sort_values("pct_change", ascending=False).head(top_n)

            return df if df is not None else pd.DataFrame()
        except Exception as e:
            log.warning(f"获取热门行业板块失败: {e}")
            return pd.DataFrame()

    def get_ths_hot(
        self, trade_date: str = None, market: str = "概念板块", is_new: str = "Y", top_n: int = 50
    ) -> pd.DataFrame:
        """
        获取同花顺热榜数据（推荐使用）

        接口说明：https://tushare.pro/document/2?doc_id=320
        获取同花顺App热榜数据，包括热股、概念板块、ETF、可转债、港美股等

        Args:
            trade_date: 交易日期 (YYYYMMDD)，如果为None则使用最新数据
            market: 热榜类型，可选值：
                - '热股': 热门股票
                - '概念板块': 热门概念板块（推荐）
                - '行业板块': 热门行业板块
                - 'ETF': 热门ETF
                - '可转债': 热门可转债
                - '港股': 热门港股
                - '美股': 热门美股
                - '热基': 热门基金
            is_new: 是否最新（默认Y，如果为N则为盘中和盘后阶段采集）
            top_n: 返回TopN热门数据

        Returns:
            热榜数据DataFrame，包含：
            - ts_code: 代码
            - ts_name: 名称
            - rank: 排行
            - pct_change: 涨跌幅%
            - hot: 热度值
            - concept: 标签/概念
            - rank_reason: 上榜解读
        """
        self.rate_limiter.wait_if_needed()

        try:
            params = {"market": market, "is_new": is_new}

            if trade_date:
                params["trade_date"] = self.format_date(trade_date)

            df = self.pro.ths_hot(**params)

            if df is not None and not df.empty:
                # 按热度值排序
                df = df.sort_values("hot", ascending=False).head(top_n)
                # 确保列名统一
                if "pct_change" in df.columns:
                    df.rename(columns={"pct_change": "pct_chg"}, inplace=True)

            return df if df is not None else pd.DataFrame()
        except Exception as e:
            log.warning(f"获取同花顺热榜失败 ({market}): {e}")
            return pd.DataFrame()

    def get_top_list(self, trade_date: str, ts_code: str = None) -> pd.DataFrame:
        """
        获取龙虎榜每日明细 (top_list)
        积分≥2000
        """
        self.rate_limiter.wait_if_needed()
        try:
            params = {"trade_date": self.format_date(trade_date)}
            if ts_code:
                params["ts_code"] = self.format_stock_code(ts_code)
            df = self.pro.top_list(**params)
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            log.debug(f"获取龙虎榜失败: {e}")
            return pd.DataFrame()

    def get_top_inst(self, trade_date: str, ts_code: str = None) -> pd.DataFrame:
        """
        获取龙虎榜机构成交明细 (top_inst)
        积分≥5000
        """
        self.rate_limiter.wait_if_needed()
        try:
            params = {"trade_date": self.format_date(trade_date)}
            if ts_code:
                params["ts_code"] = self.format_stock_code(ts_code)
            df = self.pro.top_inst(**params)
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            log.debug(f"获取龙虎榜机构明细失败: {e}")
            return pd.DataFrame()

    def get_moneyflow(
        self,
        ts_code: str = None,
        trade_date: str = None,
        start_date: str = None,
        end_date: str = None,
    ) -> pd.DataFrame:
        """
        获取个股资金流向数据（主力净流入等）

        接口说明：https://tushare.pro/document/2?doc_id=170
        用户积分≥2000可调取

        Args:
            ts_code: 股票代码，若为None则拉取当日全市场
            trade_date: 交易日期 (YYYYMMDD)，与 start_date/end_date 互斥
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)

        Returns:
            DataFrame 含 ts_code, trade_date,
            buy_elg_amount(特大单买入), sell_elg_amount(特大单卖出),
            buy_lg_amount(大单买入), sell_lg_amount(大单卖出),
            buy_md_amount(中单买入), sell_md_amount(中单卖出),
            buy_sm_amount(小单买入), sell_sm_amount(小单卖出),
            net_mf_amount(净流入额万元) 等
        """
        self.rate_limiter.wait_if_needed()
        try:
            params = {}
            if trade_date:
                params["trade_date"] = self.format_date(trade_date)
            else:
                if start_date:
                    params["start_date"] = self.format_date(start_date)
                if end_date:
                    params["end_date"] = self.format_date(end_date)
            if ts_code:
                params["ts_code"] = self.format_stock_code(ts_code)

            df = self.pro.moneyflow(**params)
            if df is not None and not df.empty:
                if "trade_date" in df.columns:
                    df["trade_date"] = pd.to_datetime(df["trade_date"])
                    df = df.sort_values("trade_date").reset_index(drop=True)
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            log.warning(f"获取资金流向失败: {e}")
            return pd.DataFrame()

    def get_north_moneyflow(self, trade_date: str) -> pd.DataFrame:
        """
        获取沪深港通资金流向 (moneyflow_hsgt)
        积分≥2000

        Returns:
            DataFrame with columns: trade_date, ggt_ss, ggt_sz, hgt, sgt,
            north_money, south_money
        """
        self.rate_limiter.wait_if_needed()
        try:
            df = self.pro.moneyflow_hsgt(trade_date=self.format_date(trade_date))
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            log.debug(f"获取北向资金失败: {e}")
            return pd.DataFrame()

    # ─── ETF / Fund Methods ───

    def get_etf_list(self, market: str = "E", status: str = "L") -> pd.DataFrame:
        """
        获取场内 ETF 基金基础信息 (fund_basic)
        积分≥2000

        Args:
            market: 交易市场 ('E'=场内, 'O'=场外)
            status: 上市状态 ('L'=上市, 'D'=退市, 'P'=募集)

        Returns:
            DataFrame with columns:
                ts_code, name, management, custodian, fund_type, status,
                invest_type, type, benchmark, issue_date, delist_date,
                list_date, issue_amount, m_fee, c_fee, first_amount,
                last_amount, year_yld, total_nav, adj_nav, update_date 等
        """
        self.rate_limiter.wait_if_needed()
        try:
            df = self.pro.fund_basic(market=market, status=status)
            if df is not None and not df.empty:
                log.info(f"获取ETF列表成功: {len(df)} 只")
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            log.warning(f"获取ETF列表失败: {e}")
            return pd.DataFrame()

    def get_etf_daily(
        self,
        ts_code: str = None,
        trade_date: str = None,
        start_date: str = None,
        end_date: str = None,
    ) -> pd.DataFrame:
        """
        获取 ETF 日线行情 (fund_daily)
        积分≥5000

        Args:
            ts_code: ETF代码，如 '510330.SH'
            trade_date: 交易日期 (YYYYMMDD)
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)

        Returns:
            DataFrame with columns:
                ts_code, trade_date, open, high, low, close,
                pre_close, change, pct_chg, vol, amount
        """
        self.rate_limiter.wait_if_needed()
        try:
            params = {}
            if ts_code:
                params["ts_code"] = ts_code
            if trade_date:
                params["trade_date"] = self.format_date(trade_date)
            if start_date:
                params["start_date"] = self.format_date(start_date)
            if end_date:
                params["end_date"] = self.format_date(end_date)

            df = self.pro.fund_daily(**params)
            if df is not None and not df.empty:
                if "trade_date" in df.columns:
                    df["trade_date"] = pd.to_datetime(df["trade_date"])
                    df = df.sort_values("trade_date").reset_index(drop=True)
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            log.warning(f"获取ETF日线失败 ({ts_code}): {e}")
            return pd.DataFrame()

    def get_etf_nav(
        self,
        ts_code: str = None,
        trade_date: str = None,
        start_date: str = None,
        end_date: str = None,
    ) -> pd.DataFrame:
        """
        获取 ETF 基金净值 (fund_nav)
        积分≥2000

        Args:
            ts_code: ETF代码
            trade_date: 交易日期 (YYYYMMDD)
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            DataFrame with columns:
                ts_code, ann_date, nav_date, unit_nav, accum_nav, div_nav,
                net_asset, total_asset, adj_nav, update_flag
        """
        self.rate_limiter.wait_if_needed()
        try:
            params = {}
            if ts_code:
                params["ts_code"] = ts_code
            if trade_date:
                params["nav_date"] = self.format_date(trade_date)
            if start_date:
                params["start_date"] = self.format_date(start_date)
            if end_date:
                params["end_date"] = self.format_date(end_date)

            df = self.pro.fund_nav(**params)
            if df is not None and not df.empty:
                if "nav_date" in df.columns:
                    df["nav_date"] = pd.to_datetime(df["nav_date"])
                    df = df.sort_values("nav_date").reset_index(drop=True)
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            log.warning(f"获取ETF净值失败 ({ts_code}): {e}")
            return pd.DataFrame()

    def get_etf_share(
        self,
        ts_code: str = None,
        trade_date: str = None,
        start_date: str = None,
        end_date: str = None,
    ) -> pd.DataFrame:
        """
        获取 ETF 基金份额 (fund_share)
        积分≥2000

        Args:
            ts_code: ETF代码
            trade_date: 交易日期 (YYYYMMDD)
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            DataFrame with columns:
                ts_code, trade_date, fd_share, fd_share_change
        """
        self.rate_limiter.wait_if_needed()
        try:
            params = {}
            if ts_code:
                params["ts_code"] = ts_code
            if trade_date:
                params["trade_date"] = self.format_date(trade_date)
            if start_date:
                params["start_date"] = self.format_date(start_date)
            if end_date:
                params["end_date"] = self.format_date(end_date)

            df = self.pro.fund_share(**params)
            if df is not None and not df.empty:
                if "trade_date" in df.columns:
                    df["trade_date"] = pd.to_datetime(df["trade_date"])
                    df = df.sort_values("trade_date").reset_index(drop=True)
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            log.warning(f"获取ETF份额失败 ({ts_code}): {e}")
            return pd.DataFrame()
