#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型版本晋升控制器

职责：
1. 定义各环境晋升的硬性 Gates
2. 根据训练指标和回测结果自动建议晋升方向
3. 更新 current.json 中的版本指针
4. staging 始终作为 production 的镜像

Usage:
    from src.models.version_promoter import VersionPromoter
    promoter = VersionPromoter()
    decision = promoter.evaluate_training_metrics(metrics_dict)
    promoter.promote("v2.9.6-ensemble", "testing")  # 人工确认后晋升
"""

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from src.utils.logger import log

PROJECT_ROOT = Path(__file__).parent.parent.parent


@dataclass
class GateResult:
    """Gate 检查结果"""
    name: str
    passed: bool
    score: float = 0.0
    threshold: float = 0.0
    details: str = ""


@dataclass
class PromotionDecision:
    """晋升决策结果"""
    version: str
    current_stage: str  # development / testing / production
    recommended_stage: str
    can_promote: bool
    gate_results: List[GateResult] = field(default_factory=list)
    reason: str = ""


class VersionPromoter:
    """版本晋升控制器"""

    # ==================== Gates 配置 ====================

    TESTING_GATES = {
        "auc": {"min": 0.75, "weight": 1.0},
        "brier": {"max": 0.15, "weight": 1.0},  # 越低越好
        "f1": {"min": 0.50, "weight": 0.8},
        "mean_disagreement": {"max": 0.65, "weight": 0.5},
        "feature_count": {"min": 150, "weight": 0.3},
    }

    PRODUCTION_GATES = {
        "backtest_quarters": {"min": 2, "weight": 1.0},
        "return_improvement_pct": {"min": 5.0, "weight": 1.0},
        "win_rate_improvement_pct": {"min": 3.0, "weight": 0.8},
        "max_drawdown_limit_pct": {"max": 10.0, "weight": 0.8},
        "min_trades": {"min": 200, "weight": 0.5},
    }

    def __init__(self, current_file: Optional[Path] = None):
        self.current_file = current_file or (PROJECT_ROOT / "data" / "models" / "breakout_launch_scorer" / "current.json")
        self.current = self._load_current()

    def _load_current(self) -> dict:
        """加载 current.json"""
        if self.current_file.exists():
            try:
                return json.loads(self.current_file.read_text(encoding="utf-8"))
            except Exception as e:
                log.warning(f"读取 current.json 失败: {e}")
        return {
            "production": "v2.7.0",
            "staging": "v2.7.0",
            "testing": "v2.9.1-ensemble",
            "development": None,
            "updated_at": datetime.now().isoformat(),
        }

    def _save_current(self):
        """保存 current.json"""
        self.current_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.current_file, "w", encoding="utf-8") as f:
            json.dump(self.current, f, indent=2, ensure_ascii=False)

    # ==================== Gates 检查 ====================

    def check_testing_gates(self, metrics: dict) -> List[GateResult]:
        """检查 testing 晋升 Gates"""
        results = []

        # AUC
        auc = metrics.get("ensemble_auc", 0)
        results.append(GateResult(
            name="AUC", passed=auc >= self.TESTING_GATES["auc"]["min"],
            score=auc, threshold=self.TESTING_GATES["auc"]["min"],
            details=f"AUC={auc:.4f} (门槛={self.TESTING_GATES['auc']['min']})"
        ))

        # Brier
        brier = metrics.get("ensemble_brier", 1.0)
        results.append(GateResult(
            name="Brier", passed=brier <= self.TESTING_GATES["brier"]["max"],
            score=brier, threshold=self.TESTING_GATES["brier"]["max"],
            details=f"Brier={brier:.4f} (门槛<={self.TESTING_GATES['brier']['max']})"
        ))

        # F1
        f1 = metrics.get("ensemble_f1", 0)
        results.append(GateResult(
            name="F1", passed=f1 >= self.TESTING_GATES["f1"]["min"],
            score=f1, threshold=self.TESTING_GATES["f1"]["min"],
            details=f"F1={f1:.4f} (门槛={self.TESTING_GATES['f1']['min']})"
        ))

        # 分歧度
        disagree = metrics.get("mean_disagreement", 1.0)
        results.append(GateResult(
            name="分歧度", passed=disagree <= self.TESTING_GATES["mean_disagreement"]["max"],
            score=disagree, threshold=self.TESTING_GATES["mean_disagreement"]["max"],
            details=f"分歧度={disagree:.4f} (门槛<={self.TESTING_GATES['mean_disagreement']['max']})"
        ))

        # 特征数
        feat_count = metrics.get("feature_count", 0)
        results.append(GateResult(
            name="特征数", passed=feat_count >= self.TESTING_GATES["feature_count"]["min"],
            score=feat_count, threshold=self.TESTING_GATES["feature_count"]["min"],
            details=f"特征数={feat_count} (门槛={self.TESTING_GATES['feature_count']['min']})"
        ))

        return results

    def check_production_gates(self, backtest_results: dict) -> List[GateResult]:
        """检查 production 晋升 Gates（基于回测结果）"""
        results = []

        # 回测季度数
        quarters = backtest_results.get("quarters_tested", 0)
        results.append(GateResult(
            name="回测季度数", passed=quarters >= self.PRODUCTION_GATES["backtest_quarters"]["min"],
            score=quarters, threshold=self.PRODUCTION_GATES["backtest_quarters"]["min"],
            details=f"回测季度={quarters} (门槛={self.PRODUCTION_GATES['backtest_quarters']['min']})"
        ))

        # 收益提升
        ret_imp = backtest_results.get("return_improvement_pct", 0)
        results.append(GateResult(
            name="收益提升", passed=ret_imp >= self.PRODUCTION_GATES["return_improvement_pct"]["min"],
            score=ret_imp, threshold=self.PRODUCTION_GATES["return_improvement_pct"]["min"],
            details=f"收益提升={ret_imp:.1f}% (门槛={self.PRODUCTION_GATES['return_improvement_pct']['min']})"
        ))

        # 胜率提升
        wr_imp = backtest_results.get("win_rate_improvement_pct", 0)
        results.append(GateResult(
            name="胜率提升", passed=wr_imp >= self.PRODUCTION_GATES["win_rate_improvement_pct"]["min"],
            score=wr_imp, threshold=self.PRODUCTION_GATES["win_rate_improvement_pct"]["min"],
            details=f"胜率提升={wr_imp:.1f}% (门槛={self.PRODUCTION_GATES['win_rate_improvement_pct']['min']})"
        ))

        # 最大回撤
        mdd = backtest_results.get("max_drawdown", 100)
        results.append(GateResult(
            name="最大回撤", passed=mdd <= self.PRODUCTION_GATES["max_drawdown_limit_pct"]["max"],
            score=mdd, threshold=self.PRODUCTION_GATES["max_drawdown_limit_pct"]["max"],
            details=f"最大回撤={mdd:.1f}% (门槛<={self.PRODUCTION_GATES['max_drawdown_limit_pct']['max']})"
        ))

        # 交易次数
        trades = backtest_results.get("total_trades", 0)
        results.append(GateResult(
            name="交易次数", passed=trades >= self.PRODUCTION_GATES["min_trades"]["min"],
            score=trades, threshold=self.PRODUCTION_GATES["min_trades"]["min"],
            details=f"交易次数={trades} (门槛={self.PRODUCTION_GATES['min_trades']['min']})"
        ))

        return results

    # ==================== 晋升决策 ====================

    def evaluate_training_metrics(self, version: str, metrics: dict) -> PromotionDecision:
        """根据训练指标评估是否能晋升到 testing"""
        gates = self.check_testing_gates(metrics)
        all_passed = all(g.passed for g in gates)

        if all_passed:
            recommended = "testing"
            reason = "训练指标全部达标，建议晋升到 testing"
        else:
            failed = [g.name for g in gates if not g.passed]
            recommended = "development"
            reason = f"训练指标未达标: {', '.join(failed)}，留在 development"

        return PromotionDecision(
            version=version,
            current_stage="development",
            recommended_stage=recommended,
            can_promote=all_passed,
            gate_results=gates,
            reason=reason,
        )

    def evaluate_backtest(self, version: str, backtest_results: dict) -> PromotionDecision:
        """根据回测结果评估是否能晋升到 production"""
        gates = self.check_production_gates(backtest_results)
        all_passed = all(g.passed for g in gates)

        if all_passed:
            recommended = "production"
            reason = "回测全部达标，建议晋升到 production（需人工确认）"
        else:
            failed = [g.name for g in gates if not g.passed]
            recommended = "testing"
            reason = f"回测未达标: {', '.join(failed)}，留在 testing"

        return PromotionDecision(
            version=version,
            current_stage="testing",
            recommended_stage=recommended,
            can_promote=all_passed,
            gate_results=gates,
            reason=reason,
        )

    # ==================== 版本操作 ====================

    def update_development(self, version: str, metrics: dict):
        """更新 development 版本"""
        self.current["development"] = version
        self.current["latest_train"] = {
            "version": version,
            "ensemble_auc": metrics.get("ensemble_auc"),
            "ensemble_f1": metrics.get("ensemble_f1"),
            "ensemble_brier": metrics.get("ensemble_brier"),
            "mean_disagreement": metrics.get("mean_disagreement"),
            "feature_count": metrics.get("feature_count"),
            "train_samples": metrics.get("train_samples"),
            "test_samples": metrics.get("test_samples"),
            "updated_at": datetime.now().isoformat(),
        }
        self.current["updated_at"] = datetime.now().isoformat()
        self._save_current()
        log.info(f"development → {version}")

    def promote(self, version: str, target_stage: str, force: bool = False) -> bool:
        """
        晋升版本到目标阶段

        Args:
            version: 版本号
            target_stage: 目标阶段 (testing / production)
            force: 是否跳过 gates 检查（仅用于紧急回滚）
        """
        if target_stage not in ("testing", "production"):
            log.error(f"不支持晋升到 {target_stage}")
            return False

        # 检查版本目录是否存在
        model_dir = PROJECT_ROOT / "data" / "models" / "breakout_launch_scorer" / "versions" / version / "model"
        if not model_dir.exists():
            log.error(f"版本目录不存在: {model_dir}")
            return False

        # 备份旧版本
        old_version = self.current.get(target_stage)
        if old_version and target_stage == "production":
            backup_dir = PROJECT_ROOT / "data" / "models_backup" / f"{old_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            old_model_dir = PROJECT_ROOT / "data" / "models" / "breakout_launch_scorer" / "versions" / old_version / "model"
            if old_model_dir.exists():
                backup_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(old_model_dir, backup_dir)
                log.info(f"旧版本已备份: {backup_dir}")

        # 更新 current.json
        self.current[target_stage] = version
        self.current["updated_at"] = datetime.now().isoformat()

        # staging 始终同步 production
        if target_stage == "production":
            self.current["staging"] = version
            log.info(f"staging 同步为 {version}（production 镜像）")

        self._save_current()
        log.success(f"✓ {version} 已晋升到 {target_stage}")
        return True

    def rollback(self, target_stage: str = "production") -> bool:
        """回滚到上一个版本（从 staging 恢复）"""
        if target_stage != "production":
            log.error("仅支持 production 回滚")
            return False

        staging_version = self.current.get("staging")
        production_version = self.current.get("production")

        if staging_version == production_version:
            log.warning("staging 与 production 相同，无法回滚")
            return False

        self.current["production"] = staging_version
        self.current["updated_at"] = datetime.now().isoformat()
        self._save_current()
        log.success(f"✓ production 已回滚到 {staging_version}")
        return True

    def get_status(self) -> dict:
        """获取当前版本状态"""
        return {
            "production": self.current.get("production"),
            "staging": self.current.get("staging"),
            "testing": self.current.get("testing"),
            "development": self.current.get("development"),
            "latest_train": self.current.get("latest_train"),
        }
