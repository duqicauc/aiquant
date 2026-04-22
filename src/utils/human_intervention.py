"""
人工介入提醒工具

在需要人工决策的关键环节，提供明确的提醒和检查机制
"""

import json
from pathlib import Path
from typing import Dict, Optional

from src.utils.logger import log


class HumanInterventionChecker:
    """人工介入检查器"""

    def __init__(self):
        self.interventions = []

    def check_positive_sample_criteria(self, config_path: str = "config/settings.yaml") -> Dict:
        """
        检查正样本筛选条件是否需要人工确认

        Returns:
            检查结果字典
        """
        from config.settings import settings

        criteria = settings.get("data.sample_preparation.positive_criteria", {})

        # 检查关键阈值是否使用默认值
        default_values = {
            "consecutive_weeks": 3,
            "total_return_threshold": 50,
            "max_return_threshold": 70,
            "min_listing_days": 180,
        }

        warnings = []
        suggestions = []

        for key, default_value in default_values.items():
            current_value = criteria.get(key, default_value)
            if current_value == default_value:
                warnings.append(f"⚠️  使用默认值: {key} = {current_value}。" f"请确认是否符合当前需求。")

        # 检查日期范围
        start_date = settings.get("data.sample_preparation.start_date", "20000101")
        settings.get("data.sample_preparation.end_date", None)

        if start_date == "20000101":
            suggestions.append("💡 数据起始日期为2000-01-01，请确认是否需要调整。")

        return {
            "needs_intervention": len(warnings) > 0,
            "warnings": warnings,
            "suggestions": suggestions,
            "criteria": criteria,
        }

    def check_feature_selection(self, feature_extraction_file: str = None) -> Dict:
        """
        检查特征选择是否需要人工确认

        Returns:
            检查结果字典
        """
        warnings = []
        suggestions = []

        # 检查是否使用了基础特征

        suggestions.append(
            "💡 当前使用基础特征集。考虑添加：\n"
            "  - 基本面特征（PE、PB、ROE等）\n"
            "  - 其他技术指标（KDJ、OBV、布林带等）\n"
            "  - 行业特征\n"
            "  - 市场情绪特征"
        )

        return {"needs_intervention": False, "warnings": warnings, "suggestions": suggestions}  # 特征选择是持续优化过程

    def check_model_config(self, model_name: str, config_path: str = None) -> Dict:
        """
        检查模型配置是否需要人工确认

        Args:
            model_name: 模型名称
            config_path: 配置文件路径

        Returns:
            检查结果字典
        """
        if config_path is None:
            config_path = f"config/models/{model_name}.yaml"

        config_file = Path(config_path)
        if not config_file.exists():
            return {
                "needs_intervention": True,
                "warnings": [f"⚠️  配置文件不存在: {config_path}。请先创建配置文件。"],
                "suggestions": [],
            }

        import yaml

        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        warnings = []
        suggestions = []

        # 检查是否使用默认超参数
        model_params = config.get("model_params", {})
        default_params = {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1}

        for key, default_value in default_params.items():
            current_value = model_params.get(key, default_value)
            if current_value == default_value:
                warnings.append(f"⚠️  使用默认超参数: {key} = {current_value}。" f"建议根据数据特点进行调优。")

        # 检查算法类型
        model_type = config.get("model", {}).get("type", "xgboost")
        if model_type == "xgboost":
            suggestions.append(
                "💡 当前使用XGBoost。考虑尝试：\n"
                "  - LightGBM（速度更快）\n"
                "  - CatBoost（处理类别特征）\n"
                "  - 集成模型（多模型融合）"
            )

        return {
            "needs_intervention": len(warnings) > 0,
            "warnings": warnings,
            "suggestions": suggestions,
            "config": config,
        }

    def check_training_results(self, model_name: str, version: str) -> Dict:
        """
        检查训练结果是否需要人工介入

        Args:
            model_name: 模型名称
            version: 版本号

        Returns:
            检查结果字典
        """
        try:
            # 直接读取模型目录中的指标文件
            from pathlib import Path

            model_dir = Path("data/models") / model_name / "versions" / version
            metrics_file = model_dir / "training" / "metrics.json"

            if not metrics_file.exists():
                raise FileNotFoundError(f"指标文件不存在: {metrics_file}")

            with open(metrics_file, "r") as f:
                metrics_data = json.load(f)

            metrics = metrics_data if isinstance(metrics_data, dict) else {}
            test_metrics = metrics.get("test", metrics)

            warnings = []
            suggestions = []

            # 检查指标是否达标
            auc = test_metrics.get("auc", 0)
            accuracy = test_metrics.get("accuracy", 0)
            f1 = test_metrics.get("f1", 0)

            if auc < 0.7:
                warnings.append(f"⚠️  AUC = {auc:.3f} < 0.7，模型性能可能不佳。" f"建议检查特征选择或调整超参数。")

            if accuracy < 0.75:
                warnings.append(
                    f"⚠️  准确率 = {accuracy:.3f} < 0.75，模型性能可能不佳。" f"建议检查数据质量或调整模型。"
                )

            if f1 < 0.7:
                warnings.append(f"⚠️  F1分数 = {f1:.3f} < 0.7，模型可能存在过拟合或欠拟合。" f"建议调整模型参数。")

            # 检查是否过拟合
            train_metrics = metrics.get("training", {})
            train_accuracy = train_metrics.get("accuracy", 0)
            if train_accuracy - accuracy > 0.15:
                warnings.append(
                    f"⚠️  训练准确率({train_accuracy:.3f})与测试准确率({accuracy:.3f})差距较大，"
                    f"可能存在过拟合。建议增加正则化或减少模型复杂度。"
                )

            return {
                "needs_intervention": len(warnings) > 0,
                "warnings": warnings,
                "suggestions": suggestions,
                "metrics": test_metrics,
            }
        except Exception as e:
            return {"needs_intervention": True, "warnings": [f"⚠️  无法获取训练结果: {e}"], "suggestions": []}

    def check_version_comparison(self, model_name: str, old_version: str, new_version: str) -> Dict:
        """
        检查版本对比结果，提醒是否需要人工决策

        Args:
            model_name: 模型名称
            old_version: 旧版本号
            new_version: 新版本号

        Returns:
            检查结果字典
        """
        try:
            # 直接读取模型目录中的指标文件
            from pathlib import Path

            def _load_metrics(version):
                model_dir = Path("data/models") / model_name / "versions" / version
                metrics_file = model_dir / "training" / "metrics.json"
                if not metrics_file.exists():
                    raise FileNotFoundError(f"指标文件不存在: {metrics_file}")
                with open(metrics_file, "r") as f:
                    return json.load(f)

            old_metrics_data = _load_metrics(old_version)
            new_metrics_data = _load_metrics(new_version)

            old_metrics = old_metrics_data.get("test", old_metrics_data) if isinstance(old_metrics_data, dict) else {}
            new_metrics = new_metrics_data.get("test", new_metrics_data) if isinstance(new_metrics_data, dict) else {}

            warnings = []
            suggestions = []

            # 对比关键指标
            old_auc = old_metrics.get("auc", 0)
            new_auc = new_metrics.get("auc", 0)

            old_accuracy = old_metrics.get("accuracy", 0)
            new_accuracy = new_metrics.get("accuracy", 0)

            if new_auc < old_auc:
                warnings.append(
                    f"⚠️  新版本AUC({new_auc:.3f}) < 旧版本AUC({old_auc:.3f})，" f"性能下降。建议回滚或继续优化。"
                )

            if new_accuracy < old_accuracy:
                warnings.append(
                    f"⚠️  新版本准确率({new_accuracy:.3f}) < 旧版本准确率({old_accuracy:.3f})，"
                    f"性能下降。建议回滚或继续优化。"
                )

            if new_auc > old_auc and new_accuracy > old_accuracy:
                suggestions.append(
                    f"✅ 新版本全面优于旧版本（AUC: {old_auc:.3f} → {new_auc:.3f}, "
                    f"准确率: {old_accuracy:.3f} → {new_accuracy:.3f}）。"
                    f"建议升级到新版本。"
                )

            return {
                "needs_intervention": True,  # 版本对比总是需要人工决策
                "warnings": warnings,
                "suggestions": suggestions,
                "comparison": {"old": old_metrics, "new": new_metrics},
            }
        except Exception as e:
            return {"needs_intervention": True, "warnings": [f"⚠️  无法进行版本对比: {e}"], "suggestions": []}

    def print_intervention_reminder(self, title: str, check_result: Dict):
        """
        打印人工介入提醒

        Args:
            title: 检查标题
            check_result: 检查结果
        """
        log.info("=" * 80)
        log.info(f"👤 人工介入检查: {title}")
        log.info("=" * 80)

        if check_result.get("needs_intervention", False):
            log.warning("⚠️  需要人工介入！")
        else:
            log.success("✓ 当前配置正常")

        warnings = check_result.get("warnings", [])
        if warnings:
            log.warning("\n警告:")
            for warning in warnings:
                log.warning(f"  {warning}")

        suggestions = check_result.get("suggestions", [])
        if suggestions:
            log.info("\n建议:")
            for suggestion in suggestions:
                log.info(f"  {suggestion}")

        log.info("=" * 80)

        return check_result.get("needs_intervention", False)


def require_human_confirmation(message: str, default: bool = False, timeout: Optional[int] = None) -> bool:
    """
    要求人工确认

    Args:
        message: 确认消息
        default: 默认值（如果超时）
        timeout: 超时时间（秒），None表示不超时

    Returns:
        用户确认结果
    """
    # 检查环境变量，如果设置了 AUTO_CONFIRM=1，则自动确认
    import os

    auto_confirm = os.environ.get("AUTO_CONFIRM", "0")
    if auto_confirm == "1":
        log.info("=" * 80)
        log.info("🤖 自动确认模式（AUTO_CONFIRM=1）")
        log.info("=" * 80)
        log.info(message)
        log.info("=" * 80)
        log.info(f"自动使用默认值: {default}")
        return default

    log.warning("=" * 80)
    log.warning("👤 需要人工确认")
    log.warning("=" * 80)
    log.warning(message)
    log.warning("=" * 80)

    if timeout:
        log.warning(f"⏰ 超时时间: {timeout}秒，超时将使用默认值: {default}")

    try:
        response = input(f"\n请确认 (y/n，默认: {'y' if default else 'n'}): ").strip().lower()

        if not response:
            return default

        return response in ["y", "yes", "是", "确认"]
    except (KeyboardInterrupt, EOFError):
        log.warning(f"\n用户中断，使用默认值: {default}")
        return default


def prompt_human_input(
    message: str, input_type: type = str, default: Optional[any] = None, validator: Optional[callable] = None
) -> any:
    """
    提示人工输入

    Args:
        message: 提示消息
        input_type: 输入类型
        default: 默认值
        validator: 验证函数

    Returns:
        用户输入值
    """
    log.info("=" * 80)
    log.info("👤 需要人工输入")
    log.info("=" * 80)
    log.info(message)

    if default is not None:
        log.info(f"默认值: {default}")

    log.info("=" * 80)

    while True:
        try:
            user_input = input("\n请输入: ").strip()

            if not user_input and default is not None:
                return default

            if not user_input:
                log.warning("输入不能为空，请重新输入")
                continue

            # 类型转换
            try:
                converted_value = input_type(user_input)
            except ValueError:
                log.warning(f"输入格式错误，期望类型: {input_type.__name__}")
                continue

            # 验证
            if validator and not validator(converted_value):
                log.warning("输入验证失败，请重新输入")
                continue

            return converted_value

        except (KeyboardInterrupt, EOFError):
            if default is not None:
                log.warning(f"\n用户中断，使用默认值: {default}")
                return default
            raise
