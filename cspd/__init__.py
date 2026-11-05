"""工况策略元检测第一版。

空间原型检测头 + 工况 FiLM + 策略门控。
图像扩散和 PPO 不在这一版里。
"""

from .model import ConditionStrategyDetector

__all__ = ['ConditionStrategyDetector']
