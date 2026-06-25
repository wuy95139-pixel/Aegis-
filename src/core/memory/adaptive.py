"""
自适应行为引擎
==============
在行动前综合历史经验、用户偏好和反馈规则，生成行动指导；
在行动后记录结果，形成持续改进的学习闭环。

核心循环：
  ┌──────────────────────────────────────────────┐
  │                                              │
  │  1. before_action(situation)                 │
  │     ├─ 查相似经验 (ExperienceLibrary)        │
  │     ├─ 查用户偏好 (PreferenceLearner)         │
  │     ├─ 查反馈规则 (MemoryType.FEEDBACK)       │
  │     └─ 生成 AdaptiveGuidance                  │
  │                                              │
  │  2. Agent 执行动作 (根据指导调整行为)          │
  │                                              │
  │  3. after_action(situation, approach, outcome)│
  │     ├─ 记录经验 (ExperienceLibrary.record)    │
  │     ├─ 更新偏好置信度 (PreferenceLearner)      │
  │     └─ 更新规则有效性 (强化或弱化)             │
  │                                              │
  └──────────────────────────────────────────────┘

输出形式：
  - 文本指导（注入 LLM 系统提示）
  - 结构化指导（供 Agent 代码逻辑使用）
  - 行动前检查清单
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

from src.core.memory.experience import ExperienceLibrary, Outcome, ExperienceCase
from src.core.memory.preference import PreferenceLearner, PreferenceSignal

logger = logging.getLogger(__name__)


class GuidanceLevel(str, Enum):
    """指导的强度级别"""
    MUST = "must"         # 必须遵守（高置信度强规则）
    SHOULD = "should"     # 应该遵守（中等置信度）
    MAY = "may"           # 可以参考（低置信度弱信号）
    AVOID = "avoid"       # 必须避免（高置信度失败教训）


class AdaptiveGuidance:
    """
    自适应行动指导

    综合经验 + 偏好 + 规则后的行动建议
    """

    def __init__(self):
        self.items: List[Dict[str, Any]] = []
        self.summary: str = ""
        self.confidence: float = 0.0
        self.sources: List[str] = []          # 指导来源

    def add(self, level: GuidanceLevel, instruction: str, because: str, source: str = ""):
        """添加一条指导"""
        self.items.append({
            "level": level.value,
            "instruction": instruction,
            "because": because,
            "source": source,
        })
        if source and source not in self.sources:
            self.sources.append(source)

    def add_must(self, instruction: str, because: str, source: str = ""):
        """添加强制指导"""
        self.add(GuidanceLevel.MUST, instruction, because, source)

    def add_should(self, instruction: str, because: str, source: str = ""):
        """添加建议指导"""
        self.add(GuidanceLevel.SHOULD, instruction, because, source)

    def add_may(self, instruction: str, because: str, source: str = ""):
        """添加参考指导"""
        self.add(GuidanceLevel.MAY, instruction, because, source)

    def add_avoid(self, instruction: str, because: str, source: str = ""):
        """添加避免指导"""
        self.add(GuidanceLevel.AVOID, instruction, because, source)

    def is_empty(self) -> bool:
        return len(self.items) == 0

    def to_text(self) -> str:
        """转为可注入 LLM 上下文的文本"""
        if not self.items:
            return ""

        parts = ["## 自适应行为指导 (Adaptive Guidance)"]
        parts.append(f"*基于 {len(self.sources)} 个来源的历史经验生成*\n")

        level_icons = {
            "must": "🔴 必须",
            "avoid": "⛔ 避免",
            "should": "🟡 建议",
            "may": "🟢 参考",
        }

        by_level = {"must": [], "avoid": [], "should": [], "may": []}
        for item in self.items:
            by_level[item["level"]].append(item)

        for level in ["must", "avoid", "should", "may"]:
            items = by_level[level]
            if not items:
                continue
            icon = level_icons.get(level, f"[{level}]")
            parts.append(f"### {icon}")
            for item in items:
                parts.append(f"- {item['instruction']}")
                parts.append(f"  *原因: {item['because']}*")

        if self.summary:
            parts.append(f"\n### 总结\n{self.summary}")

        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "items": self.items,
            "summary": self.summary,
            "confidence": self.confidence,
            "sources": self.sources,
        }


class AdaptiveEngine:
    """
    自适应行为引擎 — 学习闭环的核心

    使用示例:
        engine = AdaptiveEngine(experience_lib, preference_learner, memory_manager)

        # 行动前
        guidance = engine.before_action(
            situation="用户要求修改核心认证逻辑",
            action_type="code_refactor",
        )
        print(guidance.to_text())
        # → "根据历史经验，建议先生成测试再重构，避免破坏向后兼容..."

        # 行动后
        engine.after_action(
            situation="用户要求修改核心认证逻辑",
            approach="先用接口抽象隔离，再逐步替换旧实现",
            outcome=Outcome.SUCCESS,
            user_feedback="很好，这次改动很稳妥",
        )
        # → 自动记录成功经验，强化偏好
    """

    def __init__(
        self,
        experience_library: Optional[ExperienceLibrary] = None,
        preference_learner: Optional[PreferenceLearner] = None,
        memory_manager=None,
        llm=None,
    ):
        """
        Args:
            experience_library: 经验库
            preference_learner: 偏好学习器
            memory_manager: MemoryManager 实例（用于访问反馈规则等）
            llm: LLMProvider 实例
        """
        self.experiences = experience_library
        self.preferences = preference_learner
        self.memory = memory_manager
        self.llm = llm

        # 统计
        self._action_count = 0
        self._guidance_history: List[Dict[str, Any]] = []

    # ==================== 行动前：生成指导 ====================

    def before_action(
        self,
        situation: str,
        action_type: str = "general",
        context: Optional[Dict[str, Any]] = None,
    ) -> AdaptiveGuidance:
        """
        在行动前生成自适应指导

        综合三个来源：
          1. 经验库 — 类似情况下的成功/失败经验
          2. 偏好学习器 — 用户的编码/沟通等偏好
          3. 反馈规则 — 用户明确设定的行为规则

        Args:
            situation: 当前情况描述
            action_type: 行动类型 (code_refactor / code_generate / explain / search / file_ops / general)
            context: 额外上下文信息

        Returns:
            AdaptiveGuidance 指导对象
        """
        guidance = AdaptiveGuidance()

        # === 来源 1: 经验库 ===
        if self.experiences:
            exp_guidance = self.experiences.get_guidance(situation)

            if exp_guidance.get("has_relevant_experience"):
                confidence = exp_guidance.get("confidence", 0)

                for rec in exp_guidance.get("recommend", []):
                    level = GuidanceLevel.MUST if confidence > 0.7 else GuidanceLevel.SHOULD
                    because = exp_guidance.get("because", [""])[
                        min(
                            exp_guidance["recommend"].index(rec),
                            len(exp_guidance.get("because", [])) - 1,
                        )
                    ] if exp_guidance.get("because") else ""
                    guidance.add(level, rec, because, "experience_library")

                for av in exp_guidance.get("avoid", []):
                    idx = len(exp_guidance.get("recommend", []))
                    avoid_idx = exp_guidance["avoid"].index(av)
                    because = exp_guidance.get("because", [""])[
                        min(idx + avoid_idx, len(exp_guidance.get("because", [])) - 1)
                    ] if exp_guidance.get("because") else ""
                    guidance.add(GuidanceLevel.AVOID, f"不要: {av}", because, "experience_library")

        # === 来源 2: 偏好学习器 ===
        if self.preferences:
            active_prefs = self.preferences.get_active_preferences(min_confidence=0.3)

            # 根据行动类型筛选相关偏好
            relevant_prefs = self._filter_prefs_by_action(active_prefs, action_type)

            for pref in relevant_prefs[:5]:
                if pref.is_strong():
                    guidance.add_should(
                        pref.preference,
                        f"用户强偏好 (置信度: {pref.confidence:.0%}, 观察 {pref.occurrences} 次)",
                        "preference_learner",
                    )
                elif pref.is_active():
                    guidance.add_may(
                        pref.preference,
                        f"用户偏好 (置信度: {pref.confidence:.0%})",
                        "preference_learner",
                    )

        # === 来源 3: 反馈规则 ===
        if self.memory:
            rules = self.memory.get_feedback_rules()
            for rule in rules:
                severity = rule.get("severity", "medium")
                level = GuidanceLevel.MUST if severity == "high" else GuidanceLevel.SHOULD

                guidance.add(
                    level,
                    rule.get("rule", ""),
                    rule.get("why", "用户反馈"),
                    f"feedback_rule:{rule.get('name', 'unknown')}",
                )

        # === LLM 综合 ===
        if self.llm and not guidance.is_empty():
            guidance.summary = self._synthesize_guidance(situation, guidance)

        guidance.confidence = self._calculate_confidence(guidance)
        self._guidance_history.append({
            "situation": situation,
            "action_type": action_type,
            "guidance_items": len(guidance.items),
            "timestamp": datetime.now().isoformat(),
        })
        self._action_count += 1

        return guidance

    def before_action_text(self, situation: str, action_type: str = "general") -> str:
        """获取行动前指导文本（用于注入 LLM 系统提示）"""
        guidance = self.before_action(situation, action_type)
        return guidance.to_text()

    # ==================== 行动后：记录结果 ====================

    def after_action(
        self,
        situation: str,
        approach: str,
        outcome: Outcome,
        user_feedback: str = "",
        context_tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        行动后记录结果，完成学习闭环

        这会:
          1. 将本次行动作为经验记录到经验库
          2. 根据用户反馈更新偏好置信度
          3. 如果用户反馈包含新的偏好信号，触发检测

        Args:
            situation: 行动时的情况
            approach: 采取的方法
            outcome: 结果
            user_feedback: 用户的反馈（如有）
            context_tags: 上下文标签

        Returns:
            学习结果汇总
        """
        result = {
            "experience_recorded": False,
            "preferences_updated": [],
            "lessons_learned": [],
        }

        # 1. 记录经验
        if self.experiences:
            case = self.experiences.record(
                situation=situation,
                approach=approach,
                outcome=outcome,
                context_tags=context_tags,
                source="adaptive_engine",
            )
            result["experience_recorded"] = True
            result["case_id"] = case.case_id
            result["lessons_learned"].append(case.lesson)

        # 2. 根据用户反馈更新偏好
        if user_feedback and self.preferences:
            # 正面反馈 → 强化相关偏好
            if self._is_positive_feedback(user_feedback):
                active_prefs = self.preferences.get_active_preferences(min_confidence=0.2)
                for pref in active_prefs:
                    if self._is_related(pref.preference, approach):
                        pref.reinforce(
                            context=situation,
                            evidence=f"用户正面反馈: {user_feedback[:150]}",
                        )
                        result["preferences_updated"].append({
                            "signal": pref.preference[:80],
                            "action": "reinforced",
                            "new_confidence": pref.confidence,
                        })

            # 负面反馈 → 弱化相关偏好 + 检测新偏好信号
            elif self._is_negative_feedback(user_feedback):
                signals = self.preferences.detect_signals(
                    user_message=user_feedback,
                    context=situation,
                    source="after_action_feedback",
                )
                result["preferences_updated"] = [
                    {"signal": s.preference[:80], "action": "detected", "new_confidence": s.confidence}
                    for s in signals
                ]

        logger.info(
            f"After action recorded: [{outcome.value}] approach={approach[:60]}..."
        )
        return result

    def learn_from_interaction(
        self,
        user_message: str,
        assistant_response: str,
        user_reaction: str = "",
        context: str = "",
    ) -> Dict[str, Any]:
        """
        从一次完整交互中学习（便捷方法）

        等价于：检测偏好 + 隐式记录经验
        """
        result = {
            "preferences_learned": [],
            "experience_recorded": False,
        }

        # 1. 偏好学习
        if self.preferences:
            obs = self.preferences.observe_interaction(
                user_message=user_message,
                assistant_response=assistant_response,
                user_reaction=user_reaction,
                context=context,
            )
            result["preferences_learned"] = obs

        # 2. 如果有用户反应，记录经验
        if user_reaction:
            outcome = Outcome.SUCCESS if self._is_positive_feedback(user_reaction) else (
                Outcome.FAILURE if self._is_negative_feedback(user_reaction) else Outcome.PARTIAL
            )

            result["experience_recorded"] = self.after_action(
                situation=context or user_message[:200],
                approach=assistant_response[:200],
                outcome=outcome,
                user_feedback=user_reaction,
            )

        return result

    # ==================== 便捷查询 ====================

    def get_behavioral_prompt(self, situation: str = "", action_type: str = "general") -> str:
        """
        获取完整的行为指导提示（用于注入 LLM 系统提示）

        生成一个综合性的提示，包含：
          - 经验指导
          - 用户偏好
          - 应避免的事项
        """
        parts = []

        # 1. 经验指导
        if self.experiences:
            exp_text = self.experiences.get_guidance_text(situation)
            if exp_text:
                parts.append(exp_text)

        # 2. 偏好提示
        if self.preferences:
            pref_text = self.preferences.get_preference_prompt()
            if pref_text:
                parts.append(pref_text)

        # 3. 行动指导
        guidance = self.before_action(situation, action_type)
        guidance_text = guidance.to_text()
        if guidance_text:
            parts.append(guidance_text)

        return "\n\n".join(parts)

    def get_learning_summary(self) -> Dict[str, Any]:
        """获取学习总结"""
        return {
            "total_actions_analyzed": self._action_count,
            "experiences": self.experiences.get_statistics() if self.experiences else {},
            "preferences": self.preferences.get_statistics() if self.preferences else {},
            "recent_guidance": self._guidance_history[-5:],
        }

    # ==================== 内部方法 ====================

    def _filter_prefs_by_action(
        self, prefs: List[PreferenceSignal], action_type: str
    ) -> List[PreferenceSignal]:
        """根据行动类型筛选相关偏好"""
        action_to_categories = {
            "code_refactor": ["coding_style", "architecture", "workflow"],
            "code_generate": ["coding_style", "architecture", "tool"],
            "explain": ["communication", "coding_style"],
            "search": ["tool", "architecture"],
            "file_ops": ["workflow", "tool"],
            "general": [],  # 全部
        }

        relevant_cats = action_to_categories.get(action_type, [])
        if not relevant_cats:
            return prefs

        return [p for p in prefs if p.category in relevant_cats]

    def _synthesize_guidance(self, situation: str, guidance: AdaptiveGuidance) -> str:
        """使用 LLM 综合生成指导摘要"""
        if not self.llm:
            return ""

        items_text = "\n".join(
            f"- [{item['level']}] {item['instruction']}"
            for item in guidance.items[:10]
        )

        prompt = f"""基于以下针对当前情况的行为指导，生成一个 2-3 句话的执行摘要：

当前情况：{situation[:300]}

行为指导：
{items_text}

摘要（聚焦于最重要的一两条建议）："""

        try:
            response = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200,
            )
            return response["content"].strip()
        except Exception:
            return ""

    def _calculate_confidence(self, guidance: AdaptiveGuidance) -> float:
        """计算指导的综合置信度"""
        if not guidance.items:
            return 0.0

        level_weights = {"must": 1.0, "avoid": 0.9, "should": 0.6, "may": 0.3}
        total_weight = sum(level_weights.get(item["level"], 0.3) for item in guidance.items)
        return min(0.95, total_weight / (len(guidance.items) * 2))

    def _is_positive_feedback(self, feedback: str) -> bool:
        """判断是否是正面反馈"""
        feedback_lower = feedback.lower()
        positive = ["好", "不错", "很好", "对", "是的", "谢谢", "good", "great", "perfect", "thanks", "exactly", "right"]
        return any(p in feedback_lower for p in positive) and len(feedback) < 200

    def _is_negative_feedback(self, feedback: str) -> bool:
        """判断是否是负面反馈"""
        feedback_lower = feedback.lower()
        negative = [
            "不对", "错了", "不要", "别", "不好", "不行", "太", "不够",
            "wrong", "no ", "don't", "bad", "too ", "not ",
        ]
        return any(n in feedback_lower for n in negative)

    def _is_related(self, preference: str, approach: str) -> bool:
        """判断偏好和方法是否相关"""
        # 简单的关键词重叠检测
        pref_words = set(preference.lower().split())
        approach_words = set(approach.lower().split())
        overlap = pref_words & approach_words
        return len(overlap) >= 2

    def __repr__(self):
        return f"<AdaptiveEngine(actions={self._action_count})>"
