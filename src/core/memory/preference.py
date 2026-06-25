"""
偏好学习器
==========
从用户交互中检测、强化、管理用户偏好。

两种偏好来源：
  1. 显式偏好 (explicit): 用户直接说出的偏好
     "我不喜欢用 ORM"、"用 async/await 更好"
  2. 隐式偏好 (implicit): 从行为模式中推断
     用户多次跳过某个方案 → 可能不喜欢
     用户总是选择某种风格 → 可能有偏好

三个核心机制：
  1. 信号检测 (Signal Detection) — 从交互中识别偏好信号
  2. 置信度累积 (Confidence Accumulation) — 重复观察增强置信度
  3. 矛盾解决 (Conflict Resolution) — 检测并处理矛盾的偏好

偏好演化：
  新观察到的偏好信号 → 检查是否与已有偏好一致 →
    一致 → 增强置信度
    矛盾 → 标记矛盾，降低置信度或询问用户
    新的 → 创建新偏好，置信度从低开始
"""

import uuid
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class SignalType(str, Enum):
    """偏好信号类型"""
    EXPLICIT = "explicit"       # 用户明确说出
    IMPLICIT = "implicit"       # 从行为推断
    CORRECTION = "correction"   # 用户纠正了做法
    CONFIRMATION = "confirmation"  # 用户确认了做法
    PATTERN = "pattern"         # 跨多次交互的行为模式


class PreferenceSignal:
    """
    一个偏好信号 — 可能是新的偏好或对已有偏好的强化

    设计：
      - 不直接等于偏好，需要积累足够置信度才成为"确认的偏好"
      - 置信度 < 0.3: 弱信号，仅供参考
      - 置信度 0.3-0.6: 中等信号，值得注意
      - 置信度 > 0.6: 强信号，应主动应用
    """

    def __init__(
        self,
        preference: str,
        signal_type: SignalType = SignalType.EXPLICIT,
        confidence: float = 0.3,
        context: str = "",
        category: str = "general",
        source: str = "",
    ):
        self.signal_id = str(uuid.uuid4())[:8]
        self.preference = preference              # 偏好内容
        self.signal_type = signal_type            # 信号来源类型
        self.confidence = confidence              # 当前置信度 0-1
        self.occurrences = 1                      # 观察到的次数
        self.first_seen = datetime.now()
        self.last_seen = datetime.now()
        self.context = context                    # 在什么上下文中观察到的
        self.category = category                  # coding_style / communication / tool / workflow / other
        self.source = source                      # 来源对话 ID
        self.related_signals: List[str] = []      # 相关信号 ID
        self.contradictory_to: List[str] = []     # 矛盾于哪些已有偏好
        self.evidence: List[str] = []             # 支持证据（用户原话或行为描述）

    def reinforce(self, context: str = "", evidence: str = "") -> None:
        """强化：再次观察到相同偏好"""
        self.occurrences += 1
        self.last_seen = datetime.now()
        # 置信度递增，但增速递减（避免无限增长）
        self.confidence = min(0.95, self.confidence + (1.0 - self.confidence) * 0.3)
        if context:
            self.context = context
        if evidence:
            self.evidence.append(evidence)

    def weaken(self, reason: str = "") -> None:
        """弱化：观察到矛盾信号"""
        self.confidence = max(0.05, self.confidence * 0.5)
        if reason:
            self.evidence.append(f"[弱化] {reason}")

    def is_active(self, threshold: float = 0.4) -> bool:
        """是否已足够确信可以应用"""
        return self.confidence >= threshold

    def is_strong(self) -> bool:
        """是否是强偏好（高置信度 + 多次观察）"""
        return self.confidence >= 0.7 and self.occurrences >= 2

    def age_days(self) -> float:
        """距离上次观察的天数"""
        delta = datetime.now() - self.last_seen
        return delta.total_seconds() / 86400

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "preference": self.preference,
            "signal_type": self.signal_type.value,
            "confidence": round(self.confidence, 2),
            "occurrences": self.occurrences,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "category": self.category,
            "evidence": self.evidence[-3:],  # 最近 3 条证据
        }

    def __repr__(self):
        strength = "强" if self.is_strong() else ("中" if self.is_active() else "弱")
        return f"<Signal({strength}) {self.category}: {self.preference[:50]}...>"


class PreferenceLearner:
    """
    偏好学习器 — 积累用户偏好并管理其生命周期

    使用示例:
        learner = PreferenceLearner(llm=llm_provider)

        # 从交互中检测偏好
        signals = learner.detect_signals(
            user_message="能不能不要每次都在末尾加总结？我自己会看",
            context="代码审查场景",
        )

        # 获取当前活跃的偏好
        active = learner.get_active_preferences()
        for pref in active:
            print(f"{pref.preference} (置信度: {pref.confidence})")

        # 生成偏好注入文本
        prompt = learner.get_preference_prompt()
    """

    # 偏好类别
    CATEGORIES = [
        "coding_style",     # 代码风格偏好
        "communication",    # 沟通方式偏好
        "tool",             # 工具/技术栈偏好
        "workflow",         # 工作流偏好
        "architecture",     # 架构偏好
        "other",            # 其他
    ]

    def __init__(self, file_store=None, llm=None):
        """
        Args:
            file_store: FileStore 实例
            llm: LLMProvider 实例
        """
        self.file_store = file_store
        self.llm = llm

        # 所有检测到的信号
        self._signals: Dict[str, PreferenceSignal] = {}

        # 类别索引
        self._by_category: Dict[str, List[str]] = defaultdict(list)

    # ==================== 信号检测 ====================

    def detect_signals(
        self,
        user_message: str,
        context: str = "",
        source: str = "",
    ) -> List[PreferenceSignal]:
        """
        从用户消息中检测偏好信号

        使用 LLM 识别：
          - 显式偏好陈述（"我喜欢X"、"不要Y"、"Z更好"）
          - 纠正信号（"不对，应该是..."）
          - 确认信号（"对，就这样"）

        Args:
            user_message: 用户消息
            context: 当前的上下文
            source: 来源

        Returns:
            检测到的偏好信号列表
        """
        if len(user_message) < 5:
            return []

        signals = []

        # 1. 基于规则的快速检测（不依赖 LLM）
        rule_signals = self._rule_based_detect(user_message, context, source)
        signals.extend(rule_signals)

        # 2. LLM 深度检测
        if self.llm and self._needs_deep_analysis(user_message):
            llm_signals = self._llm_detect(user_message, context, source)
            signals.extend(llm_signals)

        # 3. 对每个信号进行去重/合并/更新
        final_signals = []
        for signal in signals:
            merged = self._merge_or_create(signal)
            final_signals.append(merged)

        return final_signals

    def observe_interaction(
        self,
        user_message: str,
        assistant_response: str,
        user_reaction: str = "",
        context: str = "",
        source: str = "",
    ) -> Dict[str, Any]:
        """
        观察完整的一次交互（请求→响应→反应），从中学习

        这不仅仅是检测偏好，还包括：
          - 用户接受了什么（隐式正面信号）
          - 用户纠正了什么（显式负面信号 → 偏好反转）
          - 用户追问了什么（兴趣信号）

        Returns:
            学习结果汇总
        """
        result = {
            "signals_detected": [],
            "signals_reinforced": [],
            "signals_weakened": [],
            "new_habits": [],
        }

        # 1. 从用户消息中检测
        signals = self.detect_signals(user_message, context, source)
        result["signals_detected"] = [s.to_dict() for s in signals]

        # 2. 如果用户有后续反应，分析
        if user_reaction:
            reaction_signals = self._analyze_reaction(user_reaction, assistant_response)
            for sig in reaction_signals:
                merged = self._merge_or_create(sig)
                if merged.occurrences > 1:
                    result["signals_reinforced"].append(merged.to_dict())
                result["signals_detected"].append(merged.to_dict())

        # 3. 检测跨 session 的行为模式
        habits = self._detect_habits()
        result["new_habits"] = habits

        return result

    # ==================== 读取操作 ====================

    def get_active_preferences(
        self,
        min_confidence: float = 0.4,
        category: Optional[str] = None,
    ) -> List[PreferenceSignal]:
        """
        获取当前活跃的偏好（置信度足够高的）

        Args:
            min_confidence: 最低置信度
            category: 限定类别

        Returns:
            偏好列表（按置信度排序）
        """
        signals = list(self._signals.values())

        if category:
            category_ids = set(self._by_category.get(category, []))
            signals = [s for s in signals if s.signal_id in category_ids]

        active = [s for s in signals if s.is_active(min_confidence)]
        active.sort(key=lambda s: (s.confidence, s.occurrences), reverse=True)
        return active

    def get_strong_preferences(self) -> List[PreferenceSignal]:
        """获取强偏好（高置信度 + 多次观察）"""
        return [s for s in self._signals.values() if s.is_strong()]

    def get_preference_prompt(self) -> str:
        """
        生成可注入 LLM 系统提示的偏好文本

        Returns:
            格式化的偏好指导文本
        """
        active = self.get_active_preferences(min_confidence=0.3)
        if not active:
            return ""

        # 按类别分组
        by_cat = defaultdict(list)
        for sig in active:
            by_cat[sig.category].append(sig)

        parts = ["## 用户偏好 (已学习)"]

        category_labels = {
            "coding_style": "代码风格",
            "communication": "沟通方式",
            "tool": "工具偏好",
            "workflow": "工作流程",
            "architecture": "架构偏好",
            "other": "其他",
        }

        for cat, signals in by_cat.items():
            label = category_labels.get(cat, cat)
            parts.append(f"\n### {label}")
            for sig in signals[:5]:  # 每类最多 5 条
                strength_mark = "★" if sig.is_strong() else "●"
                parts.append(f"- {strength_mark} {sig.preference}")
                if sig.evidence:
                    parts.append(f"  *证据: {sig.evidence[-1][:100]}*")

        return "\n".join(parts)

    def get_conflicts(self) -> List[Dict[str, Any]]:
        """检测偏好之间的矛盾"""
        conflicts = []
        signals = list(self._signals.values())

        for i, s1 in enumerate(signals):
            for s2 in signals[i + 1:]:
                if s2.signal_id in s1.contradictory_to or s1.signal_id in s2.contradictory_to:
                    conflicts.append({
                        "signal_a": s1.preference,
                        "signal_b": s2.preference,
                        "confidence_a": s1.confidence,
                        "confidence_b": s2.confidence,
                    })

        return conflicts

    def get_statistics(self) -> Dict[str, Any]:
        """偏好统计"""
        all_signals = list(self._signals.values())
        by_cat = defaultdict(int)
        by_strength = {"strong": 0, "active": 0, "weak": 0}

        for sig in all_signals:
            by_cat[sig.category] += 1
            if sig.is_strong():
                by_strength["strong"] += 1
            elif sig.is_active():
                by_strength["active"] += 1
            else:
                by_strength["weak"] += 1

        return {
            "total_signals": len(all_signals),
            "strong_preferences": by_strength["strong"],
            "active_preferences": by_strength["active"],
            "weak_signals": by_strength["weak"],
            "by_category": dict(by_cat),
            "conflicts": len(self.get_conflicts()),
        }

    # ==================== 内部方法 ====================

    def _rule_based_detect(
        self, message: str, context: str, source: str
    ) -> List[PreferenceSignal]:
        """基于规则快速检测偏好信号（不依赖 LLM）"""

        signals = []
        msg_lower = message.lower()

        # 显式偏好模式
        explicit_patterns = [
            ("我喜欢", SignalType.EXPLICIT, 0.5),
            ("我偏好", SignalType.EXPLICIT, 0.5),
            ("我更倾向", SignalType.EXPLICIT, 0.5),
            ("我习惯", SignalType.EXPLICIT, 0.5),
            ("我希望", SignalType.EXPLICIT, 0.4),
            ("我不喜欢", SignalType.EXPLICIT, 0.5),
            ("我讨厌", SignalType.EXPLICIT, 0.5),
            ("不要", SignalType.EXPLICIT, 0.6),
            ("别", SignalType.EXPLICIT, 0.4),
            ("能不能不要", SignalType.EXPLICIT, 0.6),
            ("最好不要", SignalType.EXPLICIT, 0.5),
            ("no ", SignalType.EXPLICIT, 0.3),
            ("don't", SignalType.EXPLICIT, 0.3),
        ]

        for pattern, sig_type, base_conf in explicit_patterns:
            if pattern in msg_lower:
                # 尝试提取完整的偏好语句
                preference = self._extract_preference_sentence(message, pattern)
                category = self._classify_category(preference)

                signals.append(PreferenceSignal(
                    preference=preference,
                    signal_type=sig_type,
                    confidence=base_conf,
                    context=context,
                    category=category,
                    source=source,
                ))
                break  # 只取第一个匹配

        # 纠正模式
        correction_patterns = [
            "不对", "错了", "应该是", "正确的做法是",
            "no ", "wrong", "actually", "instead",
        ]
        for pattern in correction_patterns:
            if pattern in msg_lower:
                preference = self._extract_preference_sentence(message, pattern)
                signals.append(PreferenceSignal(
                    preference=preference,
                    signal_type=SignalType.CORRECTION,
                    confidence=0.55,  # 纠正信号可信度较高
                    context=context,
                    category=self._classify_category(preference),
                    source=source,
                ))
                break

        # 确认模式
        confirmation_patterns = [
            "对", "是的", "没错", "就这样", "很好", "perfect",
            "exactly", "right", "yes", "good",
        ]
        for pattern in confirmation_patterns:
            if msg_lower.startswith(pattern) and len(message) < 50:
                # 简短确认 → 隐式偏好信号
                signals.append(PreferenceSignal(
                    preference=f"喜欢当前处理方式 (上下文: {context[:80]})",
                    signal_type=SignalType.CONFIRMATION,
                    confidence=0.25,  # 确认信号需要更多累积
                    context=context,
                    category="communication",
                    source=source,
                ))
                break

        return signals

    def _llm_detect(
        self, message: str, context: str, source: str
    ) -> List[PreferenceSignal]:
        """使用 LLM 深度检测偏好"""
        if not self.llm:
            return []

        prompt = f"""分析以下用户消息，检测是否有偏好相关的信号。

偏好类型：
- coding_style: 代码风格、命名、注释、格式化
- communication: 回复长度、详细程度、语气、格式
- tool: 工具、库、框架选择
- workflow: 工作流程、PR 大小、提交频率
- architecture: 架构模式、设计决策
- other: 其他

用户消息：
---
{message[:1000]}
---

上下文：{context[:200]}

如果有明确的偏好信号，返回 JSON 格式（如无则返回空数组）：
[{{"preference": "偏好内容一句话", "category": "类别", "signal_type": "explicit/correction/confirmation", "confidence": 0.5}}]"""

        try:
            response = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=500,
            )
            import json
            content = response["content"].strip()
            if "```" in content:
                content = content.split("```")[1].split("```")[0]
                if content.startswith("json"):
                    content = content[4:]

            items = json.loads(content)
            if not isinstance(items, list):
                return []

            signals = []
            for item in items:
                sig_type = SignalType.EXPLICIT
                if item.get("signal_type") == "correction":
                    sig_type = SignalType.CORRECTION
                elif item.get("signal_type") == "confirmation":
                    sig_type = SignalType.CONFIRMATION

                signals.append(PreferenceSignal(
                    preference=item.get("preference", ""),
                    signal_type=sig_type,
                    confidence=float(item.get("confidence", 0.4)),
                    context=context,
                    category=item.get("category", "other"),
                    source=source,
                ))

            return signals
        except Exception as e:
            logger.debug(f"LLM preference detection failed: {e}")
            return []

    def _analyze_reaction(
        self, reaction: str, assistant_response: str
    ) -> List[PreferenceSignal]:
        """分析用户对助手回复的反应"""
        signals = []
        reaction_lower = reaction.lower()

        # 正面反应
        positive = ["好", "对", "谢谢", "perfect", "great", "thanks", "exactly"]
        if any(p in reaction_lower for p in positive) and len(reaction) < 100:
            # 用户喜欢这个回复方式 → 隐式偏好
            style_desc = self._describe_response_style(assistant_response)
            signals.append(PreferenceSignal(
                preference=f"喜欢回复风格: {style_desc}",
                signal_type=SignalType.CONFIRMATION,
                confidence=0.25,
                context=f"回复内容: {assistant_response[:100]}",
                category="communication",
            ))

        # 负面反应
        negative = ["太长", "太短", "太详细", "不够详细", "不要", "too long", "too short"]
        if any(n in reaction_lower for n in negative):
            signals.append(PreferenceSignal(
                preference=reaction[:200],
                signal_type=SignalType.CORRECTION,
                confidence=0.5,
                context=f"回复内容: {assistant_response[:100]}",
                category="communication",
            ))

        return signals

    def _merge_or_create(self, new_signal: PreferenceSignal) -> PreferenceSignal:
        """将新信号与已有信号合并或创建新信号"""
        # 查找相似的已有信号
        existing = self._find_similar(new_signal)

        if existing:
            # 强化已有信号
            existing.reinforce(
                context=new_signal.context,
                evidence=new_signal.preference,
            )
            logger.debug(f"Reinforced preference: {existing.preference[:50]} (conf={existing.confidence:.2f})")
            return existing
        else:
            # 检查是否与任何已有偏好矛盾
            conflicts = self._check_conflicts(new_signal)
            if conflicts:
                new_signal.contradictory_to = [c.signal_id for c in conflicts]
                # 如果有矛盾，降低新信号的初始置信度
                new_signal.confidence *= 0.7

            # 存储新信号
            self._signals[new_signal.signal_id] = new_signal
            self._by_category[new_signal.category].append(new_signal.signal_id)
            logger.debug(f"New preference signal: {new_signal.preference[:50]} (conf={new_signal.confidence:.2f})")
            return new_signal

    def _find_similar(self, signal: PreferenceSignal) -> Optional[PreferenceSignal]:
        """查找相似的已有信号（用于合并）"""
        signal_lower = signal.preference.lower()

        for existing in self._signals.values():
            existing_lower = existing.preference.lower()

            # 完全相同
            if signal_lower == existing_lower:
                return existing

            # 高度重叠（>70% 字符重叠）
            common = sum(1 for c in signal_lower if c in existing_lower)
            overlap = common / max(len(signal_lower), len(existing_lower), 1)
            if overlap > 0.7 and existing.category == signal.category:
                return existing

        return None

    def _check_conflicts(self, signal: PreferenceSignal) -> List[PreferenceSignal]:
        """检查新信号是否与已有偏好矛盾"""
        conflicts = []

        # 简单的矛盾检测：如果新偏好包含否定词且内容相似
        signal_lower = signal.preference.lower()
        is_negative = any(n in signal_lower for n in ["不要", "不喜欢", "讨厌", "don't", "no "])

        if is_negative:
            # 提取否定后的内容
            for neg in ["不要", "不喜欢", "讨厌", "don't ", "no "]:
                if neg in signal_lower:
                    positive_part = signal_lower.split(neg, 1)[-1].strip()
                    # 检查是否有偏好肯定相同的内容
                    for existing in self._signals.values():
                        if positive_part[:10] in existing.preference.lower():
                            conflicts.append(existing)
                            existing.weaken(f"矛盾信号: {signal.preference[:100]}")
                    break

        return conflicts

    def _detect_habits(self) -> List[Dict[str, Any]]:
        """
        检测跨 session 的行为模式（习惯）

        判断标准：
          - 同一类别下有多个高置信度信号
          - 同一信号被多次强化
          - 信号之间有内在一致性
        """
        habits = []

        for cat, signal_ids in self._by_category.items():
            cat_signals = [self._signals[sid] for sid in signal_ids if sid in self._signals]
            strong_signals = [s for s in cat_signals if s.is_strong()]

            if len(strong_signals) >= 2:
                # 同类别多个强偏好 → 可能形成习惯
                habits.append({
                    "category": cat,
                    "pattern": f"在 {cat} 方面有明确的偏好模式",
                    "signals": [s.preference for s in strong_signals],
                    "confidence": sum(s.confidence for s in strong_signals) / len(strong_signals),
                })

        # 检测高频类别（隐式习惯）
        for cat, signal_ids in self._by_category.items():
            if len(signal_ids) >= 5:
                habits.append({
                    "category": cat,
                    "pattern": f"频繁表达对 {cat} 的关注",
                    "signal_count": len(signal_ids),
                    "confidence": 0.4,
                })

        return habits

    def _needs_deep_analysis(self, message: str) -> bool:
        """判断是否需要 LLM 深度分析"""
        # 已有规则匹配到则跳过 LLM（节省成本）
        quick_indicators = [
            "我喜欢", "我不喜欢", "我偏好", "我希望", "不要", "别",
            "prefer", "don't", "hate", "love",
        ]
        return len(message) > 30 and not any(ind in message.lower() for ind in quick_indicators[:5])

    def _extract_preference_sentence(self, message: str, pattern: str) -> str:
        """从消息中提取包含偏好模式的完整句子"""
        idx = message.lower().find(pattern.lower())
        if idx < 0:
            return message[:200]

        # 从模式位置向前后扩展到句子边界
        start = max(0, idx - 20)
        end = min(len(message), idx + 200)
        return message[start:end].strip("，。,.")

    def _classify_category(self, text: str) -> str:
        """基于关键词分类偏好类别"""
        text_lower = text.lower()

        keywords = {
            "coding_style": ["代码", "命名", "注释", "格式", "style", "code", "naming", "comment", "format", "缩进"],
            "communication": ["回复", "回答", "详细", "简短", "语气", "总结", "response", "detail", "summary"],
            "tool": ["工具", "库", "框架", "语言", "tool", "library", "framework", "language", "ORM"],
            "workflow": ["流程", "PR", "提交", "commit", "审查", "review", "发布", "deploy"],
            "architecture": ["架构", "设计", "模式", "pattern", "architecture", "design", "重构", "refactor"],
        }

        for cat, kws in keywords.items():
            if any(kw in text_lower for kw in kws):
                return cat

        return "other"

    def _describe_response_style(self, response: str) -> str:
        """描述回复的风格特征"""
        length = len(response)
        has_code = "```" in response
        has_bullets = "- " in response or "* " in response

        features = []
        if length < 200:
            features.append("简洁")
        elif length > 1000:
            features.append("详细")

        if has_code:
            features.append("包含代码示例")
        if has_bullets:
            features.append("使用要点列表")

        return "、".join(features) if features else "标准回复"

    def __repr__(self):
        stats = self.get_statistics()
        return (
            f"<PreferenceLearner("
            f"signals={stats['total_signals']}, "
            f"strong={stats['strong_preferences']}, "
            f"active={stats['active_preferences']})>"
        )
