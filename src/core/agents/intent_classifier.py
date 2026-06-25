"""
意图分类器 (IntentClassifier)
==============================
从 Orchestrator 水平拆分出的意图识别模块。

职责:
  - 快速预检：对明显的闲聊/问候直接返回 general_chat，跳过 LLM 调用
  - LLM 分类：使用 function calling 进行精确意图识别 + 参数提取
  - 关键词回退：LLM 调用失败时的兜底安全网

使用:
    classifier = IntentClassifier(llm)
    intent_info = classifier.classify(user_message, attached_file, context)
    # → {"intent": "reminder_set", "params": {"title": "...", ...}}
"""

import re
import logging
from typing import Optional

from src.core.llm.provider import LLMProvider
from src.core.tools.time_tools import get_time_context
from src.utils.common import extract_json_dict, sanitize_for_prompt

logger = logging.getLogger(__name__)


# ===================== Function Calling 工具定义 =====================

AVAILABLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前日期、星期几和精确时间。当用户问'现在几点'或需要知道当前日期/星期几时调用。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "parse_time",
            "description": "将中文自然语言时间表达解析为精确的 ISO 8601 datetime 格式。支持：'明天下午3点'、'下周一上午10点'、'3天后'、'后天早上9点'、'周五晚上8点'等。当你需要精确的时间计算（如设置提醒）时，用此工具而非自己估算。",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "用户输入的中文时间表达文本，如'明天下午3点'、'下周一上午10点'",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "time_to_cron",
            "description": "将中文重复时间表达转为 cron 表达式。支持：'每天早上8点'→'0 8 * * *'、'每周一早上9点'→'0 9 * * 1'、'每周五下午5点'→'0 17 * * 5'、'每月1号上午9点'→'0 9 1 * *'、'工作日早上9点'→'0 9 * * 1-5'。当用户要设置周期性提醒时必须调用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "中文重复时间表达，如'每周一早上9点'、'每天早上8点'、'每月15号下午2点'",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_future_date",
            "description": "计算未来日期并返回具体年月日+星期几。支持：'下周五'、'3天后'、'下周三'等。当用户问'X是几月几号'或需要知道某个相对日期的具体日期时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "中文日期表达，如'下周五'、'3天后'、'下周三'",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_overdue",
            "description": "判断一个时间或提醒是否已过期，返回距离现在的相对时间（多少分钟/小时/天后）。当用户问'这个提醒过期了吗'或需要检查截止时间时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "time_str": {
                        "type": "string",
                        "description": "要检查的时间，可以是 ISO 格式（如'2025-06-15T14:00'）或中文表达（如'明天下午3点'）",
                    },
                },
                "required": ["time_str"],
            },
        },
    },
]


class IntentClassifier:
    """意图分类器 — 快速预检 + LLM 分类 + 关键词回退（三层策略）"""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    # ===================== 公开 API =====================

    def classify(self, msg: str, has_file: Optional[str] = None, context: str = "") -> dict:
        """
        一站式意图分类：快速预检 → LLM 分类 → 关键词回退

        Args:
            msg: 用户输入消息
            has_file: 上传文件的路径（有则为路径字符串，无则为 None）
            context: 记忆检索上下文

        Returns:
            {"intent": "intent_type", "params": {...}}
        """
        quick = self._quick_intent_check(msg, has_file)
        if quick:
            logger.debug(f"Quick intent: {quick['intent']}")
            return quick

        try:
            return self._llm_classify_intent(msg, has_file, context)
        except Exception as e:
            logger.warning(f"LLM intent classification failed: {e}, falling back to keyword")
            return self._keyword_classify(msg, has_file)

    # ===================== 第一层：快速预检 =====================

    def _quick_intent_check(self, msg: str, has_file: Optional[str]) -> Optional[dict]:
        """
        快速预检：对于明显是闲聊/问候的消息，直接返回 general_chat，跳过 LLM 分类。

        节省 ~1 次 LLM 调用（约 500ms-1.5s）。

        Returns:
            意图字典 或 None（None 表示无法确定，需要 LLM 分类）
        """
        m = msg.strip().lower()
        m_clean = re.sub(r"[，,。.！!？?\s]+", "", m)

        # 有文件时不做快速判断，让 LLM 精确分类
        if has_file:
            return None

        # 明显的动作关键词 → 不做快速判断，交给 LLM
        action_keywords = [
            "提醒", "别忘了", "记得", "设个闹钟", "定时", "准时",
            "翻译", "译成", "译", "翻成", "translate",
            "润色", "优化", "改写", "polish", "措辞",
            "生成ppt", "幻灯片", "演示文稿", "做ppt", "slides",
            "搜索", "研究", "查一下", "查", "search", "research",
            "待办", "任务", "todo", "要做", "还有什么",
            "简报", "早报", "briefing", "日报",
            "总结", "概括", "归纳", "summarize",
            "之前", "记忆", "说过", "remember", "recall",
            "提取待办", "行动项",
            "现在几点", "今天几号", "星期几", "日期",
            "几点了", "什么时间", "当前时间",
            "可视化分析", "全面分析", "深度分析", "综合分析", "数据分析", "完整分析",
            "画图", "画个", "图表", "生成图", "可视化", "仪表板", "看板", "dashboard",
        ]
        if any(kw in m for kw in action_keywords):
            return None

        # 明显的闲聊/社交表达 → 直接 general_chat
        chat_only = [
            # 问候 (多字符优先)
            "好久不见", "在吗", "在不在", "晚上好", "下午好", "早上好",
            "你好", "嗨", "哈喽", "早啊",
            "hi", "hello", "hey", "good morning", "good evening",
            # 感谢
            "谢谢", "感谢", "多谢", "辛苦了", "thx", "thanks", "thank you",
            # 告别
            "再见", "拜拜", "晚安", "bye", "see you", "回头见", "明天见",
            # 确认
            "知道了", "明白了", "懂了", "了解", "收到", "got it", "okay",
            "好的", "ok", "可以",
            # 自我介绍
            "你是谁", "你叫什么", "你是什么", "你的名字",
            "你能做什么", "你有什么功能", "你会什么", "你怎么用","你可以帮我做什么",
            "介绍一下你自己", "who are you", "what can you do",
            "你好吗", "你怎么样", "how are you",
            # 闲聊
            "哈哈", "呵呵", "嘿嘿", "lol", "有意思", "有趣",
            "讲个笑话", "笑话", "joke",
            "今天天气", "天气怎么样",
        ]
        for phrase in chat_only:
            if phrase in m_clean:
                return {"intent": "general_chat", "params": {}}

        # 单字确认词/感叹词：仅当消息几乎只有该单字时才算闲聊
        # 避免 "好" 匹配 "帮我做好这个PPT"
        single_char_chat = {"嗯", "哦", "行", "好", "对", "是", "早", "thank"}
        if m_clean in single_char_chat:
            return {"intent": "general_chat", "params": {}}

        # 极短消息（≤4个字符）无文件 → 大概率是闲聊
        if len(m_clean) <= 4:
            return {"intent": "general_chat", "params": {}}

        # 无法确定
        return None

    # ===================== 第二层：LLM 分类 =====================

    def _llm_classify_intent(self, msg: str, has_file: Optional[str], context: str) -> dict:
        """
        使用 LLM 进行意图识别 + 参数提取

        设计决策：相比关键词匹配，LLM 可以理解:
          - "还有什么没做完的？" → task_inquiry
          - "之前聊的那个项目怎样了" → memory_search
          - "帮我把这段改得更专业一点" → file_polish (如果有文件) 或翻译润色
        """
        file_hint = f"用户已上传文件: {has_file}" if has_file else "用户未上传文件"
        time_context = get_time_context()

        prompt = f"""你是 Aegis 的意图识别模块。根据用户输入的含义（不是关键词）来判断意图，语义相近的表达应该归为同一意图。返回 JSON。

{time_context}
{file_hint}

意图类型及示例（语义相近都算，不限以下列举）:
- file_translate: 翻译 — "翻译成英文"/"翻成日语"/"把这段话译成韩语"/"translate to Chinese"/"英文怎么说"/"日文版"/"帮我翻译一下这段: xxx"/"将xxx译成法语"/"这段话用英语怎么说"/"翻一下"/"帮我译成德语"/"用英文表达以下内容"/"英文翻译"/"中译英"/"英译中"
- file_polish: 润色改进 — "润色一下"/"改得更专业"/"优化文字"/"帮我改改"/"措辞优化"/"polish"/"改写一下"/"把这改得正式一点"/"精炼这段"/"语病修改"/"润色成学术风格"/"帮我润润色"/"文字美化"/"措辞改好一点"
- file_generate_ppt: 生成PPT — "生成PPT"/"做成幻灯片"/"做份演示文稿"/"转成PPT"/"弄个幻灯片"/"帮我做个PPT"/"生成演示文稿"/"创建PPT"/"做一份slides"
- file_extract_todos: 提取待办 — "提取待办"/"有什么需要做的"/"分派任务"/"行动项"/"任务分配"/"看看有什么待办"/"从中提取任务"/"分配一下"/"识别行动点"（仅上传文件时）
- file_qa: 针对文件内容提问/分析 — "分析这个数据"/"进行可视化分析"/"可视化分析这个数据"/"当中提到的xxx是什么"/"文件里说的xxx是什么意思"/"根据文件回答xxx"/"什么是xxx"(有文件时)/"文件提到xxx了吗"/"里面讲了什么"/"帮我分析这个文件"/"总结一下这份文件"（仅上传文件时，用户基于文件内容问了具体问题或要求分析。这是最重要的文件意图之一，优先级高于 research）
- file_parse: 查看文件 — "看看文件内容"/"解析"/"打开看看"/"浏览文件"/"展示全文"/"把内容显示出来"（仅上传文件时，用户只是要看文件、没有具体问题）
- research: 搜索网络/外部信息 — "最近xxx"/"热点"/"新闻"/"搜索xxx"/"xxx趋势"/"查一下"/"现在最火的xxx"/"最近发生了什么"/"行业动态"/"最新的xxx"。⚠️ 当用户已上传文件且提问"xxx是什么"时，应优先判断为 file_qa 而非 research！只有明确涉及网络搜索、最新动态、热点新闻的才归类为 research。用户明确说"生成报告"/"写份报告"/"研究分析"/"出份研究报告"时才生成完整报告（含SWOT）
- reminder_set: 设置提醒 — 仅在用户提到具体时间或使用"提醒"关键词时触发。有时间："提醒我下午3点开会"/"明天早上8点叫我"/"设个闹钟到3点"/"到时间叫我"/"定时提醒"/"到xxx的时候告诉我"。有"提醒/记/别忘了"关键词："提醒我xxx"/"别忘了xxx"/"记得xxx"/"别忘记xxx"/"帮我记一下（后面跟时间）"。⚠️ 如果用户只是在列举要做的事但既没有时间也没有"提醒"关键词（如"我有几件事要做：第一xxx第二xxx"、"帮我记着：xxx, xxx"、"接下来要：xxx, xxx"），不要归为 reminder_set，应归为 task_inquiry（查询/管理待办）或 general_chat
- reminder_check: 查看提醒 — "有什么提醒"/"需要跟进"/"待跟进"/"要看什么"/"还有什么没做"/"有哪些提醒"/"跟进事项"/"催一下"
- reminder_cancel: 删除/取消提醒 — "删除所有提醒"/"全部提醒清除掉"/"取消全部提醒"/"把提醒都删了"/"删掉'开会'这条提醒"/"取消'xxx'这个提醒"/"清除提醒"/"移除提醒"/"把提醒删掉"
- task_add: 创建待办任务 — 用户列举行要做的事，但既没有时间也没有"提醒"关键词。如"我有几件事要做：第一xxx第二xxx"/"帮我记着：xxx, xxx, xxx"/"接下来要完成：1.xxx 2.xxx 3.xxx"/"需要做：一xxx二xxx三xxx"/"添加到待办：xxx、xxx"/"有几项任务：xxx；xxx"。⚠️ 如果用户说了"提醒"或给出了时间（"下午3点"、"明天"），那属于 reminder_set 而不是 task_add
- task_inquiry: 任务待办/查询 — "有哪些待办"/"今天要干嘛"/"任务列表"/"还没做的"/"日程"/"工作计划"/"待处理"/"有什么事"/"有什么任务"/"查看任务"/"要做些什么"/"接下来做什么"/"接下来应该做什么"/"还有什么要做的"/"我还有其它待办吗"/"还有别的任务吗"/"其它待办"
- memory_search: 搜索记忆 — "之前聊过"/"以前说过"/"记不记得"/"回顾"/"搜索记忆"/"聊过xxx吗"/"上次说的"/"回忆一下"/"之前提到的"
- memory_summarize: 总结对话 — "总结一下"/"聊聊什么了"/"回顾对话"/"整理一下"/"归纳"/"概括"/"刚才聊了什么"
- briefing: 每日简报 — "今日简报"/"早报"/"今日概览"/"morning briefing"/"日报"/"每日汇总"/"今天摘要"
- audio_transcribe: 语音转录 — "帮我转写这段录音"/"这段会议谁说了什么"/"把音频转成文字"/"识别录音里的对话"/"语音转文字"/"会议记录整理"/"帮我听听这段"（仅上传音频文件时）
- chart_generate: 生成图表 — "画个柱状图"/"生成折线图"/"做个饼图"/"可视化数据"/"画张图"/"数据图表"/"帮我做个图表"/"用图表展示"/"画一个趋势图"/"生成图表"/"对比图"/"占比图"
- dashboard_create: 创建数据看板 — "做个仪表板"/"创建数据看板"/"生成分析看板"/"做一个dashboard"/"综合看板"/"数据大屏"/"可视化看板"/"分析仪表盘"
- visual_analysis: 综合可视化分析 — "进行可视化分析"/"全面分析这个数据"/"深度分析"/"综合分析"/"数据分析"/"完整分析"/"帮我全面分析"/"做个完整分析"（上传了CSV/Excel文件时，用户要求全面分析数据）
- workload_check: 负荷感知 — "今天忙不忙"/"看看今天的工作量"/"今天日程负载"/"今天还有精力吗"/"任务太多了"/"帮我看看负荷"/"还能接活吗"/"检查工作量"
- general_chat: 问候/闲聊/问答/不匹配以上意图的

参数提取（需要时）:
- target_lang: 目标语言 (zh-CN/en/ja/ko/fr/de/es/pt/ru/ar...)，未指明默认 zh-CN
- style: 润色风格 (professional/casual/academic/concise)，未指明默认 professional
- topic: 搜索/研究主题关键词
- title: 提醒/PPT标题
- description: 提醒描述（非时间信息）
- query: 记忆搜索关键词
- text_content: 从用户消息中提取的要翻译/润色/生成PPT的文本内容（无文件时必提取）
- report_mode: 研究意图时，判断用户是要快速回答("quick")还是完整报告("full")。默认"quick"，只有明确说"报告"/"研究分析"/"SWOT"才用"full"

⚠️ 不要提取 trigger_time 和 cron_expression！提醒时间会由专用的时间解析工具处理，比 LLM 更精准。
⚠️ 对于 reminder_set，只返回 {{"intent": "reminder_set", "params": {{"title": "...", "description": "..."}}}} 即可。

核心原则（按优先级执行）:
- 🔴 最高优先级：用户已上传文件 + 问了关于文件内容的具体问题（如"什么是xxx"/"xxx是什么意思"/"里面提到xxx了吗"）→ 必须归类为 file_qa，绝不归为 research 或 general_chat
- 🔴 reminder_set 触发条件：必须有时间表达（"下午3点"、"明天"、"下周一"等）或明确的"提醒/别忘了/记得"关键词。没有时间也没有提醒关键词的列举（如"我有几件事要做：第一X第二Y"、"帮我记着：X, Y, Z"、"接下来要完成xxx"）→ 不是 reminder_set，应归为 task_add
- 🔴 task_add vs task_inquiry 区分：用户列举要做的事（陈述式）→ task_add（创建任务）；用户问有什么要做（疑问式）→ task_inquiry（查询任务）
- 🔴 求评价/征求意见的句子：如果用户输入整体是一个问句，在征求意见或评价（句尾带"可以吗"/"行不行"/"怎么样"/"好不好"/"这样可以吗"/"怎么做比较好"/"你觉得呢"/"对吗"），即使包含编号列举（"第一...第二..."），也应归为 general_chat，绝不能归为 reminder_set
- 语义理解：变体/近义词/口语/中英混合/方言都能识别，不要死记关键词
- 无文件时翻译/润色/PPT：**必须**从用户消息中提取 text_content（如"将xxx翻译成英文"→text_content="xxx", target_lang="en"）
- 有文件时：按用户说的做，仅执行那一个操作。如果用户问了关于文件内容的具体问题（如"里面提到的xxx是什么"），→ file_qa。如果用户只是想看文件内容没有具体问题 → file_parse
- 无文件时说"提取待办"：回复提示需要上传文件

用户输入: "{sanitize_for_prompt(msg, max_len=2000)}"

只返回 JSON:
{{"intent": "intent_type", "params": {{...}}}}"""

        resp = self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500,
        )

        return extract_json_dict(resp["content"])

    # ===================== 第三层：关键词回退 =====================

    def _keyword_classify(self, msg: str, has_file: Optional[str]) -> dict:
        """关键词回退分类 — LLM 分类失败时的兜底安全网（中英双语）"""
        m = msg.lower()
        if has_file:
            if any(kw in m for kw in ["翻译", "译", "translate", "翻成", "译成", "译成", "用中文", "用英文", "用日语"]):
                return {"intent": "file_translate", "params": {}}
            if any(kw in m for kw in ["转录", "转写", "语音转文字", "会议记录", "录音", "音频", "transcribe", "audio", "语音"]):
                return {"intent": "audio_transcribe", "params": {}}
            if any(kw in m for kw in ["润色", "优化", "改写", "polish", "措辞", "refine", "improve", "rewrite", "改得更", "精炼"]):
                return {"intent": "file_polish", "params": {}}
            if any(kw in m for kw in ["ppt", "幻灯片", "演示文稿", "slides", "presentation", "生成ppt", "做ppt", "做一份slides"]):
                return {"intent": "file_generate_ppt", "params": {}}
            if any(kw in m for kw in ["待办", "提取", "行动项", "分配", "todo", "action item", "extract task", "分派"]):
                return {"intent": "file_extract_todos", "params": {}}
            # 用户问了具体问题 → 针对文件内容问答
            if any(q in m for q in ["什么", "怎么", "如何", "吗", "呢", "？", "?", "为什么", "哪里", "谁", "哪", "是否", "提到", "提到过", "当中",
                                       "分析", "总结", "概括", "可视化", "图表", "统计",
                                       "what", "how", "why", "where", "who", "which", "when", "does", "mention", "according", "based on"]):
                return {"intent": "file_qa", "params": {}}
            return {"intent": "file_parse", "params": {}}

        # 无文件时的关键词匹配（中英双语）
        if any(kw in m for kw in ["删除提醒", "取消提醒", "清除提醒", "删掉提醒", "移除提醒", "全部提醒", "所有提醒"]):
            return {"intent": "reminder_cancel", "params": {}}
        if any(kw in m for kw in ["提醒", "别忘了", "记得", "remind", "reminder", "alert", "notify", "don't forget", "记住"]):
            return {"intent": "reminder_set", "params": {}}
        if any(kw in m for kw in ["搜索", "研究", "查", "search", "research", "find", "look up", "trend", "news", "最新", "热点"]):
            return {"intent": "research", "params": {}}
        if any(kw in m for kw in ["帮我记着", "帮我记录", "记下来", "有几件事", "有几项任务", "添加到待办", "加个待办", "帮我记几件"]):
            return {"intent": "task_add", "params": {}}
        if any(kw in m for kw in ["待办", "任务", "todo", "task", "schedule", "日程", "要做", "还有什么", "接下来做什么"]):
            return {"intent": "task_inquiry", "params": {}}
        if any(kw in m for kw in ["简报", "早报", "briefing", "morning brief", "日报", "daily summary", "每日汇总"]):
            return {"intent": "briefing", "params": {}}
        if any(kw in m for kw in ["总结", "聊了什么", "summarize", "summary", "recap", "概括", "归纳", "回顾对话"]):
            return {"intent": "memory_summarize", "params": {}}
        if any(kw in m for kw in ["之前", "记忆", "说过", "remember", "recall", "memory", "以前", "上次", "回顾", "搜索记忆", "聊过"]):
            return {"intent": "memory_search", "params": {}}
        if any(kw in m for kw in ["可视化分析", "全面分析", "深度分析", "综合分析", "数据分析", "完整分析", "comprehensive analysis", "full analysis"]):
            return {"intent": "visual_analysis", "params": {}}
        if any(kw in m for kw in ["负荷", "工作量", "忙不忙", "超载", "排期", "今天忙吗", "日程负载", "任务太多", "还能接活吗", "精力", "负载", "workload", "overload", "忙不"]):
            return {"intent": "workload_check", "params": {}}
        if any(kw in m for kw in ["图表", "柱状图", "折线图", "饼图", "画图", "画个", "生成图", "可视化", "chart", "bar chart", "line chart", "pie chart", "趋势图", "占比图", "对比图"]):
            return {"intent": "chart_generate", "params": {}}
        if any(kw in m for kw in ["仪表板", "看板", "dashboard", "数据大屏", "分析看板", "可视化看板", "仪表盘"]):
            return {"intent": "dashboard_create", "params": {}}
        return {"intent": "general_chat", "params": {}}
