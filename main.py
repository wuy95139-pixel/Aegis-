"""
Aegis - 多智能体个人工作助理
=============================
主入口文件

运行方式:
    # 交互模式
    python main.py

    # 单次处理模式 (文件处理)
    python main.py --file ./docs/meeting.docx --action translate

    # 研究模式
    python main.py --research "2024年AI发展趋势" --output report.md

    # 检查提醒
    python main.py --check-reminders

完整示例流程 (见文件末尾的 example_workflow 函数):
    用户上传会议纪要 → 翻译 → 提取待办 → 自动分派 → 设置提醒
"""

import sys
import argparse
from pathlib import Path

# 确保 src 目录在 Python 路径中
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.core.llm.provider import create_llm_provider
from src.core.memory.memory_manager import MemoryManager
from src.core.agents.orchestrator import Orchestrator
from src.workflows.file_processing import run_file_processing_workflow
from src.workflows.research_report import run_research_workflow
from src.workflows.reminder_followup import run_reminder_followup_workflow


def init_aegis():
    """
    初始化 Aegis 系统

    初始化顺序:
      1. 加载配置
      2. 设置日志
      3. 初始化 LLM Provider
      4. 初始化记忆系统 (短期 + 长期)
      5. 创建 Orchestrator
    """
    print("=" * 60)
    print("  Aegis - 多智能体个人工作助理  v0.1.0")
    print("=" * 60)

    # 1. 加载配置
    config = Config()
    log_level = config.get("system.log_level", "INFO")
    log_dir = config.get("system.log_dir", "./logs")

    # 2. 设置日志
    logger = setup_logger("aegis", level=log_level, log_dir=log_dir)
    logger.info("Initializing Aegis...")

    # 3. 初始化 LLM
    llm_config = config.get("llm", {})
    llm = create_llm_provider({
        "provider": llm_config.get("provider", "openai"),
        "model": llm_config.get("model", "deepseek-v4-pro"),
        "api_key": llm_config.get("api_key"),
        "api_base": llm_config.get("api_base"),
        "temperature": llm_config.get("temperature", 0.7),
        "max_tokens": llm_config.get("max_tokens", 4096),
    })

    # 4. 初始化记忆系统（统一 MemoryManager，含 8 个子系统）
    mm = MemoryManager(llm=llm, config=config.get_all())
    mm.initialize(
        file_store_dir=config.get("memory.file_store.base_dir", "./data/memory"),
        chroma_dir=config.get("memory.long_term.persist_dir", "./data/chroma_db"),
        collection_name=config.get("memory.long_term.collection_name", "aegis_long_term_memory"),
        embedding_model=config.get("embedding.model", "text-embedding-3-small"),
        short_term_max_tokens=config.get("memory.short_term.max_tokens", 16000),
        short_term_window=config.get("memory.short_term.window_size", 20),
    )

    # 5. 创建 Orchestrator（使用记忆检索器 + 经验库 + 偏好学习器）
    orchestrator = Orchestrator(
        llm=llm,
        memory=mm.retriever,
        config=config.get_all(),
    )
    # 注入学习层，orchestrator 可以直接使用自适应能力
    orchestrator.memory_manager = mm

    logger.info("Aegis initialized successfully!")
    print("✅ Aegis 初始化完成\n")

    return orchestrator, config, logger


def interactive_mode(orchestrator):
    """交互模式 — 持续对话"""
    print("进入交互模式，输入 'exit' 或 'quit' 退出，输入 'help' 查看帮助。\n")

    help_text = """
可用命令:
  file <path> [action]  - 处理文件 (action: translate/polish/parse)
  research <topic>      - 研究分析话题
  remind <title> <time> - 设置提醒 (time: YYYY-MM-DDTHH:MM)
  tasks                 - 查看任务和提醒
  help                  - 显示帮助
  exit/quit             - 退出
"""

    while True:
        try:
            user_input = input("🤖 Aegis > ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit"):
                print("再见！")
                break

            if user_input.lower() == "help":
                print(help_text)
                continue

            if user_input.lower() == "tasks":
                result = orchestrator.execute_task_inquiry("")
                print(result["response"])
                continue

            # 解析命令
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd == "file":
                # file <path> [action]
                subparts = arg.split(maxsplit=1)
                filepath = subparts[0]
                action = subparts[1] if len(subparts) > 1 else "translate"
                result = orchestrator.process_user_request(
                    f"请{action}这个文件",
                    attached_file=filepath,
                )
                print(result.get("response", result))

            elif cmd == "research":
                result = orchestrator.process_user_request(
                    f"研究一下 {arg}",
                )
                print(result.get("response", result))

            elif cmd == "remind":
                # remind <title> <time>
                result = orchestrator.process_user_request(
                    f"设置提醒: {arg}",
                )
                print(result.get("response", result))

            else:
                # 通用对话
                result = orchestrator.process_user_request(user_input)
                print(result.get("response", result))

        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            print(f"❌ 错误: {e}")


# ===================== 示例工作流 =====================

def example_workflow():
    """
    完整示例流程：会议纪要文件处理

    流程演示:
      用户上传一个会议纪要文件 → 翻译成中文 →
      从内容中提取待办事项 → 按规则自动指派给相关人员 →
      设置提醒 → 输出最终结果

    这是一个端到端的完整演示，展示多个 Agent 如何协作完成一个复合任务。

    注意：此函数使用模拟文件，演示完整流程。
          实际使用时替换为真实文件路径。
    """
    print("=" * 60)
    print("  示例: 会议纪要文件处理完整流程")
    print("=" * 60)

    orchestrator, config, logger = init_aegis()

    # Step 0: 创建模拟的会议纪要文件
    from tests.fixtures.sample_data import create_sample_meeting_doc
    sample_file = create_sample_meeting_doc()
    print(f"\n📁 模拟文件已创建: {sample_file}\n")

    # Step 1: 用户上传文件并提出处理要求
    user_request = (
        "请将这个会议纪要翻译成中文，然后提取其中的待办事项，"
        "按角色自动分派给相关人员，并为每个待办设置截止提醒。"
    )

    print(f"👤 用户请求: {user_request}\n")
    print("=" * 60)
    print("  开始多 Agent 协作处理...")
    print("=" * 60)

    # Step 2: Orchestrator 分析意图并调度 Agent 协作
    result = orchestrator.process_user_request(
        user_message=user_request,
        attached_file=sample_file,
    )

    # Step 3: 展示各 Agent 的处理结果
    print("\n" + "=" * 60)
    print("  📊 处理结果")
    print("=" * 60)

    print(f"\n🎯 识别意图: {result.get('intent', 'unknown')}")

    print(f"\n{result.get('response', '无响应内容')}")

    # 详细结果展示
    file_result = result.get("file_result", {})
    if file_result:
        parsed = file_result.get("parsed_file")
        if parsed:
            print(f"\n📄 文件详情:")
            print(f"   - 文件名: {parsed.filename}")
            print(f"   - 类型: {parsed.file_type.value}")
            print(f"   - 提取字符数: {len(parsed.raw_text)}")

    dispatch_result = result.get("dispatch_result", {})
    if dispatch_result:
        print(f"\n📋 任务分派详情:")
        print(f"   - {dispatch_result.get('summary', '')}")
        for todo in dispatch_result.get("assigned_todos", []):
            assignee_name = todo.assignee.name if todo.assignee else "未分配"
            print(f"   - [{todo.priority.value}] {todo.title} → {assignee_name}")

        if dispatch_result.get("unassigned"):
            print(f"   - 未分配: {len(dispatch_result['unassigned'])} 个")

    print("\n" + "=" * 60)
    print("  ✅ 多 Agent 协作完成")
    print("=" * 60)

    return result


# ===================== CLI =====================

def main():
    parser = argparse.ArgumentParser(
        description="Aegis - 多智能体个人工作助理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py                              # 交互模式
  python main.py --example                    # 运行完整示例
  python main.py --file meeting.docx --action translate
  python main.py --research "AI agent trends 2024" --output report.md
  python main.py --check-reminders
        """,
    )
    parser.add_argument("--file", "-f", type=str, help="要处理的文件路径")
    parser.add_argument("--action", "-a", type=str, default="translate",
                        choices=["translate", "polish", "parse", "generate_ppt"],
                        help="文件处理操作 (默认: translate)")
    parser.add_argument("--target-lang", type=str, default="zh-CN", help="翻译目标语言")
    parser.add_argument("--research", "-r", type=str, help="研究主题")
    parser.add_argument("--output", "-o", type=str, help="输出文件路径")
    parser.add_argument("--check-reminders", action="store_true", help="检查到期提醒")
    parser.add_argument("--example", "-e", action="store_true", help="运行完整示例流程")
    parser.add_argument("--config", "-c", type=str, help="配置文件路径")

    args = parser.parse_args()

    # 运行示例
    if args.example:
        example_workflow()
        return

    orchestrator = None
    try:
        # 初始化系统
        if args.config:
            Config(args.config)

        orchestrator, config, logger = init_aegis()

        # 单次文件处理
        if args.file:
            result = run_file_processing_workflow(
                orchestrator=orchestrator,
                filepath=args.file,
                action=args.action,
                target_lang=args.target_lang,
            )
            print(f"\n处理结果: {result.get('status', 'unknown')}")
            print(f"文件: {result.get('filename', '')}")
            print(f"操作: {result.get('action', '')}")
            print(f"提取待办: {result.get('todos_count', 0)}")
            print(f"已分派: {result.get('assigned_count', 0)}")
            print(f"已创建提醒: {result.get('reminders_created', 0)}")
            return

        # 研究分析
        if args.research:
            result = run_research_workflow(
                orchestrator=orchestrator,
                topic=args.research,
                save_to_file=args.output,
            )
            if args.output:
                print(f"报告已保存至: {args.output}")
            else:
                print(result.get("report_markdown", ""))
            return

        # 检查提醒
        if args.check_reminders:
            result = run_reminder_followup_workflow(orchestrator)
            print(result.get("summary", ""))
            return

        # 默认: 交互模式
        interactive_mode(orchestrator)

    finally:
        if orchestrator is not None:
            orchestrator.shutdown()


if __name__ == "__main__":
    main()
