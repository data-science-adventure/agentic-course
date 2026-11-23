"""Main entrypoint for the Agentic Course CLI runner.

Dynamically scans module folders inside `agentic_course`, lists executable lessons,
and provides an interactive menu to run any lesson seamlessly.
"""

import importlib
import inspect
import pkgutil
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

console = Console()


def discover_lessons():
    """Scans the `agentic_course` directory for module folders and executable Python scripts.

    Returns:
        list[dict]: A sorted list of dictionaries containing module info, file paths,
                    import paths, and lesson names.
    """
    base_dir = Path(__file__).parent
    lessons = []

    # Find directories that start with module_ or represent course sections
    module_dirs = sorted([
        d for d in base_dir.iterdir()
        if d.is_dir() and not d.name.startswith(("_", "."))
    ])

    for mod_dir in module_dirs:
        # Find all python files inside the module folder except __init__.py and cli.py
        py_files = sorted([
            f for f in mod_dir.glob("*.py")
            if not f.name.startswith(("_", ".")) and f.name != "cli.py"
        ])

        for py_file in py_files:
            import_path = f"agentic_course.{mod_dir.name}.{py_file.stem}"
            lessons.append({
                "module": mod_dir.name,
                "file_name": py_file.name,
                "import_path": import_path,
                "title": py_file.stem.replace("_", " ").title(),
            })

    return lessons


def render_menu(lessons):
    """Renders a styled Rich Table listing all discovered lessons."""
    table = Table(
        title="📚 Agentic Course - Available Lessons",
        header_style="bold magenta",
        show_lines=True,
    )
    table.add_column("Option", justify="center", style="cyan", no_wrap=True)
    table.add_column("Module Folder", style="bold yellow")
    table.add_column("Lesson Script", style="green")
    table.add_column("Import Path", style="dim white")

    for idx, lesson in enumerate(lessons, start=1):
        table.add_row(
            str(idx),
            lesson["module"],
            lesson["file_name"],
            lesson["import_path"],
        )

    console.clear()
    console.print(table)
    console.print()


def execute_lesson(lesson):
    """Dynamically imports the selected lesson module and runs its `main()` function."""
    console.print(
        Panel(
            f"[bold green]Executing:[bold green] {lesson['title']}\n"
            f"[dim]Path: {lesson['import_path']}[/dim]",
            title="🚀 Starting Lesson",
            border_style="cyan",
        )
    )

    try:
        # Dynamically import the python module
        mod = importlib.import_module(lesson["import_path"])

        # Execute main() if it exists in the target script
        if hasattr(mod, "main") and callable(mod.main):
            mod.main()
        else:
            console.print(
                f"[bold red]❌ Error:[bold red] '{lesson['file_name']}' does not contain a main() function."
            )
    except Exception as e:
        console.print(f"[bold red]💥 Execution Error:[bold red] {e}")
        console.print_exception(show_locals=True)


def main():
    """Main CLI menu loop."""
    while True:
        lessons = discover_lessons()

        if not lessons:
            console.print(
                "[bold red]No lesson scripts found inside `src/agentic_course/`![/bold red]"
            )
            return

        render_menu(lessons)

        # Build prompt options (1..N plus 'q' to quit)
        choices = [str(i) for i in range(1, len(lessons) + 1)] + ["q", "Q"]

        selected = Prompt.ask(
            "[bold white]Select a lesson to run (or 'q' to quit)[/bold white]",
            choices=choices,
            default="1",
        )

        if selected.lower() == "q":
            console.print("\n[bold cyan] 👋 Goodbye! Happy Learning![/bold cyan]\n")
            break

        lesson_idx = int(selected) - 1
        selected_lesson = lessons[lesson_idx]

        execute_lesson(selected_lesson)

        # Pause after execution before re-rendering the menu
        console.print()
        Prompt.ask("[bold yellow]Press Enter to return to the menu...[/bold yellow]")


if __name__ == "__main__":
    main()