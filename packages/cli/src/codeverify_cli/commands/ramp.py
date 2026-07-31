"""CodeVerify CLI - Ramp commands."""

from __future__ import annotations

import click
from rich.panel import Panel

from codeverify_cli.commands import console


@click.group()
def ramp() -> None:
    """Gradual verification ramp commands.

    Manage warnings-only onboarding mode for repositories.
    """
    pass


@ramp.command("start")
@click.argument("repository")
@click.option("--baseline-days", type=int, default=7, help="Days for baseline collection")
@click.option("--observation-days", type=int, default=14, help="Days for observation period")
@click.option("--transition-days", type=int, default=14, help="Days for transition period")
@click.pass_context
def ramp_start(
    ctx: click.Context,
    repository: str,
    baseline_days: int,
    observation_days: int,
    transition_days: int,
) -> None:
    """Start verification ramp for a repository.

    Examples:

        codeverify ramp start myorg/myrepo
        codeverify ramp start myrepo --baseline-days 3 --observation-days 7
    """
    from codeverify_core.gradual_ramp import GradualVerificationRamp, RampSchedule

    console.print(
        Panel.fit("[bold blue]CodeVerify[/bold blue] - Gradual Ramp", subtitle="Onboarding Mode")
    )

    schedule = RampSchedule(
        baseline_days=baseline_days,
        observation_days=observation_days,
        transition_days=transition_days,
    )

    ramp_manager = GradualVerificationRamp(default_schedule=schedule)
    ramp_manager.start_ramp(repository)

    console.print(f"[green]✓ Ramp started for {repository}[/green]")
    console.print("\nSchedule:")
    console.print(f"  Baseline: {baseline_days} days")
    console.print(f"  Observation: {observation_days} days")
    console.print(f"  Transition: {transition_days} days")
    console.print(
        f"\nTotal: {baseline_days + observation_days + transition_days} days until full enforcement"
    )


@ramp.command("status")
@click.argument("repository")
@click.pass_context
def ramp_status(ctx: click.Context, repository: str) -> None:
    """Show ramp status for a repository.

    Examples:

        codeverify ramp status myorg/myrepo
    """
    from codeverify_core.gradual_ramp import GradualVerificationRamp

    ramp_manager = GradualVerificationRamp()
    progress = ramp_manager.get_progress_report(repository)

    if not progress:
        console.print(f"[yellow]No ramp found for {repository}[/yellow]")
        console.print("Use 'codeverify ramp start' to begin")
        return

    console.print(
        Panel.fit(
            f"[bold blue]Ramp Status:[/bold blue] {repository}",
        )
    )

    phase_colors = {
        "baseline": "blue",
        "observation": "yellow",
        "transition": "orange3",
        "enforcing": "green",
    }
    phase = progress.current_phase.value
    color = phase_colors.get(phase, "white")

    console.print(f"Phase: [{color}]{phase.upper()}[/{color}]")
    console.print(f"Enforcement: {progress.enforcement_level.value}")
    console.print(f"Progress: {progress.percent_complete:.1f}%")
    console.print(f"Days elapsed: {progress.days_elapsed}")
    console.print(f"Days remaining: {progress.days_remaining}")
    console.print(f"\nNext milestone: {progress.next_milestone}")


@ramp.command("pause")
@click.argument("repository")
@click.pass_context
def ramp_pause(ctx: click.Context, repository: str) -> None:
    """Pause ramp for a repository."""
    from codeverify_core.gradual_ramp import GradualVerificationRamp

    ramp_manager = GradualVerificationRamp()
    if ramp_manager.pause_ramp(repository):
        console.print(f"[yellow]⏸ Ramp paused for {repository}[/yellow]")
    else:
        console.print(f"[red]No ramp found for {repository}[/red]")


@ramp.command("resume")
@click.argument("repository")
@click.pass_context
def ramp_resume(ctx: click.Context, repository: str) -> None:
    """Resume paused ramp for a repository."""
    from codeverify_core.gradual_ramp import GradualVerificationRamp

    ramp_manager = GradualVerificationRamp()
    if ramp_manager.resume_ramp(repository):
        console.print(f"[green]▶ Ramp resumed for {repository}[/green]")
    else:
        console.print(f"[red]No ramp found for {repository}[/red]")


@ramp.command("end")
@click.argument("repository")
@click.option("--confirm", is_flag=True, help="Skip confirmation")
@click.pass_context
def ramp_end(ctx: click.Context, repository: str, confirm: bool) -> None:
    """End ramp and enable full enforcement."""
    from codeverify_core.gradual_ramp import GradualVerificationRamp

    if not confirm and not click.confirm(f"End ramp and enable full enforcement for {repository}?"):
        return

    ramp_manager = GradualVerificationRamp()
    if ramp_manager.end_ramp(repository):
        console.print(f"[green]✓ Full enforcement enabled for {repository}[/green]")
    else:
        console.print(f"[red]No ramp found for {repository}[/red]")
