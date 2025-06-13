import click
import yaml
import asyncio
import sys
from pathlib import Path
import time
import json
from core.engine.unified_engine import UnifiedEngine
from core.ai.ai_optimizer import AIOptimizer

@click.group()
def cli():
    """Unified DevOps Nexus CLI"""
    pass

@cli.command()
@click.option('--output', '-o', default="nexus-infra.yml", show_default=True, help="Output YAML file name")
@click.option('--nl', is_flag=True, help="Use natural language input mode")
def generate(output, nl):
    """Interactively generate a new infrastructure config."""
    click.secho("Unified DevOps Nexus Config Generator", fg="cyan", bold=True)

    if nl:
        prompt = click.prompt("Describe your infrastructure needs in natural language")
        # TODO: Implement GPT-4 integration for parsing natural language
        click.secho("Natural language mode coming soon! Using standard mode instead.", fg="yellow")

    provider = click.prompt("Cloud provider", type=click.Choice(['aws', 'gcp', 'azure'], case_sensitive=False))
    env = click.prompt("Environment", type=click.Choice(['dev', 'staging', 'prod'], case_sensitive=False))
    service = click.prompt("Primary service", type=click.Choice(['k8s', 'redis', 'rds', 'vm'], case_sensitive=False))
    regions = click.prompt("Regions (comma-separated, e.g. us-east-1,us-west-2)", default="us-east-1").replace(" ", "").split(",")

    node_type = min_nodes = max_nodes = None
    if service == "k8s":
        node_type = click.prompt("K8s node type", default="t3.medium")
        min_nodes = click.prompt("Minimum nodes", default=2, type=int)
        max_nodes = click.prompt("Maximum nodes", default=5, type=int)

    compliance = {
        "hipaa": click.confirm("Enable HIPAA compliance?", default=False),
        "pci": click.confirm("Enable PCI compliance?", default=False),
        "auto_remediate": click.confirm("Enable auto-remediation?", default=True)
    }

    service_spec = {"type": service}
    if service == "k8s":
        service_spec["resources"] = [{
            "kind": "eks",
            "specs": {
                "nodeType": node_type,
                "minNodes": min_nodes,
                "maxNodes": max_nodes
            }
        }]

    config = {
        "version": "1.0",
        "metadata": {"environment": env},
        "providers": {
            provider: {
                "enabled": True,
                "regions": regions,
                "services": [service_spec]
            }
        },
        "compliance": compliance
    }

    with open(output, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    click.secho(f"\nGenerated {output}!\n", fg="green")
    click.echo(yaml.safe_dump(config, sort_keys=False))

@cli.command()
@click.argument('config', type=click.Path(exists=True))
def deploy(config):
    """Deploy infrastructure as defined in the config file."""
    click.secho("Starting deployment...", fg="green")
    engine = UnifiedEngine(config)
    results = asyncio.run(engine.deploy())
    click.echo(results)

@cli.command()
@click.option('--config', '-c', default="nexus-infra.yml", show_default=True, help="Path to config YAML")
def review(config):
    """Stub: Review infrastructure config for compliance and best practices."""
    click.secho(f"Reviewing config: {config}", fg="cyan")
    # TODO: Implement real review logic
    click.secho("Review complete. (This is a stub.)", fg="green")

@cli.command()
@click.option('--config', '-c', default="nexus-infra.yml", show_default=True, help="Path to config YAML")
def cost_estimate(config):
    """Stub: Estimate and compare cloud costs for the given config."""
    click.secho(f"Estimating costs for config: {config}", fg="cyan")
    # TODO: Integrate with cloud provider cost APIs
    click.secho("AWS: $1,200/mo\nGCP: $980/mo\n(Sample output, not real data)", fg="green")

@cli.command()
def completion():
    """Show instructions for enabling shell auto-completion."""
    click.echo("To enable shell auto-completion, run one of the following commands in your shell:\n")
    click.echo("  # Bash")
    click.echo("  eval \"$(_NEXUS_CLI_COMPLETE=source_bash python cli/nexus_cli.py)\"")
    click.echo("\n  # Zsh")
    click.echo("  eval \"$(_NEXUS_CLI_COMPLETE=source_zsh python cli/nexus_cli.py)\"")
    click.echo("\n  # Fish")
    click.echo("  eval (env _NEXUS_CLI_COMPLETE=source_fish python cli/nexus_cli.py)")
    click.echo("\nFor permanent setup, add the appropriate command to your shell profile (e.g., .bashrc, .zshrc).")

class UnifiedProvider:
    """Base class for all provider plugins."""
    def __init__(self, config):
        self.config = config

    def deploy(self):
        """Deploy resources for this provider."""
        raise NotImplementedError("Provider must implement deploy()")

@cli.command()
@click.argument('config', type=click.Path(exists=True))
@click.option('--remediate', is_flag=True, help="Automatically remediate detected drift.")
def drift_detect(config, remediate):
    """Detect and optionally auto-remediate infrastructure drift."""
    click.secho("Drift detection started...", fg="yellow")
    engine = UnifiedEngine(config)
    drift_report = asyncio.run(engine.detect_drift())
    if any(drift_report.values()):
        click.secho("Drift detected:", fg="red")
        click.echo(json.dumps(drift_report, indent=2))
        if remediate:
            click.secho("Auto-remediation started...", fg="yellow")
            remediation_report = asyncio.run(engine.remediate_drift())
            click.secho("Remediation complete:", fg="green")
            click.echo(json.dumps(remediation_report, indent=2))
    else:
        click.secho("No drift detected.", fg="green")

@cli.command()
@click.argument('prompt', type=str)
@click.option('--output', '-o', default="nexus-infra.yml", show_default=True, help="Output YAML file name")
@click.option('--model', '-m', default="gemini-pro", help="AI model to use for generation (e.g., 'gpt-4', 'gemini-pro', 'openrouter/mistralai/mistral-7b-instruct')")
@click.option('--gemini-api-key', envvar='GEMINI_API_KEY', help="Gemini API Key (can be set via GEMINI_API_KEY environment variable)")
@click.option('--openai-api-key', envvar='OPENAI_API_KEY', help="OpenAI API Key (can be set via OPENAI_API_KEY environment variable)")
@click.option('--openrouter-api-key', envvar='OPENROUTER_API_KEY', help="OpenRouter API Key (can be set via OPENROUTER_API_KEY environment variable)")
def generate_nl(prompt, output, model, gemini_api_key, openai_api_key, openrouter_api_key):
    """Generate infrastructure config from natural language prompt."""
    click.secho(f"Processing natural language request using {model}...", fg="cyan")

    if model == "gemini-pro" and not gemini_api_key:
        click.secho("Error: Gemini API Key not provided. Please set --gemini-api-key or GEMINI_API_KEY environment variable.", fg="red")
        sys.exit(1)
    if model == "gpt-4" and not openai_api_key:
        click.secho("Error: OpenAI API Key not provided. Please set --openai-api-key or OPENAI_API_KEY environment variable.", fg="red")
        sys.exit(1)
    if model.startswith("openrouter/") and not openrouter_api_key:
        click.secho("Error: OpenRouter API Key not provided. Please set --openrouter-api-key or OPENROUTER_API_KEY environment variable.", fg="red")
        sys.exit(1)

    ai_optimizer = AIOptimizer(openai_api_key=openai_api_key, gemini_api_key=gemini_api_key, openrouter_api_key=openrouter_api_key)

    try:
        config = asyncio.run(ai_optimizer.generate_infrastructure(prompt, model_name=model))
        with open(output, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        click.secho(f"\nGenerated {output} from prompt: {prompt}\n", fg="green")
        click.echo(yaml.safe_dump(config, sort_keys=False))
    except Exception as e:
        click.secho(f"Error generating infrastructure: {str(e)}", fg="red")
        sys.exit(1)

if __name__ == "__main__":
    cli()