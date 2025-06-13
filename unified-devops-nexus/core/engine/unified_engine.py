class UnifiedEngine:
    """Unified Engine for managing infrastructure deployments."""

    def __init__(self, config_path):
        self.config_path = config_path
        self.providers = {}

    def initialize_providers(self):
        """Initialize providers based on the configuration."""
        import yaml

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        for provider_name, provider_config in config.get('providers', {}).items():
            if provider_config.get('enabled'):
                provider_class = self._get_provider_class(provider_name)
                if provider_class:
                    self.providers[provider_name] = provider_class(provider_config)

    def _get_provider_class(self, provider_name):
        """Dynamically load the provider class based on the provider name."""
        module_name = f"core.plugins.{provider_name}_provider"
        class_name = f"{provider_name.capitalize()}Provider"
        try:
            module = __import__(module_name, fromlist=[class_name])
            return getattr(module, class_name)
        except (ImportError, AttributeError):
            return None

    async def deploy(self):
        """Deploy resources using the initialized providers."""
        results = {}
        for provider_name, provider in self.providers.items():
            results[provider_name] = await provider.deploy()
        return results