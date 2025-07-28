#!/usr/bin/env python3
"""
Configuration Management CLI for NIDS Autoencoder.

This script provides command-line tools for managing configuration:
- Generate environment-specific configurations
- Validate configuration files
- Convert between configuration formats
- Update configuration values
"""

import argparse
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from utils.enterprise_config import (
        ConfigurationManager, 
        ConfigurationError,
        Environment,
        ConfigFormat
    )
    from utils.config_validation import validate_config_dict
except ImportError as e:
    print(f"❌ Failed to import required modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


def generate_configs(output_dir: Path, base_config: Optional[Path] = None) -> None:
    """Generate environment-specific configuration files.
    
    Args:
        output_dir: Directory to output configuration files
        base_config: Base configuration file to use as template
    """
    print(f"🔧 Generating environment configurations in: {output_dir}")
    
    try:
        # Initialize configuration manager
        if base_config and base_config.exists():
            config_manager = ConfigurationManager(config_path=base_config)
        else:
            config_manager = ConfigurationManager()
        
        # Create environment-specific configs
        config_manager.create_environment_configs(output_dir)
        
        # Also create a master config template
        master_config_path = output_dir / "nids_config_template.yaml"
        config_manager.save_configuration(master_config_path)
        
        print("✅ Configuration files generated successfully:")
        for env in Environment:
            env_file = output_dir / f"{env.value}.yaml"
            if env_file.exists():
                print(f"   📄 {env_file}")
        print(f"   📄 {master_config_path}")
        
    except Exception as e:
        print(f"❌ Failed to generate configurations: {e}")
        sys.exit(1)


def validate_config(config_file: Path) -> None:
    """Validate a configuration file.
    
    Args:
        config_file: Path to configuration file to validate
    """
    print(f"🔍 Validating configuration: {config_file}")
    
    try:
        # Load configuration
        config_manager = ConfigurationManager(config_path=config_file)
        
        # Get configuration as dictionary for validation
        config_dict = config_manager.config.__dict__
        
        # Convert dataclass instances to dictionaries
        def convert_to_dict(obj):
            if hasattr(obj, '__dict__'):
                return {k: convert_to_dict(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, list):
                return [convert_to_dict(item) for item in obj]
            else:
                return obj
        
        config_dict = convert_to_dict(config_dict)
        
        # Validate using schema
        is_valid, errors = validate_config_dict(config_dict)
        
        if is_valid:
            print("✅ Configuration is valid!")
            
            # Show configuration summary
            summary = config_manager.get_config_summary()
            print("\n📊 Configuration Summary:")
            for section, data in summary.items():
                print(f"   {section}: {data}")
                
        else:
            print("❌ Configuration validation failed:")
            for error in errors:
                print(f"   • {error}")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Failed to validate configuration: {e}")
        sys.exit(1)


def convert_config(input_file: Path, output_file: Path) -> None:
    """Convert configuration between formats.
    
    Args:
        input_file: Input configuration file
        output_file: Output configuration file
    """
    print(f"🔄 Converting {input_file} → {output_file}")
    
    try:
        # Load configuration
        config_manager = ConfigurationManager(config_path=input_file)
        
        # Determine output format
        if output_file.suffix.lower() in ['.yaml', '.yml']:
            format_type = ConfigFormat.YAML
        elif output_file.suffix.lower() == '.json':
            format_type = ConfigFormat.JSON
        else:
            raise ValueError(f"Unsupported output format: {output_file.suffix}")
        
        # Save in new format
        config_manager.save_configuration(output_file, format_type)
        
        print("✅ Configuration converted successfully!")
        
    except Exception as e:
        print(f"❌ Failed to convert configuration: {e}")
        sys.exit(1)


def update_config(config_file: Path, updates: Dict[str, Any]) -> None:
    """Update configuration values.
    
    Args:
        config_file: Configuration file to update
        updates: Dictionary of updates to apply
    """
    print(f"✏️  Updating configuration: {config_file}")
    
    try:
        # Load configuration
        config_manager = ConfigurationManager(config_path=config_file)
        
        # Apply updates
        for key, value in updates.items():
            section, param = key.split('.', 1)
            if hasattr(config_manager.config, section):
                section_obj = getattr(config_manager.config, section)
                if hasattr(section_obj, param):
                    # Convert string values to appropriate types
                    current_value = getattr(section_obj, param)
                    if isinstance(current_value, bool):
                        value = value.lower() in ('true', '1', 'yes', 'on')
                    elif isinstance(current_value, int):
                        value = int(value)
                    elif isinstance(current_value, float):
                        value = float(value)
                    elif isinstance(current_value, list):
                        if isinstance(value, str):
                            value = [x.strip() for x in value.split(',')]
                    
                    setattr(section_obj, param, value)
                    print(f"   ✓ {key} = {value}")
                else:
                    print(f"   ❌ Unknown parameter: {key}")
            else:
                print(f"   ❌ Unknown section: {section}")
        
        # Save updated configuration
        config_manager.save_configuration(config_file)
        
        print("✅ Configuration updated successfully!")
        
    except Exception as e:
        print(f"❌ Failed to update configuration: {e}")
        sys.exit(1)


def show_config(config_file: Path, section: Optional[str] = None) -> None:
    """Show configuration contents.
    
    Args:
        config_file: Configuration file to show
        section: Specific section to show (optional)
    """
    try:
        # Load configuration
        config_manager = ConfigurationManager(config_path=config_file)
        
        if section:
            # Show specific section
            if hasattr(config_manager.config, section):
                section_obj = getattr(config_manager.config, section)
                print(f"📋 Configuration section '{section}':")
                for key, value in section_obj.__dict__.items():
                    print(f"   {key}: {value}")
            else:
                print(f"❌ Unknown section: {section}")
                sys.exit(1)
        else:
            # Show summary
            summary = config_manager.get_config_summary()
            print("📋 Configuration Summary:")
            for section_name, data in summary.items():
                print(f"\n{section_name}:")
                if isinstance(data, dict):
                    for key, value in data.items():
                        print(f"   {key}: {value}")
                else:
                    print(f"   {data}")
                    
    except Exception as e:
        print(f"❌ Failed to show configuration: {e}")
        sys.exit(1)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="NIDS Configuration Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate environment configurations
  python config_cli.py generate --output-dir config/environments

  # Validate a configuration file
  python config_cli.py validate --config config/production.yaml

  # Convert YAML to JSON
  python config_cli.py convert --input config.yaml --output config.json

  # Update configuration values
  python config_cli.py update --config config.yaml --set training.epochs=100 --set api.port=8080

  # Show configuration
  python config_cli.py show --config config.yaml --section training
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate environment configurations')
    generate_parser.add_argument(
        '--output-dir', 
        type=Path, 
        required=True,
        help='Output directory for configuration files'
    )
    generate_parser.add_argument(
        '--base-config', 
        type=Path,
        help='Base configuration file to use as template'
    )
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate configuration file')
    validate_parser.add_argument(
        '--config', 
        type=Path, 
        required=True,
        help='Configuration file to validate'
    )
    
    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert configuration format')
    convert_parser.add_argument(
        '--input', 
        type=Path, 
        required=True,
        help='Input configuration file'
    )
    convert_parser.add_argument(
        '--output', 
        type=Path, 
        required=True,
        help='Output configuration file'
    )
    
    # Update command
    update_parser = subparsers.add_parser('update', help='Update configuration values')
    update_parser.add_argument(
        '--config', 
        type=Path, 
        required=True,
        help='Configuration file to update'
    )
    update_parser.add_argument(
        '--set', 
        action='append', 
        dest='updates',
        help='Set configuration value (format: section.param=value)'
    )
    
    # Show command
    show_parser = subparsers.add_parser('show', help='Show configuration')
    show_parser.add_argument(
        '--config', 
        type=Path, 
        required=True,
        help='Configuration file to show'
    )
    show_parser.add_argument(
        '--section', 
        help='Specific section to show'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    try:
        if args.command == 'generate':
            generate_configs(args.output_dir, args.base_config)
        
        elif args.command == 'validate':
            validate_config(args.config)
        
        elif args.command == 'convert':
            convert_config(args.input, args.output)
        
        elif args.command == 'update':
            if not args.updates:
                print("❌ No updates specified. Use --set section.param=value")
                sys.exit(1)
            
            updates = {}
            for update in args.updates:
                if '=' not in update:
                    print(f"❌ Invalid update format: {update}")
                    sys.exit(1)
                key, value = update.split('=', 1)
                updates[key] = value
            
            update_config(args.config, updates)
        
        elif args.command == 'show':
            show_config(args.config, args.section)
    
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
