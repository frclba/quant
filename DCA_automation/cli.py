#!/usr/bin/env python3
"""
DCA System Command Line Interface

Single CLI for the complete DCA automated system.
Orchestrates all layers through simple commands.

Usage:
    python cli.py run                    # Run complete pipeline
    python cli.py collect                # Data collection only  
    python cli.py process                # Data processing only
    python cli.py status                 # Show system status
    python cli.py --help                 # Show help
"""

import argparse
import asyncio
import json
from pathlib import Path
from datetime import datetime

from dca_system import DCASystem, SystemConfig, CollectionMode


def setup_parser():
    """Setup command line argument parser"""
    
    parser = argparse.ArgumentParser(
        description='DCA Automated System - Cryptocurrency Investment System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py run                    # Run complete DCA pipeline
  python cli.py collect --full         # Full data collection
  python cli.py collect --incremental  # Incremental update (default)
  python cli.py process                # Process existing data
  python cli.py status                 # Show system status
  python cli.py run --no-layer2        # Data collection only
        """
    )
    
    # Commands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run command (complete pipeline)
    run_parser = subparsers.add_parser('run', help='Run complete DCA pipeline')
    run_parser.add_argument('--mode', choices=['incremental', 'full'], default='incremental',
                           help='Data collection mode (default: incremental)')
    run_parser.add_argument('--no-layer2', action='store_true',
                           help='Skip Layer 2 processing')
    run_parser.add_argument('--max-features', type=int, default=100,
                           help='Maximum features per asset (default: 100)')
    
    # Collect command (Layer 1 only)  
    collect_parser = subparsers.add_parser('collect', help='Run data collection only')
    collect_parser.add_argument('--mode', choices=['incremental', 'full'], default='incremental',
                               help='Collection mode (default: incremental)')
    
    # Process command (Layer 2 only)
    process_parser = subparsers.add_parser('process', help='Run data processing only')
    process_parser.add_argument('--max-features', type=int, default=100,
                               help='Maximum features per asset (default: 100)')
    process_parser.add_argument('--no-feature-selection', action='store_true',
                               help='Disable feature selection')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show system status')
    status_parser.add_argument('--json', action='store_true',
                              help='Output status as JSON')
    
    # Global options
    parser.add_argument('--data-path', default='data',
                       help='Path to data directory (default: data)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level (default: INFO)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results to disk')
    
    return parser


async def run_command(args):
    """Execute run command"""
    print("🚀 DCA System - Complete Pipeline")
    print("=" * 40)
    
    # Setup configuration
    config = SystemConfig(
        data_path=args.data_path,
        collection_mode=CollectionMode.FULL_REFRESH if args.mode == 'full' else CollectionMode.INCREMENTAL,
        enable_layer2=not args.no_layer2,
        max_features_per_asset=args.max_features,
        log_level=args.log_level,
        save_results=not args.no_save
    )
    
    # Run system
    system = DCASystem(config)
    results = await system.run_complete_pipeline()
    
    # Display results
    if 'error' in results:
        print(f"❌ Pipeline failed: {results['error']}")
        return 1
    
    print(f"\n✅ Pipeline completed successfully!")
    
    # Show Layer 1 results
    if results.get('layer1_results'):
        layer1_stats = results['layer1_results']
        print(f"\n📊 Layer 1 - Data Collection:")
        print(f"  • Assets collected: {getattr(layer1_stats, 'symbols_successful', 0)}")
        print(f"  • Data points: {getattr(layer1_stats, 'total_data_points', 0):,}")
        print(f"  • Collection mode: {args.mode}")
    
    # Show Layer 2 results
    if results.get('layer2_results'):
        layer2_stats = results['layer2_results']
        print(f"\n🔄 Layer 2 - Data Processing:")
        print(f"  • Features created: {layer2_stats.processing_stats.get('features_created', 0):,}")
        print(f"  • Assets processed: {layer2_stats.processing_stats.get('assets_processed', 0)}")
        print(f"  • Processing time: {layer2_stats.processing_stats.get('duration', 'N/A')}")
    
    # Show pipeline stats
    if results.get('pipeline_stats'):
        stats = results['pipeline_stats']
        print(f"\n⏱️  Pipeline Statistics:")
        print(f"  • Total duration: {stats.get('total_duration', 'N/A')}")
        print(f"  • Layers executed: {', '.join(stats.get('layers_executed', []))}")
    
    print(f"\n🎯 DCA System ready for next phase!")
    return 0


async def collect_command(args):
    """Execute collect command"""
    print("📊 DCA System - Data Collection")
    print("=" * 35)
    
    config = SystemConfig(
        data_path=args.data_path,
        collection_mode=CollectionMode.FULL_REFRESH if args.mode == 'full' else CollectionMode.INCREMENTAL,
        enable_layer2=False,
        log_level=args.log_level,
        save_results=not args.no_save
    )
    
    system = DCASystem(config)
    results = await system.run_data_collection_only()
    
    if 'error' in results:
        print(f"❌ Data collection failed: {results['error']}")
        return 1
    
    layer1_stats = results['layer1_results']
    print(f"\n✅ Data collection completed!")
    print(f"  • Assets collected: {getattr(layer1_stats, 'symbols_successful', 0)}")
    print(f"  • Failed: {getattr(layer1_stats, 'symbols_failed', 0)}")
    print(f"  • Up to date: {getattr(layer1_stats, 'symbols_up_to_date', 0)}")
    print(f"  • Data points: {getattr(layer1_stats, 'total_data_points', 0):,}")
    print(f"  • Duration: {getattr(layer1_stats, 'duration', 'N/A')}")
    
    return 0


async def process_command(args):
    """Execute process command"""
    print("🔄 DCA System - Data Processing")
    print("=" * 35)
    
    config = SystemConfig(
        data_path=args.data_path,
        enable_layer2=True,
        feature_selection=not args.no_feature_selection,
        max_features_per_asset=args.max_features,
        log_level=args.log_level,
        save_results=not args.no_save
    )
    
    system = DCASystem(config)
    results = await system.run_processing_only()
    
    if 'error' in results:
        print(f"❌ Data processing failed: {results['error']}")
        return 1
    
    layer2_stats = results['layer2_results']
    print(f"\n✅ Data processing completed!")
    print(f"  • Features created: {layer2_stats.processing_stats.get('features_created', 0):,}")
    print(f"  • Assets processed: {layer2_stats.processing_stats.get('assets_processed', 0)}")
    print(f"  • Memory usage: {layer2_stats.processing_stats.get('memory_usage_mb', 0):.1f} MB")
    print(f"  • Processing time: {layer2_stats.processing_stats.get('duration', 'N/A')}")
    
    # Show top asset scores if available
    if hasattr(layer2_stats, 'asset_scores_preview') and layer2_stats.asset_scores_preview:
        print(f"\n🏆 Top Asset Scores:")
        sorted_scores = sorted(layer2_stats.asset_scores_preview.items(), 
                             key=lambda x: x[1], reverse=True)
        for i, (asset, score) in enumerate(sorted_scores[:5]):
            print(f"  {i+1}. {asset}: {score:.6f}")
    
    return 0


def status_command(args):
    """Execute status command"""
    
    config = SystemConfig(data_path=args.data_path, log_level=args.log_level)
    system = DCASystem(config)
    status = system.get_system_status()
    
    if args.json:
        print(json.dumps(status, indent=2, default=str))
        return 0
    
    print("📊 DCA System Status")
    print("=" * 25)
    
    # System configuration
    print(f"📁 Data path: {status['system_config']['data_path']}")
    print(f"📁 Data path exists: {'✅' if status['data_path_exists'] else '❌'}")
    
    # Data availability
    if status.get('data_available'):
        data_info = status['data_available']
        print(f"\n📈 Available Data:")
        print(f"  • Altcoins: {data_info['altcoin_files']} files")
        print(f"  • Major cryptos: {data_info['major_crypto_files']} files")
        print(f"  • Total assets: {data_info['total_assets']}")
    else:
        print(f"\n📈 Available Data: None (data path does not exist)")
    
    # Layer availability  
    print(f"\n🏗️  Layer Availability:")
    for layer, available in status['layers_available'].items():
        status_icon = "✅" if available else "⏳"
        layer_num = layer.replace('layer', 'Layer ')
        print(f"  • {layer_num}: {status_icon}")
    
    # System configuration
    print(f"\n⚙️  System Configuration:")
    for key, value in status['system_config'].items():
        if key not in ['data_path']:  # Already shown above
            print(f"  • {key}: {value}")
    
    return 0


def main():
    """Main CLI entry point"""
    
    parser = setup_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'run':
            return asyncio.run(run_command(args))
        elif args.command == 'collect':
            return asyncio.run(collect_command(args))
        elif args.command == 'process':
            return asyncio.run(process_command(args))
        elif args.command == 'status':
            return status_command(args)
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
        return 130
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
