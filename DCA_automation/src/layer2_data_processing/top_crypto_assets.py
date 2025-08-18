#!/usr/bin/env python3
"""
Top Cryptocurrency Assets by Market Cap

This module contains the list of top 150 cryptocurrencies by market capitalization
that we should focus our analysis on. This is a practical approach used by real
portfolio managers to avoid computational overhead while still capturing the most
important market segments.

Data based on market cap rankings (updated periodically).
"""

# Top 150 Cryptocurrencies by Market Cap (as of late 2024/early 2025)
TOP_150_CRYPTO_ASSETS = [
    # Top 10 - Major cryptocurrencies
    'BTC', 'ETH', 'USDT', 'BNB', 'USDC', 'XRP', 'SOL', 'DOGE', 'ADA', 'STETH',
    
    # 11-30 - Large cap altcoins
    'TRX', 'AVAX', 'TON', 'SHIB', 'DOT', 'WBTC', 'LINK', 'BCH', 'SUI', 'NEAR',
    'UNI', 'LTC', 'PEPE', 'DAI', 'ICP', 'APT', 'HEX', 'WIF', 'POL', 'ARB',
    
    # 31-60 - Mid-large cap
    'FET', 'CRO', 'ETC', 'MNT', 'OP', 'IMX', 'ATOM', 'OKB', 'HBAR', 'BONK',
    'VET', 'FIL', 'MKR', 'RNDR', 'ALGO', 'XMR', 'OM', 'TAO', 'ONDO', 'XLM',
    'INJ', 'FDUSD', 'GRT', 'KAS', 'MANA', 'AAVE', 'TIA', 'STX', 'JUP', 'SEI',
    
    # 61-100 - Mid cap
    'LDO', 'PYTH', 'THETA', 'AR', 'RUNE', 'ROSE', 'FTM', 'SAND', 'AXS', 'EGLD',
    'BLUR', 'BEAM', 'FLOW', 'KCS', 'MEME', 'KAVA', 'PENDLE', 'ZRO', 'WLD', 'JASMY',
    'ENS', 'EIGEN', 'GMX', 'FLOKI', 'AIOZ', 'NOT', 'MATIC', 'COMP', 'GRASS', 'STRK',
    'QNT', 'ORDI', 'SAFE', 'SNX', 'CAKE', 'GALA', 'CHZ', 'ETHFI', 'KSM', 'ENA',
    
    # 101-150 - Smaller mid cap but still significant
    'APE', 'OSMO', 'DEXE', 'SYN', 'ALPHA', 'PIXEL', 'CORE', 'PRIME', 'SUPER', 'REZ',
    'IO', 'MASK', 'ALCX', 'TRB', 'CRV', 'JST', 'ARKM', 'BICO', 'STORJ', 'AUDIO',
    'CTSI', 'BAL', 'RLC', 'LRC', 'ANKR', 'BAND', 'SLP', 'HOT', 'ENJ', 'WIN',
    'BTT', 'CELR', 'SKL', 'DUSK', 'POWR', 'GLM', 'REQ', 'UMA', 'NMR', 'ZRX',
    'LQTY', 'GODS', 'ALICE', 'YFI', 'CVX', 'SUSHI', 'BNT', 'REN', 'RARE', 'OXT'
]

def get_top_crypto_assets():
    """Return the list of top 150 cryptocurrency assets by market cap"""
    return TOP_150_CRYPTO_ASSETS

def filter_assets_by_market_cap(available_assets):
    """
    Filter available assets to only include those in the top 150 by market cap
    
    Args:
        available_assets: List of asset symbols available in the data
        
    Returns:
        List of asset symbols that are in both available_assets and top 150
    """
    top_assets_set = set(TOP_150_CRYPTO_ASSETS)
    available_assets_set = set(available_assets)
    
    # Find intersection - assets that are both available and in top 150
    filtered_assets = list(top_assets_set.intersection(available_assets_set))
    
    return filtered_assets

def get_asset_tier(asset_symbol):
    """
    Get the market cap tier for an asset
    
    Args:
        asset_symbol: Asset symbol (e.g., 'BTC')
        
    Returns:
        String indicating tier: 'tier1' (1-10), 'tier2' (11-30), 'tier3' (31-60), etc.
    """
    if asset_symbol not in TOP_150_CRYPTO_ASSETS:
        return 'not_tracked'
    
    index = TOP_150_CRYPTO_ASSETS.index(asset_symbol)
    
    if index < 10:
        return 'tier1_major'
    elif index < 30:
        return 'tier2_large'
    elif index < 60:
        return 'tier3_mid_large'
    elif index < 100:
        return 'tier4_mid'
    else:
        return 'tier5_smaller'
