"""NSE India "Operator Intent" daily screener.

Combines NSE Cash Bhavcopy (price + delivery + VWAP) with the F&O UDiff
Bhavcopy (per-expiry Open Interest) and labels each F&O stock with one of:
Long Build-up, Short Covering, Short Build-up, Long Unwinding — plus a
High Momentum Watch flag for Long Build-ups within 15% of the 52-week high.
"""
