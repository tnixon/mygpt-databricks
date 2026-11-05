-- The fact table behind most widgets: one row per day x component (filters apply everywhere).
SELECT usage_date, component, category, lifecycle, allocation, native_unit,
       native_qty, dbus, list_usd, list_usd * (1 - :discount_pct / 100) AS net_usd
FROM cost_rows
ORDER BY usage_date
