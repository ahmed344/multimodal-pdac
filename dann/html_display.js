/* Scientific display formatting; numerical arrays always retain original values. */
function formatNumber(value, kind = 'float') {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return 'Unavailable';
  value = Number(value);
  if (kind === 'integer') return String(Math.trunc(value));
  if (kind === 'mz') return value.toFixed(4);
  if (value === 0) return '0';
  const magnitude = Math.abs(value);
  if (magnitude < .0001) return value.toExponential(2).replace(/e([+-])(\d)$/, 'e$10$2');
  return value.toFixed(magnitude >= 1 ? 2 : magnitude >= .01 ? 3 : 4);
}
