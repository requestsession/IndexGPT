export function pickDisplaySeries(logSeries, liveSeries) {
  if (Array.isArray(logSeries) && logSeries.length) return logSeries;
  if (Array.isArray(liveSeries) && liveSeries.length) return liveSeries;
  return [];
}
